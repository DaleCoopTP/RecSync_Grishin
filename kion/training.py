"""Validation-only model selection. Final test execution is a separate stage."""
import gc
import hashlib
import json
import importlib.metadata
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from .dataset import feature_columns, categorical_columns
from .metrics import evaluate_scores
from .io import write_json, sha256


def model_pool(frame, columns, categorical, threads=1):
    return Pool(frame[columns], label=frame.label, group_id=frame.query_id, cat_features=categorical, thread_count=threads)


def assert_validation_protocol(train, validation, cfg):
    if set(train.cutoff.unique()) - set(cfg.train_cutoffs):
        raise ValueError("Training contains an unapproved cutoff")
    if set(validation.cutoff.unique()) != {cfg.validation_cutoff}:
        raise ValueError("Model selection accepts only the validation cutoff, never test")
    if train.label.nunique() < 2:
        raise ValueError("Training requires positive and negative candidates")


def reuse_prefix(path, columns, categorical, cfg, depth, l2):
    """Select using only the allowed history prefix; preserve the original checkpoint."""
    model = CatBoostRanker(); model.load_model(str(path))
    params = model.get_all_params()
    expected = {"loss_function": "YetiRank", "depth": depth, "l2_leaf_reg": l2,
                "learning_rate": cfg.learning_rate, "random_seed": cfg.seed, "max_ctr_complexity": 1}
    for key, value in expected.items():
        actual = params.get(key)
        if isinstance(value, (float, int)):
            valid = actual is not None and np.isclose(actual, value, rtol=1e-6)
        else:
            valid = actual == value
        if not valid:
            raise ValueError(f"Prefix checkpoint parameter mismatch: {key}")
    if model.feature_names_ != columns or model.get_cat_feature_indices() != [columns.index(c) for c in categorical]:
        raise ValueError("Prefix checkpoint feature mismatch")
    history = model.get_evals_result().get("validation", {})
    curve = next((v for k, v in history.items() if k.startswith("NDCG")), None)
    if not curve:
        raise ValueError("Prefix checkpoint has no validation learning curve")
    allowed = curve[:cfg.iterations]
    best = 0
    used = 0
    for i, score in enumerate(allowed):
        used = i + 1
        if score > allowed[best]:
            best = i
        if i - best >= cfg.early_stopping:
            break
    if best + 1 > model.tree_count_:
        raise ValueError("Required prefix trees were not saved in the source model")
    provenance = {"source_checkpoint": str(Path(path).resolve()), "source_sha256": sha256(path),
                  "source_trained_iterations": len(curve), "source_saved_trees": model.tree_count_,
                  "source_threads": json.loads(model.get_metadata()["params"])["flat_params"].get("thread_count"), "prefix_budget": cfg.iterations,
                  "note": "Only the allowed prefix of validation history selects trees; later trees and metrics are excluded."}
    model.shrink(ntree_end=best + 1)
    return model, used, provenance


def fit_grid(train, validation, targets, cfg, out, prefix_source=None, input_signature=None):
    assert_validation_protocol(train, validation, cfg)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    columns = feature_columns(train)
    categorical = categorical_columns(train, columns)
    if input_signature is None:
        digest = hashlib.sha256()
        for frame in [train, validation]:
            digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
        digest.update(json.dumps({int(k): sorted(v) for k, v in targets.items()}, sort_keys=True).encode())
        input_signature = digest.hexdigest()
    identity = {"config": cfg.to_dict(), "inputs": input_signature, "features": columns, "categorical": categorical}
    identity_path = out / "grid_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Cannot resume grid with different data, features or parameters")
    write_json(identity_path, identity)
    progress_path = out / "grid_progress.json"
    completed = {r["name"]: r for r in json.loads(progress_path.read_text())} if progress_path.exists() else {}
    train_pool = model_pool(train, columns, categorical, cfg.threads)
    val_pool = model_pool(validation, columns, categorical, cfg.threads)
    results = []
    best_score = -np.inf
    best = None
    for depth in cfg.depths:
        for l2 in cfg.l2_values:
            name = f"depth{depth}_l2{l2}"
            checkpoint = out / f"{name}.cbm"
            if name in completed:
                record = completed[name]
                if sha256(checkpoint) != record["model_sha256"]:
                    raise ValueError("Completed grid checkpoint checksum mismatch")
                print(f"Resuming completed {name}", flush=True)
            else:
                print(f"Training {name}: {len(train):,} rows, {len(columns)} features, threads={cfg.threads}, budget={cfg.iterations}", flush=True)
                start = time.perf_counter()
                source = Path(prefix_source) / "models" / f"{name}.cbm" if prefix_source else None
                provenance = None
                if source is not None and source.exists():
                    model, trained_iterations, provenance = reuse_prefix(source, columns, categorical, cfg, depth, l2)
                    print(f"Reusing verified prefix: {model.tree_count_} saved trees, {trained_iterations} allowed iterations", flush=True)
                    old_progress = json.loads((Path(prefix_source) / "models" / "grid_progress.json").read_text())
                    provenance["original_exploratory_result"] = next(r for r in old_progress if r["name"] == name)
                else:
                    model = CatBoostRanker(loss_function="YetiRank", eval_metric="NDCG:top=10", iterations=cfg.iterations,
                        depth=depth, l2_leaf_reg=l2, learning_rate=cfg.learning_rate, random_seed=cfg.seed,
                        thread_count=cfg.threads, allow_writing_files=False, used_ram_limit="6gb", max_ctr_complexity=1)
                    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=cfg.early_stopping, verbose=50)
                    trained_iterations = max(len(v) for v in model.get_evals_result()["validation"].values())
                scored = validation[["user_id", "item_id"]].copy()
                scored["score"] = model.predict(val_pool, thread_count=cfg.threads)
                metrics, per_user = evaluate_scores(scored, targets, "score")
                record = {"name": name, "depth": depth, "l2_leaf_reg": l2, "iterations": model.tree_count_,
                          "trained_iterations": trained_iterations, "seconds": time.perf_counter() - start, "metrics": metrics,
                          "prefix_reuse": provenance}
                model.save_model(str(checkpoint))
                record["model_sha256"] = sha256(checkpoint)
                per_user.to_parquet(out / f"{name}_per_user.parquet")
                print(f"Validation {name}: {metrics}", flush=True)
                del model, scored
                gc.collect()
            results.append(record)
            completed[name] = record
            write_json(progress_path, list(completed.values()))
            if record["metrics"]["ndcg@10"] > best_score:
                best_score = record["metrics"]["ndcg@10"]
                best = record.copy()
                shutil.copyfile(checkpoint, out / "selected_model.cbm")
    del train_pool, val_pool
    gc.collect()
    chosen = CatBoostRanker(); chosen.load_model(str(out / "selected_model.cbm"))
    importances = chosen.get_feature_importance(type="PredictionValuesChange", thread_count=cfg.threads)
    pd.DataFrame({"feature": columns, "importance": importances}).sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    manifest = {"selected": best, "grid": results, "features": columns, "categorical_features": categorical,
                "config": cfg.to_dict(), "prediction_cutoff": cfg.validation_cutoff,
                "training_cutoffs": cfg.train_cutoffs, "test_evaluated": False,
                "model_sha256": sha256(out / "selected_model.cbm"), "python": platform.python_version(),
                "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "catboost", "pyarrow"]},
                "selection_note": "Early stopping uses candidate-normalized NDCG; configuration selection uses full-ground-truth NDCG@10."}
    write_json(out / "model_manifest.json", manifest)
    return manifest


def fit_ablation(train, validation, targets, cfg, manifest, out):
    columns = feature_columns(train, ablate=True)
    cats = categorical_columns(train, columns)
    params = manifest["selected"]
    model = CatBoostRanker(loss_function="YetiRank", iterations=params["iterations"], depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"], learning_rate=cfg.learning_rate, random_seed=cfg.seed,
        thread_count=cfg.threads, allow_writing_files=False, used_ram_limit="6gb", max_ctr_complexity=1)
    print(f"Training ablation without personal ranking features: {len(columns)} features", flush=True)
    start = time.perf_counter()
    model.fit(model_pool(train, columns, cats, cfg.threads), verbose=100)
    frame = validation[["user_id", "item_id"]].copy()
    frame["score"] = model.predict(Pool(validation[columns], cat_features=cats, thread_count=cfg.threads), thread_count=cfg.threads)
    metrics, per_user = evaluate_scores(frame, targets, "score")
    model.save_model(str(Path(out) / "ablation_model.cbm"))
    per_user.to_parquet(Path(out) / "ablation_per_user.parquet")
    result = {"metrics": metrics, "features": columns, "seconds": time.perf_counter() - start,
              "note": "Personalized candidates are unchanged; only personal reranking features are removed."}
    write_json(Path(out) / "ablation.json", result)
    return result
