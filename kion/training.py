"""Validation-only model selection. Final test execution is a separate stage."""
import gc
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


def model_pool(frame, columns, categorical):
    return Pool(frame[columns], label=frame.label, group_id=frame.query_id, cat_features=categorical)


def assert_validation_protocol(train, validation, cfg):
    if set(train.cutoff.unique()) - set(cfg.train_cutoffs):
        raise ValueError("Training contains an unapproved cutoff")
    if set(validation.cutoff.unique()) != {cfg.validation_cutoff}:
        raise ValueError("Model selection accepts only the validation cutoff, never test")
    if train.label.nunique() < 2:
        raise ValueError("Training requires positive and negative candidates")


def fit_grid(train, validation, targets, cfg, out):
    assert_validation_protocol(train, validation, cfg)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    columns = feature_columns(train)
    categorical = categorical_columns(train, columns)
    train_pool = model_pool(train, columns, categorical)
    val_pool = model_pool(validation, columns, categorical)
    results = []
    best_score = -np.inf
    best = None
    for depth in cfg.depths:
        for l2 in cfg.l2_values:
            name = f"depth{depth}_l2{l2}"
            print(f"Training {name}: {len(train):,} rows, {len(columns)} features", flush=True)
            start = time.perf_counter()
            model = CatBoostRanker(loss_function="YetiRank", eval_metric="NDCG:top=10", iterations=cfg.iterations,
                depth=depth, l2_leaf_reg=l2, learning_rate=cfg.learning_rate, random_seed=cfg.seed,
                thread_count=cfg.threads, allow_writing_files=False, used_ram_limit="6gb", max_ctr_complexity=1)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=cfg.early_stopping, verbose=100)
            scored = validation[["user_id", "item_id"]].copy()
            scored["score"] = model.predict(val_pool)
            metrics, per_user = evaluate_scores(scored, targets, "score")
            record = {"name": name, "depth": depth, "l2_leaf_reg": l2, "iterations": model.tree_count_,
                      "seconds": time.perf_counter() - start, "metrics": metrics}
            results.append(record)
            model.save_model(str(out / f"{name}.cbm"))
            per_user.to_parquet(out / f"{name}_per_user.parquet")
            print(f"Validation {name}: {metrics}", flush=True)
            write_json(out / "grid_progress.json", results)
            if metrics["ndcg@10"] > best_score:
                best_score = metrics["ndcg@10"]
                best = record.copy()
                shutil.copyfile(out / f"{name}.cbm", out / "selected_model.cbm")
            del model, scored
            gc.collect()
    del train_pool, val_pool
    gc.collect()
    chosen = CatBoostRanker(); chosen.load_model(str(out / "selected_model.cbm"))
    importances = chosen.get_feature_importance(type="PredictionValuesChange")
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
    model.fit(model_pool(train, columns, cats), verbose=100)
    frame = validation[["user_id", "item_id"]].copy()
    frame["score"] = model.predict(Pool(validation[columns], cat_features=cats))
    metrics, per_user = evaluate_scores(frame, targets, "score")
    model.save_model(str(Path(out) / "ablation_model.cbm"))
    per_user.to_parquet(Path(out) / "ablation_per_user.parquet")
    result = {"metrics": metrics, "features": columns, "seconds": time.perf_counter() - start,
              "note": "Personalized candidates are unchanged; only personal reranking features are removed."}
    write_json(Path(out) / "ablation.json", result)
    return result
