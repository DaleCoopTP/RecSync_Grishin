"""Bounded-memory snapshot caching and validation-only orchestration."""
from dataclasses import replace
import gc
import hashlib
import json
import importlib.metadata
from pathlib import Path
import platform
import resource
import subprocess
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostRanker, Pool
from .config import Config
from .data import SOURCE_COMMIT, ARCHIVE_SHA256
from .dataset import iter_dataset, KEY_COLUMNS
from .features import FeatureStore
from .candidates import CandidateRetriever
from .io import sha256, write_json
from .metrics import evaluate_baselines, evaluate_scores
from .protocol import make_snapshot, prepare, load_prepared, prepared_dir
from .training import fit_grid, fit_ablation


def smoke_config(cfg):
    return replace(cfg, sample_users=1000, train_users_per_cutoff=200, iterations=20,
                   early_stopping=5, depths=[4], l2_values=[3], threads=2,
                   artifact_dir=str(Path(cfg.artifact_dir) / "smoke"), results_dir=str(Path(cfg.results_dir) / "smoke"))


def code_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def context(cfg, tables):
    digest = hashlib.sha256()
    for table in tables:
        digest.update(pd.util.hash_pandas_object(table, index=False).values.tobytes())
    data_hash = digest.hexdigest()
    code_hash = code_fingerprint()
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "catboost", "pyarrow"]}
    key = hashlib.sha256((cfg.fingerprint() + data_hash + code_hash + json.dumps(versions, sort_keys=True) + platform.python_version()).encode()).hexdigest()[:20]
    out = cfg.path("artifact_dir") / key
    out.mkdir(parents=True, exist_ok=True)
    return out, {"run_id": key, "data_fingerprint": data_hash, "code_fingerprint": code_hash,
                 "config_fingerprint": cfg.fingerprint(), "config": cfg.to_dict(), "versions": versions,
                 "python": platform.python_version()}


def cache_snapshot(snapshot, users, items, cfg, out, training):
    path = out / f"{snapshot.cutoff}_{'train' if training else 'validation'}.parquet"
    info = path.with_suffix(".json")
    if path.exists() and info.exists():
        meta = json.loads(info.read_text())
        if sha256(path) != meta["sha256"]:
            raise ValueError(f"Snapshot cache checksum mismatch: {path}")
        print(f"Using cache {path.name}: {meta['rows']:,} rows", flush=True)
        return path, meta
    temporary = path.with_suffix(".tmp")
    writer = None
    rows, queries, positives = 0, 0, 0
    print(f"Building {snapshot.cutoff} ({'training' if training else 'validation'}), history={len(snapshot.history):,}", flush=True)
    started = time.perf_counter()
    try:
        for number, frame in enumerate(iter_dataset(snapshot, users, items, cfg, training)):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame); queries += frame.query_id.nunique(); positives += int(frame.label.sum())
            if number % 8 == 0:
                print(f"  {snapshot.cutoff}: {rows:,} rows, {queries:,} queries", flush=True)
        if writer is None:
            raise ValueError(f"No usable candidates for {snapshot.cutoff}")
    finally:
        if writer:
            writer.close()
    temporary.replace(path)
    meta = {**snapshot.diagnostics, "training": training, "rows": rows, "queries": queries, "positive_candidates": positives,
            "seconds": time.perf_counter() - started, "sha256": sha256(path)}
    write_json(info, meta)
    return path, meta


def compact(frame):
    for col in frame:
        if col not in KEY_COLUMNS and not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].astype("category")
    return frame


def _save_evaluation(cfg, out, validation, targets, manifest, ablation, run):
    baseline_frame = validation[["user_id", "item_id", "cand_pop_score", "cand_recent_score", "cand_knn_score", "rrf_score"]]
    baselines, _ = evaluate_baselines(baseline_frame, targets)
    result = {**run, "source_commit": SOURCE_COMMIT, "archive_sha256": ARCHIVE_SHA256,
              "split": "validation", "cutoff": cfg.validation_cutoff, "test_evaluated": False,
              "baselines": baselines, "selected": manifest["selected"], "ablation": ablation,
              "model_sha256": manifest["model_sha256"], "model_artifact": str((out / "models" / "selected_model.cbm").relative_to(cfg.root)),
              "feature_count": len(manifest["features"]), "categorical_features": manifest["categorical_features"]}
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cfg.root, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        result["git_commit"] = None
    records = [{"method": name, **metrics} for name, metrics in baselines.items()]
    records.append({"method": "catboost", **manifest["selected"]["metrics"]})
    if ablation:
        records.append({"method": "catboost_without_personal_features", **ablation["metrics"]})
    pd.DataFrame(records).to_csv(cfg.path("results_dir") / "validation_results.csv", index=False)
    write_json(cfg.path("results_dir") / "validation_manifest.json", result)
    write_json(out / "run_manifest.json", result)
    return result


def run_validation(cfg, tables=None, fit=True):
    started = time.perf_counter()
    cfg.path("results_dir").mkdir(parents=True, exist_ok=True)
    if tables is None:
        if not (prepared_dir(cfg) / "manifest.json").exists():
            prepare(cfg)
        tables = load_prepared(cfg)
    events, users, items = tables
    out, run = context(cfg, tables)
    paths, cache_info = [], []
    for cutoff in cfg.train_cutoffs:
        path, meta = cache_snapshot(make_snapshot(events, cutoff, cfg.horizon_days), users, items, cfg, out, True)
        paths.append(path); cache_info.append(meta)
    snapshot = make_snapshot(events, cfg.validation_cutoff, cfg.horizon_days)
    val_path, val_info = cache_snapshot(snapshot, users, items, cfg, out, False)
    cache_info.append(val_info)
    # Read only already sampled training queries; no user x full-catalog table is materialized.
    train = compact(pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)) if fit else None
    validation = compact(pd.read_parquet(val_path))
    models = out / "models"
    if fit:
        manifest = fit_grid(train, validation, snapshot.targets, cfg, models)
        ablation = fit_ablation(train, validation, snapshot.targets, cfg, manifest, models)
        del train
        gc.collect()
    else:
        manifest = json.loads((models / "model_manifest.json").read_text())
        ablation = json.loads((models / "ablation.json").read_text())
        if sha256(models / "selected_model.cbm") != manifest["model_sha256"]:
            raise ValueError("Model checksum mismatch")
        restored = CatBoostRanker(); restored.load_model(str(models / "selected_model.cbm"))
        scored = validation[["user_id", "item_id"]].copy()
        scored["score"] = restored.predict(Pool(validation[manifest["features"]], cat_features=manifest["categorical_features"]))
        repeated, _ = evaluate_scores(scored, snapshot.targets, "score")
        if any(abs(repeated[m] - manifest["selected"]["metrics"][m]) > 1e-10 for m in ["ndcg@10", "recall@10", "map@10"]):
            raise ValueError("Reloaded model metrics differ from recorded validation results")
    run.update(snapshots=cache_info, elapsed_seconds=time.perf_counter() - started,
               peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if platform.system() == "Darwin" else 1024),
               validation_users=len(snapshot.targets), artifact_directory=str(out.relative_to(cfg.root)))
    return _save_evaluation(cfg, out, validation, snapshot.targets, manifest, ablation, run)


def recommend(cfg, user_id, k=10):
    if k < 1 or k > 10:
        raise ValueError("Supported recommendation length is 1..10")
    tables = load_prepared(cfg)
    out, _ = context(cfg, tables)
    manifest = json.loads((out / "models" / "model_manifest.json").read_text())
    if sha256(out / "models" / "selected_model.cbm") != manifest["model_sha256"]:
        raise ValueError("Model checksum mismatch")
    events, users, items = tables
    snapshot = make_snapshot(events, manifest["prediction_cutoff"], cfg.horizon_days)
    store = FeatureStore(snapshot, users, items, cfg)
    if user_id not in store.user_index:
        order = np.lexsort((store.item_ids, -store.item_features.i_count_all.to_numpy()))[:k]
        return {"user_id": user_id, "as_of": snapshot.cutoff, "method": "popularity_fallback", "item_ids": store.item_ids[order].tolist()}
    pairs = CandidateRetriever(store, cfg).generate([user_id])
    if pairs.empty:
        return {"user_id": user_id, "as_of": snapshot.cutoff, "method": "catboost", "item_ids": []}
    frame = store.transform(pairs)
    model = CatBoostRanker(); model.load_model(str(out / "models" / "selected_model.cbm"))
    scores = model.predict(Pool(frame[manifest["features"]], cat_features=manifest["categorical_features"]))
    order = np.lexsort((frame.item_id.to_numpy(), -scores))[:k]
    return {"user_id": user_id, "as_of": snapshot.cutoff, "method": "catboost", "item_ids": frame.item_id.to_numpy()[order].tolist(), "scores": scores[order].tolist()}
