"""Fixed-budget ranker training: evaluation labels never enter model.fit."""
import gc
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd

from kion.io import sha256
from kion.metrics import evaluate_scores
from .baseline import KEYS, assert_pairs
from .engine import frame_digest
from .io import checked_receipt, digest, write_json


class RSSMonitor:
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()

        def sample():
            while not self.stop.wait(.05):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def predict_model(path, frame, columns, categorical, threads=2):
    from catboost import CatBoostRanker, Pool
    model = CatBoostRanker()
    model.load_model(str(path))
    if model.feature_names_ != columns or model.get_cat_feature_indices() != [columns.index(c) for c in categorical]:
        raise ValueError("Saved model feature schema differs from its receipt")
    return model.predict(Pool(frame[columns], cat_features=categorical, thread_count=threads), thread_count=threads)


def fit_evaluate(train, validation, targets, columns, categorical, parameters, out, identity,
                 repeats=3):
    from catboost import CatBoostRanker, Pool
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert_pairs(train)
    assert_pairs(validation)
    if (not columns or len(columns) != len(set(columns)) or set(columns) & set(KEYS)
            or not set(categorical).issubset(columns) or parameters["thread_count"] not in (1, 2)):
        raise ValueError("Invalid ranker feature schema or thread limit")
    allowed = ("cand_", "u_", "i_", "pair_", "auto__")
    if any(c != "rrf_score" and not c.startswith(allowed) for c in columns):
        raise ValueError("Undeclared ranker feature")
    if train.label.nunique() != 2:
        raise ValueError("Ranker training requires both classes")
    if train.groupby("query_id").user_id.nunique().max() != 1:
        raise ValueError("A ranking query cannot contain multiple users")
    signature = {"context": identity, "columns": columns, "categorical": categorical,
                 "parameters": parameters, "train": frame_digest(train[KEYS + columns]),
                 "evaluation": frame_digest(validation[["user_id", "item_id"] + columns]),
                 "targets": digest({int(u): sorted(map(int, v)) for u, v in targets.items()}),
                 "repeats": repeats}
    receipt_path = out / "fit.json"
    existing = checked_receipt(receipt_path, signature)
    if existing:
        return existing
    if not train.query_id.is_monotonic_increasing:
        train = train.sort_values("query_id", kind="stable")
    threads = parameters["thread_count"]
    with RSSMonitor() as memory:
        start = time.perf_counter()
        train_pool = Pool(train[columns], label=train.label, group_id=train.query_id,
                          cat_features=categorical, thread_count=threads)
        pool_seconds = time.perf_counter() - start
        model = CatBoostRanker(**parameters, allow_writing_files=False, used_ram_limit="6gb")
        start = time.perf_counter()
        model.fit(train_pool, verbose=False, use_best_model=False)
        fit_seconds = time.perf_counter() - start
        del train_pool
        gc.collect()
        start = time.perf_counter()
        validation_pool = Pool(validation[columns], cat_features=categorical, thread_count=threads)
        validation_pool_seconds = time.perf_counter() - start
        predicted = model.predict(validation_pool, thread_count=threads)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            repeated = model.predict(validation_pool, thread_count=threads)
            timings.append(time.perf_counter() - start)
            np.testing.assert_allclose(repeated, predicted, rtol=0, atol=1e-10)
        temporary = out / "model.tmp.cbm"
        model.save_model(str(temporary))
        temporary.replace(out / "model.cbm")
        importance = model.get_feature_importance(type="PredictionValuesChange", thread_count=threads)
        if model.tree_count_ != parameters["iterations"]:
            raise ValueError("Fixed-budget fit saved an unexpected number of trees")
        del validation_pool, model
        gc.collect()
        loaded = predict_model(out / "model.cbm", validation, columns, categorical, threads)
        np.testing.assert_allclose(loaded, predicted, rtol=0, atol=1e-10)
    scored = validation[["user_id", "item_id"]].copy()
    scored["score"] = predicted
    metrics, per_user = evaluate_scores(scored, targets, "score")
    scored.to_parquet(out / "predictions.parquet", index=False)
    per_user.to_parquet(out / "per_user.parquet")
    pd.DataFrame({"feature": columns, "importance": importance}).to_csv(out / "importance.csv", index=False)
    files = {n: sha256(out / n) for n in ("model.cbm", "predictions.parquet", "per_user.parquet", "importance.csv")}
    record = {"identity": signature, "files": files, "metrics": metrics,
              "pool_seconds": pool_seconds, "fit_seconds": fit_seconds,
              "validation_pool_seconds": validation_pool_seconds,
              "predict_seconds": timings, "predict_seconds_median": float(np.median(timings)),
              "process_peak_rss_mb": memory.peak / 1024**2, "model_bytes": (out / "model.cbm").stat().st_size,
              "train_rows": len(train), "train_queries": int(train.query_id.nunique()),
              "evaluation_rows": len(validation), "reload_verified": True, "early_stopping": False}
    write_json(receipt_path, record)
    return record
