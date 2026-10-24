"""The seven predeclared comparisons on identical outer-validation queries."""
import gc
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from kion.io import sha256
from kion.metrics import bootstrap, evaluate_scores
from kion.protocol import day_number, load_prepared, make_snapshot
from .baseline import Baseline, KEYS, attach_features
from .engine import HistoryFeatures, cache_features
from .grammar import resolve_specs
from .io import checked_receipt, digest, read_json, source_hashes, write_json
from .model import RSSMonitor, fit_evaluate, predict_model

VALIDATION_NEW = ("source_raw", "auto_no_source", "auto", "random")
FINAL_NEW = ("source", "source_raw", "auto", "random")


def research_identity(research, baseline):
    return {"inputs": baseline.prepare()["input_id"], "research": research.to_dict(), "sources": source_hashes()}


def load_selection(research, baseline):
    identity = research_identity(research, baseline)
    record = checked_receipt(research.results / "search.json", identity)
    if record is None:
        raise ValueError("Complete the inner feature search before outer validation")
    selection = read_json(research.results / "selection.json")
    if (selection["selected"] != record["selected"] or selection["random"] != record["random"]
            or len(selection["selected"]) != len(selection["random"])):
        raise ValueError("Feature selection differs from its search receipt")
    resolve_specs(selection["selected"])
    resolve_specs(selection["random"])
    return selection, identity


def variant_features(families, selection):
    s, r, m = (families[k] for k in ("source", "raw", "manual"))
    a, random = selection["selected"], selection["random"]
    if not a or len(a) != len(random):
        raise ValueError("Selected and random automatic sets must have the same positive size")
    resolve_specs(a)
    resolve_specs(random)
    return {"source": s, "source_raw": s + r, "manual": r + m, "full": s + r + m,
            "auto_no_source": r + a, "auto": s + r + a, "random": s + r + random}


def dataset_path(research, stage, spec, training):
    role = "train" if training else "evaluation"
    return research.artifacts / "datasets" / stage / f'{spec["cutoff"]}_{role}.parquet'


def build_datasets(research, baseline, selection, stage="validation"):
    if stage not in ("validation", "test"):
        raise ValueError("Unknown dataset stage")
    if stage == "test":
        from .final import verify_frozen
        verify_frozen(research, baseline)
    train_specs, evaluation_spec = baseline.snapshots(stage)
    events, users, items = load_prepared(baseline.cfg)
    # Outer validation does not read targets from the final test week.
    if stage == "validation":
        events = events.loc[events.day < day_number(baseline.cfg.test_cutoff)].copy()
    specs = resolve_specs(sorted(set(selection["selected"] + selection["random"])))
    receipts = []
    for spec, training in [*((s, True) for s in train_specs), (evaluation_spec, False)]:
        pairs = baseline.read_snapshot(spec, ["user_id", "item_id"])
        started = time.perf_counter()
        with RSSMonitor() as memory:
            engine = HistoryFeatures(events, users, items, spec["cutoff"])
            receipt = cache_features(engine, pairs, specs, dataset_path(research, stage, spec, training), research.batch_rows)
        receipts.append({"cutoff": spec["cutoff"], "training": training, "cache": receipt,
                         "request_seconds": time.perf_counter() - started,
                         "process_peak_rss_mb": memory.peak / 1024**2})
        del engine, pairs
        gc.collect()
    snapshot = make_snapshot(events, evaluation_spec["cutoff"], baseline.cfg.horizon_days)
    return train_specs, evaluation_spec, snapshot, receipts


def load_frame(research, baseline, stage, spec, training, columns):
    original = [c for c in columns if not c.startswith("auto__")]
    auto = [c for c in columns if c.startswith("auto__")]
    base = baseline.read_snapshot(spec, KEYS + original)
    if not auto:
        return base
    path = dataset_path(research, stage, spec, training)
    receipt = read_json(path.with_suffix(".json"))
    if sha256(path) != receipt["files"][path.name]:
        raise ValueError("Automatic dataset checksum mismatch")
    generated = pd.read_parquet(path, columns=["user_id", "item_id"] + auto)
    return attach_features(base, generated)


def score_reference(research, baseline, spec, snapshot, columns, model_path, expected,
                    out, identity):
    """Reuse a frozen model, but reproduce its scores and benchmark prediction."""
    from catboost import CatBoostRanker, Pool
    out = Path(out)
    signature = {"context": identity, "model_sha256": sha256(model_path), "snapshot": spec,
                 "columns": columns, "targets": digest({int(u): sorted(v) for u, v in snapshot.targets.items()}),
                 "repeats": research.prediction_repeats}
    existing = checked_receipt(out / "reference.json", signature)
    if existing:
        return existing
    frame = baseline.read_snapshot(spec, ["user_id", "item_id"] + columns)
    cats = [c for c in baseline.frozen["categorical_features"] if c in columns]
    out.mkdir(parents=True, exist_ok=True)
    with RSSMonitor() as memory:
        model = CatBoostRanker()
        model.load_model(str(model_path))
        if model.feature_names_ != columns:
            raise ValueError("Reference model schema mismatch")
        start = time.perf_counter()
        pool = Pool(frame[columns], cat_features=cats, thread_count=research.threads)
        pool_seconds = time.perf_counter() - start
        predicted = model.predict(pool, thread_count=research.threads)
        timings = []
        for _ in range(research.prediction_repeats):
            start = time.perf_counter()
            repeated = model.predict(pool, thread_count=research.threads)
            timings.append(time.perf_counter() - start)
            np.testing.assert_allclose(predicted, repeated, rtol=0, atol=1e-10)
    scored = frame[["user_id", "item_id"]].copy()
    scored["score"] = predicted
    metrics, per_user = evaluate_scores(scored, snapshot.targets, "score")
    if any(abs(metrics[m] - expected[m]) > 1e-10 for m in ("ndcg@10", "recall@10", "map@10")):
        raise ValueError("Reference scores do not reproduce the published metrics")
    scored.to_parquet(out / "predictions.parquet", index=False)
    per_user.to_parquet(out / "per_user.parquet")
    record = {"identity": signature, "metrics": metrics, "training_reused": True,
              "fit_seconds": None, "pool_seconds": pool_seconds, "predict_seconds": timings,
              "predict_seconds_median": float(np.median(timings)),
              "process_peak_rss_mb": memory.peak / 1024**2, "model_bytes": Path(model_path).stat().st_size,
              "files": {p: sha256(out / p) for p in ("predictions.parquet", "per_user.parquet")}}
    write_json(out / "reference.json", record)
    return record


def reference_validation_specs(baseline):
    root = baseline.cfg.path("results_dir")
    ablation = read_json(root / "feature_ablation_manifest.json")
    frozen = baseline.frozen
    original = read_json(root / "validation_manifest.json")
    # Preserve the exact saved feature order, rather than the S/R/M grouping order.
    values = {"full": {"path": frozen["model_artifact"], "columns": frozen["features"],
                       "metrics": original["selected"]["metrics"]}}
    for name, key in (("source", "source_only"), ("manual", "manual_only")):
        columns = [c for c in frozen["features"] if (c in baseline.families["source"]) == (name == "source")]
        values[name] = {"path": str(Path(ablation["artifact_directory"]) / f"{key}.cbm"),
                        "columns": columns, "metrics": ablation["benchmarks"][key]["metrics"]}
    return values


def collect_results(research, stage, records, identity, generation):
    out = research.artifacts / stage
    rows = [{"variant": name, **record["metrics"], "fit_seconds": record.get("fit_seconds"),
             "training_reused": record.get("training_reused", False),
             "predict_seconds_median": record["predict_seconds_median"],
             "process_peak_rss_mb": record["process_peak_rss_mb"], "model_bytes": record["model_bytes"]}
            for name, record in records.items()]
    research.results.mkdir(parents=True, exist_ok=True)
    csv_path = research.results / f"{stage}_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    per_user = {name: pd.read_parquet(out / name / "per_user.parquet") for name in records}
    primary = bootstrap(per_user["auto"], per_user["full"], n=1000, seed=42)
    paired = {name: bootstrap(per_user["auto"], per_user[name], n=1000, seed=42)
              for name in ("source", "source_raw", "random")}
    interval_path = research.results / f"{stage}_intervals.json"
    write_json(interval_path, {"primary_auto_minus_full": primary, "diagnostic_auto_minus": paired,
                              "note": "Fixed-model user bootstrap; no refit/temporal uncertainty or multiplicity correction"})
    files = {csv_path.name: sha256(csv_path), interval_path.name: sha256(interval_path)}
    for artifact in sorted(out.rglob("*")):
        if artifact.is_file() and ".tmp" not in artifact.name:
            files[os.path.relpath(artifact, research.results)] = sha256(artifact)
    record = {"identity": identity, "files": files, "models": records,
              "generation": generation, "test_evaluated": stage == "test"}
    write_json(research.results / f"{stage}.json", record)
    return record


def run_validation(research):
    baseline = Baseline(research)
    selection, identity = load_selection(research, baseline)
    existing = checked_receipt(research.results / "validation.json", identity)
    if existing:
        return existing
    if (research.results / "frozen.json").exists():
        raise ValueError("Cannot change validation after freezing")
    train_specs, spec, snapshot, generation = build_datasets(research, baseline, selection)
    variants = variant_features(baseline.families, selection)
    parameters = {**baseline.frozen["parameters"], "thread_count": research.threads}
    records = {}
    for name in VALIDATION_NEW:
        columns = variants[name]
        train = pd.concat([load_frame(research, baseline, "validation", s, True, columns)
                           for s in train_specs], ignore_index=True)
        evaluation = load_frame(research, baseline, "validation", spec, False, columns)
        cats = [c for c in baseline.frozen["categorical_features"] if c in columns]
        records[name] = fit_evaluate(train, evaluation, snapshot.targets, columns, cats, parameters,
                                     research.artifacts / "validation" / name, identity, research.prediction_repeats)
        del train, evaluation
        gc.collect()
    for name, ref in reference_validation_specs(baseline).items():
        records[name] = score_reference(research, baseline, spec, snapshot, ref["columns"],
                                        baseline.cfg.root / ref["path"], ref["metrics"],
                                        research.artifacts / "validation" / name, identity)
    return collect_results(research, "validation", records, identity, generation)
