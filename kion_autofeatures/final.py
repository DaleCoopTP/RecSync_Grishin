"""Seal the method, refit all models, then open final targets exactly once."""
import gc
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from kion.io import sha256
from kion.metrics import evaluate_scores
from kion.protocol import load_prepared, make_snapshot
from .baseline import Baseline, KEYS, assert_pairs
from .engine import frame_digest
from .experiment import (FINAL_NEW, build_datasets, collect_results, load_frame,
                         load_selection, reference_validation_specs, research_identity,
                         score_reference, variant_features)
from .io import checked_receipt, digest, read_json, source_hashes, verify_files, write_json
from .model import RSSMonitor, predict_model


def freeze(research):
    baseline = Baseline(research)
    selection, identity = load_selection(research, baseline)
    path = research.results / "frozen.json"
    if path.exists():
        return verify_frozen(research, baseline)
    validation = checked_receipt(research.results / "validation.json", identity)
    if validation is None:
        raise ValueError("Complete outer validation before freezing")
    if set(validation["models"]) != {"source", "source_raw", "manual", "full", "auto_no_source", "auto", "random"}:
        raise ValueError("Outer validation must contain all seven predeclared variants")
    files = {name: sha256(research.results / name)
             for name in ("inputs.json", "search.json", "selection.json", "validation.json")}
    files.update(validation["files"])
    files.update(read_json(research.results / "search.json")["files"])
    protocol = research.root / "docs" / "automatic_features_protocol.md"
    if not protocol.is_file():
        raise ValueError("Research protocol document is required")
    files[os.path.relpath(protocol, research.results)] = sha256(protocol)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=research.root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    record = {"identity": identity, "files": files, "selection": selection["selected"],
              "random": selection["random"], "parameters": baseline.frozen["parameters"],
              "final_variants": list(FINAL_NEW), "reference": "full", "source_commit": commit,
              "primary_comparison": "auto minus full", "bootstrap": {"samples": 1000, "seed": 42},
              "baseline_test_already_known": True,
              "limitations": ["One test week and one training seed", "Fixed-model user bootstrap",
                              "KION last-watch records and static metadata", "One random feature set",
                              "Shared manually engineered retrieval and source signals"]}
    record["seal"] = digest(record)
    write_json(path, record)
    return record


def verify_frozen(research, baseline=None):
    path = research.results / "frozen.json"
    if not path.exists():
        raise ValueError("Freeze the validated method before opening final evaluation")
    record = read_json(path)
    if record["seal"] != digest({k: v for k, v in record.items() if k != "seal"}):
        raise ValueError("Frozen research protocol is corrupted")
    baseline = baseline or Baseline(research)
    if record["identity"] != research_identity(research, baseline):
        raise ValueError("Research code, inputs or configuration changed after freezing")
    verify_files(research.results, record["files"])
    if record["final_variants"] != list(FINAL_NEW):
        raise ValueError("Final comparison differs from the predeclared variants")
    return record


def fit_final(train, columns, categorical, parameters, out, identity, allowed_cutoffs):
    """No evaluation frame or evaluation targets are accepted by this interface."""
    from catboost import CatBoostRanker, Pool
    assert_pairs(train)
    if set(train.cutoff.unique()) != set(allowed_cutoffs) or train.label.nunique() != 2:
        raise ValueError("Final training requires exactly the four approved historical cutoffs")
    if (set(columns) & set(KEYS) or not columns or len(columns) != len(set(columns))
            or not set(categorical).issubset(columns) or parameters["thread_count"] not in (1, 2)):
        raise ValueError("Invalid final model schema or thread budget")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signature = {"context": identity, "columns": columns, "categorical": categorical,
                 "parameters": parameters, "train": frame_digest(train[KEYS + columns]),
                 "cutoffs": list(allowed_cutoffs)}
    existing = checked_receipt(out / "fit.json", signature)
    if existing:
        return existing
    if not train.query_id.is_monotonic_increasing:
        train = train.sort_values("query_id", kind="stable")
    with RSSMonitor() as memory:
        started = time.perf_counter()
        pool = Pool(train[columns], label=train.label, group_id=train.query_id,
                    cat_features=categorical, thread_count=parameters["thread_count"])
        pool_seconds = time.perf_counter() - started
        model = CatBoostRanker(**parameters, allow_writing_files=False, used_ram_limit="6gb")
        started = time.perf_counter()
        model.fit(pool, verbose=False, use_best_model=False)
        fit_seconds = time.perf_counter() - started
        if model.tree_count_ != parameters["iterations"]:
            raise ValueError("Unexpected final tree count")
        temporary = out / "model.tmp.cbm"
        model.save_model(str(temporary))
        temporary.replace(out / "model.cbm")
    record = {"identity": signature, "files": {"model.cbm": sha256(out / "model.cbm")},
              "fit_seconds": fit_seconds, "training_pool_seconds": pool_seconds,
              "training_process_peak_rss_mb": memory.peak / 1024**2,
              "train_rows": len(train), "train_queries": int(train.query_id.nunique()),
              "early_stopping": False, "test_labels_used": False}
    write_json(out / "fit.json", record)
    return record


def save_segments(research, snapshot, records):
    counts = snapshot.history.groupby("user_id").size()
    rows = []
    for name in records:
        per_user = pd.read_parquet(research.artifacts / "test" / name / "per_user.parquet")
        history = counts.reindex(per_user.index).fillna(0)
        masks = {"history_1_2": history.between(1, 2), "history_3_9": history.between(3, 9),
                 "history_10_plus": history >= 10,
                 "retrieval_zero_hits": per_user["candidate_recall@200"] == 0,
                 "retrieval_some_hits": per_user["candidate_recall@200"] > 0}
        for segment, mask in masks.items():
            frame = per_user.loc[mask]
            rows.append({"variant": name, "segment": segment, "users": len(frame),
                         **{m: float(frame[m].mean()) if len(frame) else None
                            for m in ("ndcg@10", "recall@10", "map@10")}})
    path = research.results / "test_segments.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_final(research):
    baseline = Baseline(research)
    frozen = verify_frozen(research, baseline)
    identity = {"frozen_seal": frozen["seal"], **frozen["identity"]}
    existing = checked_receipt(research.results / "test.json", identity)
    if existing:
        return existing
    selection = read_json(research.results / "selection.json")
    variants = variant_features(baseline.families, selection)
    train_specs, _, _, generation = build_datasets(research, baseline, selection, "test", "train")
    allowed = baseline.cfg.train_cutoffs + [baseline.cfg.validation_cutoff]
    parameters = {**frozen["parameters"], "thread_count": research.threads}
    fitted = {}
    # Complete every fixed refit before constructing any final target dictionary.
    for name in FINAL_NEW:
        columns = variants[name]
        train = pd.concat([load_frame(research, baseline, "test", spec, True, columns)
                           for spec in train_specs], ignore_index=True)
        cats = [c for c in baseline.frozen["categorical_features"] if c in columns]
        fitted[name] = fit_final(train, columns, cats, parameters, research.artifacts / "test" / name,
                                 identity, allowed)
        del train
        gc.collect()
    _, spec, snapshot, evaluation_generation = build_datasets(research, baseline, selection, "test", "evaluation")
    records = {}
    for name in FINAL_NEW:
        # score_reference also checks model schema and writes predictions/per-user metrics.
        score = score_reference(research, baseline, spec, snapshot, variants[name],
                                 research.artifacts / "test" / name / "model.cbm", None,
                                 research.artifacts / "test" / name, identity, stage="test")
        records[name] = {**score, **{k: v for k, v in fitted[name].items() if k not in ("identity", "files")},
                         "training_reused": False}
    original = read_json(baseline.cfg.path("results_dir") / "test_manifest.json")
    ref = original["models"]["catboost"]
    records["full"] = score_reference(research, baseline, spec, snapshot, ref["identity"]["features"],
                                       baseline.cfg.root / ref["path"], original["metrics"]["catboost"],
                                       research.artifacts / "test" / "full", identity)
    segments = save_segments(research, snapshot, records)
    record = collect_results(research, "test", records, identity, generation + evaluation_generation)
    record["files"][segments.name] = sha256(segments)
    record["snapshot_diagnostics"] = snapshot.diagnostics
    write_json(research.results / "test.json", record)
    return record


def verify(research):
    baseline = Baseline(research)
    baseline.prepare()
    verified = []
    if (research.results / "search.json").exists():
        selection, identity = load_selection(research, baseline)
        verified.append("search")
    else:
        return {"verified": ["inputs"], "test_evaluated": False}
    validation = checked_receipt(research.results / "validation.json", identity)
    if validation:
        _verify_predictions(research, baseline, selection, validation, "validation")
        verified.append("validation")
    if (research.results / "frozen.json").exists():
        frozen = verify_frozen(research, baseline)
        verified.append("frozen")
        identity = {"frozen_seal": frozen["seal"], **frozen["identity"]}
        final = checked_receipt(research.results / "test.json", identity)
        if final:
            _verify_predictions(research, baseline, selection, final, "test")
            verified.append("test")
    return {"verified": ["inputs"] + verified, "test_evaluated": "test" in verified,
            "reload_verified": True}


def _verify_predictions(research, baseline, selection, record, stage):
    _, spec = baseline.snapshots(stage)
    variants = variant_features(baseline.families, selection)
    references = reference_validation_specs(baseline) if stage == "validation" else {}
    if stage == "test":
        original = read_json(baseline.cfg.path("results_dir") / "test_manifest.json")
        ref = original["models"]["catboost"]
        references["full"] = {"path": ref["path"], "columns": ref["identity"]["features"]}
    events, _, _ = load_prepared(baseline.cfg)
    snapshot = make_snapshot(events, spec["cutoff"], baseline.cfg.horizon_days)
    del events
    for name, saved in record["models"].items():
        if name in references:
            ref = references[name]
            columns, model_path = ref["columns"], baseline.cfg.root / ref["path"]
        else:
            columns = variants[name]
            model_path = research.artifacts / stage / name / "model.cbm"
        frame = load_frame(research, baseline, stage, spec, False, columns)
        cats = [c for c in baseline.frozen["categorical_features"] if c in columns]
        predicted = predict_model(model_path, frame, columns, cats, research.threads)
        recorded = pd.read_parquet(research.artifacts / stage / name / "predictions.parquet")
        if not frame[["user_id", "item_id"]].reset_index(drop=True).equals(recorded[["user_id", "item_id"]]):
            raise ValueError("Saved predictions changed candidate identity or order")
        np.testing.assert_allclose(predicted, recorded.score.to_numpy(), rtol=0, atol=1e-10)
        recorded["score"] = predicted
        metrics, _ = evaluate_scores(recorded, snapshot.targets, "score")
        if any(abs(metrics[m] - saved["metrics"][m]) > 1e-10 for m in ("ndcg@10", "recall@10", "map@10")):
            raise ValueError("Reloaded research model metrics differ")
        del frame, recorded
        gc.collect()
