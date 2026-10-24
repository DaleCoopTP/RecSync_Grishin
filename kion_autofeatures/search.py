"""Bounded feature selection using only two inner train dates and a third date."""
import gc
import hashlib
import os
import time

import numpy as np
import pandas as pd

from kion.candidates import CandidateRetriever
from kion.dataset import label_pairs
from kion.features import FeatureStore
from kion.io import sha256
from kion.protocol import day_number, load_prepared, make_snapshot
from .baseline import KEYS, Baseline, attach_features
from .engine import HistoryFeatures, cache_features
from .grammar import enumerate_features
from .io import checked_receipt, digest, source_hashes, write_json
from .model import fit_evaluate


def choose_users(ids, count, seed, cutoff):
    ids = np.sort(np.unique(ids))
    return np.sort(np.random.default_rng(seed + day_number(cutoff)).choice(
        ids, min(count, len(ids)), replace=False))


def prune_features(frame, candidates, base_columns):
    """Training-only removal of constants and exact duplicate float32 vectors."""
    signatures, kept, rejected = {}, [], {}

    def signature(column):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float32", copy=True)
        values[values == 0] = 0  # normalize negative zero
        values[np.isnan(values)] = np.nan  # normalize NaN payloads
        return hashlib.sha256(values.tobytes()).hexdigest()

    for column in base_columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            signatures.setdefault(signature(column), column)
    for column in sorted(candidates):
        if frame[column].nunique(dropna=False) <= 1:
            rejected[column] = "constant on inner training"
            continue
        key = signature(column)
        if key in signatures:
            rejected[column] = "duplicate of " + signatures[key]
        else:
            signatures[key] = column
            kept.append(column)
    return kept, rejected


def choose_prefix(records, tolerance):
    best = max(r["metrics"]["ndcg@10"] for r in records)
    return min((r for r in records if best - r["metrics"]["ndcg@10"] <= tolerance), key=lambda r: r["size"])


def inner_validation(events, users, items, baseline, research):
    cfg = baseline.cfg
    cutoff = cfg.train_cutoffs[2]
    snapshot = make_snapshot(events, cutoff, cfg.horizon_days)
    chosen = choose_users(snapshot.active_users, research.inner_users, cfg.seed, cutoff)
    store = FeatureStore(snapshot, users, items, cfg)
    retrieval_start = time.perf_counter()
    pairs = CandidateRetriever(store, cfg).generate(chosen)
    retrieval_seconds = time.perf_counter() - retrieval_start
    frame = store.transform(pairs)
    frame["label"] = label_pairs(frame, snapshot.targets)
    frame["query_id"] = frame.user_id.astype("int64") + day_number(cutoff) * 10000000
    frame["cutoff"] = cutoff
    # Keep users without a retrieved positive; do not derive targets from candidates.
    targets = {int(u): snapshot.targets[int(u)] for u in chosen}
    return frame, targets, {"users": chosen.tolist(), "retrieval_seconds": retrieval_seconds,
                            "zero_positive_queries": int((frame.groupby("query_id").label.sum() == 0).sum())}


def run_search(research):
    baseline = Baseline(research)
    inputs = baseline.prepare()
    if (research.results / "frozen.json").exists():
        raise ValueError("Search is closed after freezing; verify existing results instead")
    identity = {"inputs": inputs["input_id"], "research": research.to_dict(), "sources": source_hashes()}
    receipt_path = research.results / "search.json"
    existing = checked_receipt(receipt_path, identity)
    if existing:
        return existing
    cfg = baseline.cfg
    events, users, items = load_prepared(cfg)
    events = events.loc[events.day < day_number(cfg.validation_cutoff)].copy()
    specs = enumerate_features()
    families = baseline.families
    base_columns = families["source"] + families["raw"]
    cats = [c for c in baseline.frozen["categorical_features"] if c in base_columns]
    train_specs, _ = baseline.snapshots()
    frames, receipts = [], []
    work = research.artifacts / "search"
    for spec in train_specs[:2]:
        frame = baseline.read_snapshot(spec, KEYS + base_columns)
        chosen = choose_users(frame.user_id, research.inner_users, cfg.seed, spec["cutoff"])
        frame = frame.loc[frame.user_id.isin(chosen)].reset_index(drop=True)
        engine = HistoryFeatures(events, users, items, spec["cutoff"])
        path = work / f'{spec["cutoff"]}.parquet'
        receipts.append(cache_features(engine, frame[["user_id", "item_id"]], specs, path, research.batch_rows))
        frames.append(attach_features(frame, pd.read_parquet(path)))
        del engine
        gc.collect()
    train = pd.concat(frames, ignore_index=True)
    del frames
    validation, targets, diagnostics = inner_validation(events, users, items, baseline, research)
    path = work / "inner_validation.parquet"
    engine = HistoryFeatures(events, users, items, cfg.train_cutoffs[2])
    receipts.append(cache_features(engine, validation[["user_id", "item_id"]], specs, path, research.batch_rows))
    validation = attach_features(validation[KEYS + base_columns], pd.read_parquet(path))
    del engine, events, users, items
    gc.collect()
    eligible, rejected = prune_features(train, [s.name for s in specs], base_columns)
    if not eligible:
        raise ValueError("All generated features were constant or duplicated on inner training")
    screen_parameters = {**baseline.frozen["parameters"], "iterations": research.screen_iterations,
                         "depth": research.screen_depth, "thread_count": research.threads}
    # Importance uses training statistics only; this training-set diagnostic is
    # not a held-out score and is never used to select a prefix size.
    # Evaluation on one last training cutoff avoids duplicate user-item pairs across dates.
    screen_eval = train.loc[train.cutoff == cfg.train_cutoffs[1]].reset_index(drop=True)
    screen_targets = {int(u): set(g.loc[g.label == 1, "item_id"].astype(int))
                      for u, g in screen_eval.groupby("user_id")}
    fit_evaluate(train, screen_eval, screen_targets, base_columns + eligible, cats,
                 screen_parameters, work / "screen", identity, research.prediction_repeats)
    importance = pd.read_csv(work / "screen" / "importance.csv")
    ranked = importance.loc[importance.feature.isin(eligible)].sort_values(
        ["importance", "feature"], ascending=[False, True]).feature.tolist()[:research.max_features]
    del screen_eval
    gc.collect()
    records = []
    for size in sorted({min(n, len(ranked)) for n in research.prefix_sizes}):
        parameters = {**baseline.frozen["parameters"], "thread_count": research.threads}
        result = fit_evaluate(train, validation, targets, base_columns + ranked[:size], cats,
                              parameters, work / f"prefix_{size}", identity, research.prediction_repeats)
        records.append({"size": size, "metrics": result["metrics"]})
    chosen = choose_prefix(records, research.tie_tolerance)
    selected = ranked[:chosen["size"]]
    random = np.random.default_rng(cfg.seed).choice(sorted(eligible), len(selected), replace=False).tolist()
    selection = {"selected": selected, "random": random, "ranked": ranked,
                 "eligible": eligible, "rejected": rejected, "prefixes": records,
                 "chosen_size": chosen["size"], "inner_validation": diagnostics,
                 "feature_definitions": {s.name: {**s.to_dict(), "description": s.describe()} for s in specs},
                 "generation_receipts": receipts, "test_evaluated": False}
    path = research.results / "selection.json"
    write_json(path, selection)
    files = {"selection.json": sha256(path)}
    for artifact in sorted(work.rglob("*")):
        if artifact.is_file() and ".tmp" not in artifact.name:
            files[os.path.relpath(artifact, research.results)] = sha256(artifact)
    record = {"identity": identity, "files": files, "selected": selected, "random": random,
              "chosen_size": len(selected), "test_evaluated": False}
    write_json(receipt_path, record)
    return record
