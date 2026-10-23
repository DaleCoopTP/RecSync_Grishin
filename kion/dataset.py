import numpy as np
import pandas as pd
from .features import FeatureStore
from .candidates import CandidateRetriever
from .protocol import day_number

KEY_COLUMNS = {"user_id", "item_id", "label", "query_id", "cutoff"}


def selected_training_users(snapshot, cfg):
    # Different, deterministic RNG per historical cutoff. Never sample event rows.
    rng = np.random.default_rng(cfg.seed + day_number(snapshot.cutoff))
    return np.sort(rng.choice(snapshot.active_users, min(len(snapshot.active_users), cfg.train_users_per_cutoff), replace=False))


def label_pairs(pairs, targets):
    return np.fromiter((int(int(i) in targets.get(int(u), set())) for u, i in zip(pairs.user_id, pairs.item_id)), dtype="int8", count=len(pairs))


def iter_dataset(snapshot, users, items, cfg, training=False):
    store = FeatureStore(snapshot, users, items, cfg)
    retriever = CandidateRetriever(store, cfg)
    chosen = selected_training_users(snapshot, cfg) if training else snapshot.active_users
    for start in range(0, len(chosen), cfg.batch_users):
        frame = retriever.generate(chosen[start:start + cfg.batch_users])
        if frame.empty:
            continue
        frame = store.transform(frame)
        frame["label"] = label_pairs(frame, snapshot.targets)
        frame["query_id"] = frame.user_id.astype("int64") + day_number(snapshot.cutoff) * 10000000
        frame["cutoff"] = snapshot.cutoff
        if training:
            informative = frame.groupby("query_id").label.transform("sum") > 0
            frame = frame.loc[informative].copy()
        if len(frame):
            yield frame


def feature_columns(frame, ablate=False):
    columns = [c for c in frame if c not in KEY_COLUMNS]
    if ablate:
        columns = [c for c in columns if not c.startswith(("pair_", "cand_knn_", "cand_genre_")) and c != "rrf_score"]
    return columns


def categorical_columns(frame, columns):
    return [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
