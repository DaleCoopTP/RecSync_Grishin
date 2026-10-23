from dataclasses import replace
import numpy as np
import pandas as pd
from kion.config import Config
from kion.protocol import make_snapshot
from kion.features import FeatureStore
from kion.dataset import iter_dataset, label_pairs, feature_columns


def test_pair_features_and_labels(tables):
    e, u, i = tables
    snap = make_snapshot(e, "2021-07-19")
    store = FeatureStore(snap, u, i, Config())
    frame = store.transform(pd.DataFrame({"user_id": [1], "item_id": [12]}))
    assert frame.pair_genre_count_all.iloc[0] == 0.5
    assert frame.pair_directors_count_all.iloc[0] == 1
    assert frame.pair_directors_recency.iloc[0] == 18
    assert label_pairs(frame, snap.targets).tolist() == [1]


def test_training_filter_does_not_change_evaluation_cohort(tables):
    e, u, i = tables
    cfg = replace(Config(), train_users_per_cutoff=10)
    snap = make_snapshot(e, "2021-07-19")
    evaluation = pd.concat(iter_dataset(snap, u, i, cfg))
    train = pd.concat(iter_dataset(snap, u, i, cfg, training=True))
    assert set(evaluation.user_id) == {1, 2}
    assert set(train.user_id) == {1}
    assert not set(["label", "query_id", "user_id", "item_id", "cutoff"]) & set(feature_columns(train))
    assert not any(c.startswith("pair_") for c in feature_columns(train, ablate=True))
    assert np.isfinite(evaluation.select_dtypes(include="number")).all().all()
