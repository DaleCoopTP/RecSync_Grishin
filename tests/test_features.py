import pandas as pd
import numpy as np
from kion.config import Config
from kion.protocol import make_snapshot
from kion.features import FeatureStore


def test_aggregates_and_fractional_genres(tables):
    e, u, i = tables
    f = FeatureStore(make_snapshot(e, "2021-07-19"), u, i, Config())
    assert f.user_features.loc[1, "u_count_all"] == 2
    assert f.user_features.loc[1, "u_count_7"] == 1
    assert f.user_features.loc[1, "u_recency"] == 1
    assert f.item_features.loc[10, "i_count_all"] == 2
    pos = f.user_index.get_loc(1)
    assert f.user_genre["all"][pos, f.genres.index("comedy")] == 1.5
    assert f.user_genre["all"][pos].sum() == 2


def test_future_changes_cannot_change_features(tables):
    e, u, i = tables
    a = FeatureStore(make_snapshot(e, "2021-07-19"), u, i, Config())
    future = e.loc[e.day >= a.day].copy()
    future["total_dur"] = 999999
    altered = pd.concat([e.loc[e.day < a.day], future])
    b = FeatureStore(make_snapshot(altered, "2021-07-19"), u, i, Config())
    pd.testing.assert_frame_equal(a.user_features, b.user_features)
    pd.testing.assert_frame_equal(a.item_features, b.item_features)
    np.testing.assert_array_equal(a.user_genre["all"], b.user_genre["all"])
