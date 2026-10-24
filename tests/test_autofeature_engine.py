"""Independent, deliberately slow reference calculations on tiny histories."""
import numpy as np
import pandas as pd
import pytest

from kion.features import first_token, tokens, MISSING
from kion.protocol import day_number
from kion_autofeatures.baseline import attach_features
from kion_autofeatures.engine import HistoryFeatures, cache_features
from kion_autofeatures.grammar import FeatureSpec, enumerate_features


def reference(events, users, items, uid, iid, cutoff, spec):
    day = day_number(cutoff)
    history = events.loc[events.day < day]
    um, im = users.set_index("user_id"), items.set_index("item_id")

    def category(table, key, col):
        value = table.loc[key, col] if key in table.index else None
        if col == "genres":
            return tokens(value)
        if col in ("directors", "countries", "content_type"):
            return first_token(value)
        return MISSING if pd.isna(value) else str(value)

    def aggregate(path, window, op):
        root = path.split("_", 1)[0]
        values, weights, days = [], [], []
        for row in history.itertuples():
            if (row.user_id if root == "user" else row.item_id) != (uid if root == "user" else iid):
                continue
            if window and row.day < day - window:
                continue
            weight = 1.
            if "_" in path:
                col = path.split("_", 1)[1]
                table, old, query = (im, row.item_id, iid) if root == "user" else (um, row.user_id, uid)
                left, right = category(table, old, col), category(table, query, col)
                if col == "genres":
                    weight = len(set(left) & set(right)) / (len(left) * len(right))
                else:
                    weight = float(left == right)
            if weight == 0:
                continue
            weights.append(weight)
            days.append(row.day)
            values.append(np.log1p(row.total_dur) if "dur" in op else row.watched_pct)
        if op == "count":
            return sum(weights)
        if op == "recency":
            return day - max(days) if days else np.nan
        valid = [(v, w) for v, w in zip(values, weights) if np.isfinite(v)]
        if op == "sum_log_dur":
            return sum(v * w for v, w in valid)
        if not valid:
            return np.nan
        if op == "max_pct":
            return max(v for v, _ in valid)
        return sum(v * w for v, w in valid) / sum(w for _, w in valid)

    if spec.operation == "share":
        total = aggregate(spec.path.split("_", 1)[0], spec.window, "count")
        return aggregate(spec.path, spec.window, "count") / total if total else 0
    if spec.operation == "trend":
        return (aggregate(spec.path, spec.window, "count") + 1) / (aggregate(spec.path, spec.long_window, "count") + 1)
    return aggregate(spec.path, spec.window, spec.operation)


def test_all_features_match_independent_reference(tables):
    events, users, items = (x.copy() for x in tables)
    events.loc[0, "watched_pct"] = np.nan
    events.loc[1, "total_dur"] = np.nan
    events.loc[4, "watched_pct"] = 70.
    pairs = pd.DataFrame({"user_id": [1, 2, 4, 1], "item_id": [12, 11, 10, 10]})
    cutoff = "2021-07-19"
    specs = enumerate_features()
    result = HistoryFeatures(events, users, items, cutoff).transform(pairs, specs)
    expected = np.array([[reference(events, users, items, u, i, cutoff, s) for s in specs]
                         for u, i in pairs.itertuples(index=False, name=None)])
    np.testing.assert_allclose(result[[s.name for s in specs]].to_numpy(), expected, rtol=2e-6, atol=2e-6, equal_nan=True)


def test_future_labels_and_pair_service_fields_are_isolated(tables):
    events, users, items = tables
    pairs = pd.DataFrame({"user_id": [1, 2], "item_id": [12, 11]})
    specs = enumerate_features()
    cutoff = "2021-07-19"
    expected = HistoryFeatures(events, users, items, cutoff).transform(pairs, specs)
    future = events.loc[events.day >= day_number(cutoff)].copy()
    future["total_dur"] = 1e9
    future["watched_pct"] = 999
    changed = pd.concat([events.loc[events.day < day_number(cutoff)], future, future], ignore_index=True)
    actual = HistoryFeatures(changed.assign(label=1), users, items, cutoff).transform(pairs, specs)
    pd.testing.assert_frame_equal(actual, expected)
    for label in (0, 1):
        base = pairs.assign(label=label, query_id=[1, 2], cutoff=cutoff)
        attached = attach_features(base, actual)
        pd.testing.assert_frame_equal(attached[base.columns], base)
    with pytest.raises(ValueError, match="never labels"):
        HistoryFeatures(events, users, items, cutoff).transform(pairs.assign(label=1), specs)


def test_window_bounds_empty_relations_and_missing_metadata(tables):
    events, users, items = tables
    events = events.copy()
    events.loc[0, "day"] = day_number("2021-07-16")
    pairs = pd.DataFrame({"user_id": [1, 4], "item_id": [12, 10]})
    engine = HistoryFeatures(events, users, items, "2021-07-19")
    specs = [FeatureSpec("user", "count", 3), FeatureSpec("user", "count", 7),
             FeatureSpec("user_directors", "count", 3), FeatureSpec("user_directors", "recency"),
             FeatureSpec("user_directors", "mean_pct", 3), FeatureSpec("item_age", "count")]
    actual = engine.transform(pairs, specs)
    assert actual[specs[0].name].tolist() == [2., 0.]  # July 16 included, July 19 excluded
    assert actual[specs[2].name].tolist() == [1., 0.]
    assert actual[specs[3].name].iloc[0] == 3
    assert np.isnan(actual[specs[3].name].iloc[1])
    assert np.isnan(actual[specs[4].name].iloc[1])
    assert actual[specs[5].name].iloc[1] == 0  # missing user-age bucket has no item-10 events


def test_cache_batching_reload_and_corruption(tables, tmp_path):
    events, users, items = tables
    pairs = pd.DataFrame({"user_id": [1, 2], "item_id": [12, 11]})
    specs = enumerate_features()
    engine = HistoryFeatures(events, users, items, "2021-07-19")
    path = tmp_path / "features.parquet"
    first = cache_features(engine, pairs, specs, path, batch_rows=1)
    pd.testing.assert_frame_equal(pd.read_parquet(path), engine.transform(pairs, specs))
    assert cache_features(engine, pairs, specs, path, batch_rows=2) == first
    with pytest.raises(ValueError, match="incompatible"):
        cache_features(engine, pairs.iloc[::-1], specs, path)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        cache_features(engine, pairs, specs, path)
