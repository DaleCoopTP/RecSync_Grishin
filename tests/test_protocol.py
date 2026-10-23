import numpy as np
from kion.protocol import sample_users, make_snapshot


def test_sample_keeps_every_row_for_selected_users(tables):
    e, _, _ = tables
    a, ids = sample_users(e, 2, 42)
    b, ids2 = sample_users(e.sample(frac=1, random_state=8), 2, 42)
    assert np.array_equal(ids, ids2)
    assert a.equals(b)
    assert len(a) == e.user_id.isin(ids).sum()


def test_temporal_boundaries_and_cold_targets(tables):
    e, _, _ = tables
    s = make_snapshot(e, "2021-07-19")
    assert s.history.last_watch_dt.max().strftime("%Y-%m-%d") == "2021-07-18"
    assert s.targets[1] == {12}
    assert s.targets[2] == {14}
    assert 3 not in s.targets
    assert 14 not in s.catalog
    assert s.diagnostics["cold_active_users"] == 1
    assert s.diagnostics["warm_user_new_item_targets"] == 1
