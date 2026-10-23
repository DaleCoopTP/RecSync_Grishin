import pandas as pd
from kion.config import Config
from kion.protocol import make_snapshot
from kion.features import FeatureStore
from kion.candidates import CandidateRetriever


def test_candidates_are_historical_unseen_and_label_independent(tables):
    e, u, i = tables
    cfg = Config()
    snap = make_snapshot(e, "2021-07-19")
    retriever = CandidateRetriever(FeatureStore(snap, u, i, cfg), cfg)
    frame = retriever.generate([1, 2])
    assert not frame.duplicated(["user_id", "item_id"]).any()
    assert set(frame.item_id) <= set(snap.catalog)
    assert not set(zip(frame.user_id, frame.item_id)) & set(zip(snap.history.user_id, snap.history.item_id))
    assert 14 not in set(frame.item_id)  # genuine future target is not injected
    changed = e.copy()
    changed.loc[changed.day >= retriever.store.day, "item_id"] = 100000
    b = CandidateRetriever(FeatureStore(make_snapshot(changed, "2021-07-19"), u, i, cfg), cfg).generate([1, 2])
    pd.testing.assert_frame_equal(frame, b)
    assert retriever.popular(999, 1) == [10]
