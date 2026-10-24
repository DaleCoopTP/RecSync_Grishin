import numpy as np
import pandas as pd
import pytest

from kion_autofeatures.model import fit_evaluate
from kion_autofeatures.search import choose_prefix, choose_users, prune_features


def test_training_only_pruning_and_small_prefix_ties():
    train = pd.DataFrame({"cand_pop_score": [0., 1., 2., 3.],
                          "auto__constant": [np.nan] * 4,
                          "auto__duplicate": [0., 1., 2., 3.],
                          "auto__new": [0., 1., 1., 0.],
                          "auto__same_new": [0., 1., 1., 0.]})
    kept, rejected = prune_features(train, [c for c in train if c.startswith("auto__")], ["cand_pop_score"])
    assert kept == ["auto__new"]
    assert "constant" in rejected["auto__constant"]
    assert rejected["auto__duplicate"] == "duplicate of cand_pop_score"
    assert rejected["auto__same_new"] == "duplicate of auto__new"
    records = [{"size": n, "metrics": {"ndcg@10": score}} for n, score in [(16, .2), (32, .20005), (64, .2002)]]
    assert choose_prefix(records, .0001)["size"] == 64
    assert choose_prefix(records[:2], .0001)["size"] == 16
    assert np.array_equal(choose_users([1, 2, 3, 4], 2, 42, "2021-07-19"),
                          choose_users([4, 3, 2, 1], 2, 42, "2021-07-19"))


def toy_frames():
    frame = pd.DataFrame({"user_id": [1, 1, 2, 2, 3, 3], "item_id": [10, 11, 10, 12, 11, 12],
                          "label": [1, 0, 0, 1, 1, 0], "query_id": [1, 1, 2, 2, 3, 3],
                          "cutoff": "2021-07-19", "cand_pop_score": [3., 2., 3., 1., 2., 1.],
                          "auto__user__count__all": [2., 2., 3., 3., 4., 4.],
                          "i_content_type": ["film", "series", "film", "film", "series", "film"]})
    validation = frame.copy()
    validation["cutoff"] = "2021-08-02"
    targets = {1: {10, 99}, 2: {12}, 3: {99}}  # preserve missing positives and zero-hit query
    return frame, validation, targets


def test_fixed_fit_reload_resume_and_evaluation_label_isolation(tmp_path, monkeypatch):
    from catboost import CatBoostRanker
    train, validation, targets = toy_frames()
    columns = ["cand_pop_score", "i_content_type", "auto__user__count__all"]
    parameters = {"loss_function": "YetiRank", "iterations": 4, "depth": 2,
                  "learning_rate": .05, "l2_leaf_reg": 3, "random_seed": 42,
                  "thread_count": 1, "max_ctr_complexity": 1}
    original = CatBoostRanker.fit
    calls = []

    def fit(self, *args, **kwargs):
        assert "eval_set" not in kwargs and "early_stopping_rounds" not in kwargs
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoostRanker, "fit", fit)
    record = fit_evaluate(train, validation, targets, columns, ["i_content_type"], parameters, tmp_path, {"test": True}, 2)
    assert record["reload_verified"] and not record["early_stopping"]
    assert record["metrics"]["users"] == 3
    per_user = pd.read_parquet(tmp_path / "per_user.parquet")
    assert per_user.loc[3, "ndcg@10"] == 0
    assert per_user.loc[1, "candidate_recall@200"] == .5
    assert fit_evaluate(train, validation.assign(label=1), targets, columns, ["i_content_type"],
                        parameters, tmp_path, {"test": True}, 2) == record
    assert len(calls) == 1
    (tmp_path / "model.cbm").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        fit_evaluate(train, validation, targets, columns, ["i_content_type"], parameters, tmp_path, {"test": True}, 2)
