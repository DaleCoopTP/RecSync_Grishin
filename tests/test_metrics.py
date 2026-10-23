import numpy as np
import pandas as pd
import pytest
from kion.metrics import user_metrics, evaluate_scores, bootstrap


def test_hand_computed_multiple_targets_and_missing_candidates():
    result = user_metrics([7, 8, 9], {7, 9, 99}, 3)
    assert result["recall@3"] == pytest.approx(2 / 3)
    assert result["map@3"] == pytest.approx((1 + 2 / 3) / 3)
    assert result["ndcg@3"] == pytest.approx(1.5 / (1 + 1 / np.log2(3) + 0.5))
    assert user_metrics([], {99})["ndcg@10"] == 0


def test_cold_item_and_empty_user_remain_in_denominator():
    frame = pd.DataFrame({"user_id": [1, 1], "item_id": [10, 11], "score": [1.0, 0.0]})
    summary, per = evaluate_scores(frame, {1: {10, 99}, 2: {100}}, "score")
    assert summary["users"] == 2
    assert summary["recall@10"] == 0.25
    assert summary["candidate_recall@200"] == 0.25
    assert per.loc[2, "candidate_count"] == 0
    ci = bootstrap(per, per, n=20)
    assert ci["ndcg@10"] == {"mean": 0.0, "lower": 0.0, "upper": 0.0}


def test_ties_resolve_by_item_id():
    f = pd.DataFrame({"user_id": [1, 1], "item_id": [11, 10], "score": [0.0, 0.0]})
    assert evaluate_scores(f, {1: {10}}, "score", k=1)[0]["ndcg@1"] == 1
