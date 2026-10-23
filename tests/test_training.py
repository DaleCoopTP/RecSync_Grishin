from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRanker, Pool
from kion.config import Config
from kion.training import fit_grid, assert_validation_protocol


def tiny_ranking(cutoff, offset=0):
    rows = []
    for u in range(8):
        for item in range(6):
            rows.append({"user_id": u + offset, "item_id": item, "query_id": u + offset,
                         "cutoff": cutoff, "label": int(item == u % 6), "pair_match": int(item == u % 6),
                         "numeric": float(item), "category": str(item % 2)})
    return pd.DataFrame(rows)


def test_fit_select_save_reload_and_no_test(tmp_path):
    cfg = replace(Config(), depths=[3], l2_values=[3], iterations=8, early_stopping=3, threads=1)
    train = tiny_ranking(cfg.train_cutoffs[0])
    val = tiny_ranking(cfg.validation_cutoff, 20)
    targets = {u + 20: {u % 6} for u in range(8)}
    m = fit_grid(train, val, targets, cfg, tmp_path)
    assert m["test_evaluated"] is False
    a = CatBoostRanker(); a.load_model(str(tmp_path / "selected_model.cbm"))
    b = CatBoostRanker(); b.load_model(str(tmp_path / "depth3_l23.cbm"))
    pool = Pool(val[m["features"]], cat_features=m["categorical_features"])
    np.testing.assert_array_equal(a.predict(pool), b.predict(pool))
    val["cutoff"] = cfg.test_cutoff
    with pytest.raises(ValueError, match="never test"):
        assert_validation_protocol(train, val, cfg)
