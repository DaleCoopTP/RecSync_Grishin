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


def test_resume_skips_completed_fits_and_rejects_changed_data(tmp_path, monkeypatch):
    cfg = replace(Config(), depths=[2], l2_values=[3], iterations=4, early_stopping=2, threads=1)
    train = tiny_ranking(cfg.train_cutoffs[0])
    val = tiny_ranking(cfg.validation_cutoff, 20)
    targets = {u + 20: {u % 6} for u in range(8)}
    first = fit_grid(train, val, targets, cfg, tmp_path)
    def unexpected_fit(*args, **kwargs):
        raise AssertionError('Completed checkpoint must be reused')
    monkeypatch.setattr(CatBoostRanker, 'fit', unexpected_fit)
    again = fit_grid(train, val, targets, cfg, tmp_path)
    assert first['selected'] == again['selected']
    train.loc[0, 'numeric'] += 1
    with pytest.raises(ValueError, match='different data'):
        fit_grid(train, val, targets, cfg, tmp_path)


def test_prefix_uses_only_budget_and_preserves_original(tmp_path):
    from kion.training import model_pool, reuse_prefix
    from kion.dataset import feature_columns, categorical_columns
    from kion.io import sha256
    cfg = replace(Config(), depths=[3], l2_values=[3], iterations=12, early_stopping=100, threads=1)
    train = tiny_ranking(cfg.train_cutoffs[0])
    val = tiny_ranking(cfg.validation_cutoff, 20)
    columns = feature_columns(train); cats = categorical_columns(train, columns)
    pool = model_pool(train, columns, cats)
    val_pool = model_pool(val, columns, cats)
    source = CatBoostRanker(loss_function='YetiRank', eval_metric='NDCG:top=10', iterations=12,
        depth=3, l2_leaf_reg=3, learning_rate=cfg.learning_rate, random_seed=cfg.seed,
        thread_count=1, allow_writing_files=False, max_ctr_complexity=1)
    source.fit(pool, eval_set=val_pool, use_best_model=False, verbose=False)
    path = tmp_path / 'original.cbm'; source.save_model(str(path)); checksum = sha256(path)
    budget = replace(cfg, iterations=3, threads=2)
    prefix, used, provenance = reuse_prefix(path, columns, cats, budget, 3, 3)
    curve = next(v for k, v in source.get_evals_result()['validation'].items() if k.startswith('NDCG'))
    expected_trees = int(np.argmax(curve[:3])) + 1
    assert used == 3 and prefix.tree_count_ == expected_trees
    assert provenance['source_trained_iterations'] == 12 and sha256(path) == checksum
    np.testing.assert_array_equal(prefix.predict(val_pool), source.predict(val_pool, ntree_end=expected_trees))
    fresh = CatBoostRanker(**{**source.get_params(), 'iterations': 3, 'thread_count': 2})
    fresh.fit(pool, eval_set=val_pool, use_best_model=False, verbose=False)
    np.testing.assert_allclose(prefix.predict(val_pool), fresh.predict(val_pool, ntree_end=expected_trees), rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match='parameter mismatch'):
        reuse_prefix(path, columns, cats, budget, 4, 3)
