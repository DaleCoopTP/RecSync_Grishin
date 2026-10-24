from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRanker
from kion.config import Config
from kion.io import write_json
from kion.pipeline import run_validation
from kion.final import lock_test, load_protocol, run_final, assert_final_training, verify_files
from scripts.freeze_validation import freeze


def synthetic_tables(cfg):
    rows = []
    dates = [f'2021-05-{j + 1:02d}' for j in range(10)] + cfg.train_cutoffs + [cfg.validation_cutoff, cfg.test_cutoff]
    for user in range(20):
        for j, date in enumerate(dates):
            item = 999 if user == 1 and date == cfg.test_cutoff else (user + j) % 30
            rows.append((user, item, date, 600., 80.))
    rows.append((999, 1, cfg.test_cutoff, 600., 80.))  # Cold user remains a diagnostic, outside warm-active quality.
    events = pd.DataFrame(rows, columns=['user_id', 'item_id', 'last_watch_dt', 'total_dur', 'watched_pct'])
    events['last_watch_dt'] = pd.to_datetime(events.last_watch_dt)
    events['day'] = (events.last_watch_dt - pd.Timestamp('1970-01-01')).dt.days.astype('int32')
    users = pd.DataFrame({'user_id': range(20), 'age': ['young'] * 20, 'income': ['low'] * 20,
                          'sex': ['F'] * 20, 'kids_flg': [0] * 20})
    items = pd.DataFrame({'item_id': range(30), 'genres': ['drama, comedy'] * 30, 'countries': ['RU'] * 30,
        'directors': [str(i % 3) for i in range(30)], 'content_type': ['film'] * 30,
        'release_year': [2020] * 30, 'for_kids': [0] * 30, 'age_rating': [16] * 30})
    return events, users, items


@pytest.fixture
def locked(tmp_path):
    cfg = replace(Config(root=tmp_path), sample_users=20, train_users_per_cutoff=20, threads=1,
                  depths=[2], l2_values=[3], iterations=4, early_stopping=2, batch_users=7, bootstrap_samples=20)
    tables = synthetic_tables(cfg)
    run_validation(cfg, tables)
    write_json(cfg.path('results_dir') / 'sample_manifest.json',
               {'sample_users': 20, 'seed': cfg.seed, 'sample_rows': len(tables[0]), 'user_ids_sha256': 'synthetic'})
    freeze(cfg)
    lock_test(cfg)
    return cfg, tables


def test_locked_refit_never_uses_test_for_fit_and_resume_keeps_cold_targets(locked, monkeypatch):
    import kion.final as final
    cfg, tables = locked
    calls = []
    original_fit = CatBoostRanker.fit
    original_snapshot = final.make_snapshot
    def checked_fit(self, pool, *args, **kwargs):
        assert 'eval_set' not in kwargs and 'early_stopping_rounds' not in kwargs
        assert kwargs['use_best_model'] is False
        assert self.get_params()['iterations'] == json.loads((cfg.path('results_dir') / 'frozen_validation.json').read_text())['parameters']['iterations']
        calls.append(pool.num_row())
        return original_fit(self, pool, *args, **kwargs)
    def checked_snapshot(events, cutoff, horizon):
        if cutoff == cfg.test_cutoff:
            assert len(calls) == 2, 'Both checkpoints must be trained before test labels are constructed'
        return original_snapshot(events, cutoff, horizon)
    monkeypatch.setattr(CatBoostRanker, 'fit', checked_fit)
    monkeypatch.setattr(final, 'make_snapshot', checked_snapshot)
    result = run_final(cfg, tables)
    assert len(calls) == 2
    assert result['test_evaluated'] is True and result['test_snapshot']['cold_active_users'] == 1
    assert result['metrics']['catboost']['users'] == 20
    out = cfg.root / result['artifact_directory']
    per = pd.read_parquet(out / 'catboost_per_user.parquet')
    assert per.loc[1, 'candidate_recall@200'] == 0 and per.loc[1, 'ndcg@10'] == 0
    assert len(result['training_snapshots']) == 4
    assert cfg.test_cutoff not in [s['cutoff'] for s in result['training_snapshots']]
    assert all(not r['test_labels_used'] and not r['early_stopping'] for r in result['models'].values())
    ci = json.loads((cfg.path('results_dir') / 'test_confidence_intervals.json').read_text())
    assert ci['reference_selected_on_validation'] == result['reference_method']
    for interval in ci['paired_catboost_minus'][result['reference_method']].values():
        assert interval['lower'] <= interval['upper'] and np.isfinite(interval['mean'])
    def unexpected_fit(*args, **kwargs):
        raise AssertionError('A completed final run must never refit')
    monkeypatch.setattr(CatBoostRanker, 'fit', unexpected_fit)
    assert run_final(cfg, tables)['models'] == result['models']
    from kion.inference import recommend_saved
    monkeypatch.setattr('kion.inference.load_prepared', lambda config: tables)
    recommended = recommend_saved(cfg, 0, k=10, stage='test')
    assert recommended['as_of'] == cfg.test_cutoff and recommended['stage'] == 'test'
    assert len(recommended['item_ids']) == len(set(recommended['item_ids']))
    seen = set(tables[0].loc[(tables[0].user_id == 0) & (tables[0].last_watch_dt < cfg.test_cutoff), 'item_id'])
    assert not seen.intersection(recommended['item_ids'])
    assert recommend_saved(cfg, 999, k=10)['method'] == 'popularity_fallback'
    with pytest.raises(ValueError, match='Supported length'):
        recommend_saved(cfg, 0, k=0)
    path = cfg.path('results_dir') / 'test_results.csv'
    path.write_text(path.read_text() + '\ncorruption')
    with pytest.raises(ValueError, match='results checksum'):
        verify_files(cfg, result)


def test_final_training_rejects_test_cutoff_and_lock_rejects_changed_config(locked):
    cfg, _ = locked
    bad = pd.DataFrame({'cutoff': cfg.train_cutoffs + [cfg.test_cutoff], 'label': [0, 1, 0, 1]})
    with pytest.raises(ValueError, match='never test'):
        assert_final_training(bad, cfg)
    with pytest.raises(ValueError, match='configuration mismatch'):
        load_protocol(replace(cfg, learning_rate=0.1))
    protocol_path = cfg.path('results_dir') / 'final_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    first = next(iter(protocol['invariant_sources']))
    protocol['invariant_sources'][first] = 'changed'
    write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match='source changed'):
        load_protocol(cfg)
