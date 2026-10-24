from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRanker

from kion.config import Config
from kion.io import sha256
from kion.metrics import evaluate_scores
from kion.protocol import day_number
from kion.training import model_pool
from scripts.feature_ablation import (benchmark_variant, configure, feature_groups, fit_variant,
                                     summarize, verify_plan)


def test_feature_families_exclude_identifiers_and_partition_baseline():
    frozen = json.loads((Config().root / 'results/frozen_validation.json').read_text())
    groups = feature_groups(frozen['features'])
    assert len(groups['source_only']) == 13 and len(groups['manual_only']) == 69
    assert set(groups['source_only']).isdisjoint(groups['manual_only'])
    assert set(groups['full']) == set(groups['source_only']) | set(groups['manual_only'])
    with pytest.raises(ValueError, match='service fields'):
        feature_groups(frozen['features'] + ['label'])
    with pytest.raises(ValueError, match='service fields'):
        feature_groups(frozen['features'] + ['query_id'])


def test_fixed_ablation_roundtrip_preserves_candidates_and_never_uses_eval_set(tmp_path, monkeypatch):
    cfg = replace(Config(), root=tmp_path, threads=1, bootstrap_samples=12)
    configure(cfg)
    out = tmp_path / 'artifacts/diagnostic'
    out.mkdir(parents=True)
    columns = ['cand_pop_score', 'rrf_score', 'u_count_all', 'i_genres', 'pair_genre_share_all']
    categories = ['i_genres']
    groups = feature_groups(columns)
    snapshots = []
    frames = []
    for cutoff in cfg.train_cutoffs + [cfg.validation_cutoff]:
        frame = pd.DataFrame([
            {'user_id': u, 'item_id': item, 'query_id': day_number(cutoff) * 100 + u,
             'cutoff': cutoff, 'label': int(item == u % 6 and not (cutoff == cfg.validation_cutoff and u == 7)),
             'cand_pop_score': float(item),
             'rrf_score': 1 / (item + 1), 'u_count_all': u + 1,
             'i_genres': str(item % 2), 'pair_genre_share_all': float(item == u % 6)}
            for u in range(8) for item in range(6)
        ])
        path = out / f'{cutoff}.parquet'
        frame.to_parquet(path, index=False)
        snapshots.append({'cutoff': cutoff, 'path': str(path.relative_to(tmp_path)),
                          'rows': len(frame), 'sha256': sha256(path)})
        frames.append(frame)
    train = pd.concat(frames[:3], ignore_index=True)
    val = frames[3]
    targets = {u: {u % 6} for u in range(8)}
    targets[7] = {9999}  # A missing positive must remain in the evaluation denominator.
    params = {'loss_function': 'YetiRank', 'iterations': 4, 'depth': 2, 'l2_leaf_reg': 3,
              'learning_rate': .05, 'random_seed': 42, 'thread_count': 1, 'max_ctr_complexity': 1}
    full = CatBoostRanker(**params, allow_writing_files=False)
    full.fit(model_pool(train, columns, categories), verbose=False)
    full_path = out / 'original.cbm'
    full.save_model(str(full_path))
    metrics, _ = evaluate_scores(val[['user_id', 'item_id']].assign(
        score=full.predict(model_pool(val, columns, categories))), targets, 'score')
    events = pd.DataFrame([
        {'user_id': u, 'item_id': 99, 'day': day_number(cfg.validation_cutoff) - 1} for u in range(8)
    ] + [{'user_id': u, 'item_id': next(iter(targets[u])), 'day': day_number(cfg.validation_cutoff)} for u in range(8)]
      + [{'user_id': 999, 'item_id': 999, 'day': day_number(cfg.test_cutoff)}])
    events_path = out / 'events.parquet'
    events.to_parquet(events_path, index=False)
    protected = {s['path']: s['sha256'] for s in snapshots}
    protected.update({str(full_path.relative_to(tmp_path)): sha256(full_path),
                      str(events_path.relative_to(tmp_path)): sha256(events_path)})
    plan = {'config_fingerprint': cfg.fingerprint(), 'feature_groups': groups,
            'parameters': params, 'categorical_features': categories,
            'training_snapshots': snapshots[:3], 'validation_snapshot': snapshots[3],
            'events_path': str(events_path.relative_to(tmp_path)), 'protected_files': protected,
            'full_model': {'path': str(full_path.relative_to(tmp_path)), 'sha256': sha256(full_path),
                           'metrics': metrics}, 'artifact_directory': str(out.relative_to(tmp_path)),
            'inference_repetitions': 2, 'bootstrap_samples': 12, 'seed': 42, 'scope': 'test fixture'}
    original_fit = CatBoostRanker.fit
    calls = []

    def guarded_fit(model, pool, *args, **kwargs):
        assert not args and 'eval_set' not in kwargs and 'early_stopping_rounds' not in kwargs
        assert model.get_params()['iterations'] == 4
        calls.append(model.get_params())
        return original_fit(model, pool, **kwargs)

    monkeypatch.setattr(CatBoostRanker, 'fit', guarded_fit)
    import kion.protocol
    original_snapshot = kion.protocol.make_snapshot

    def validation_snapshot(events, cutoff, horizon):
        assert cutoff == cfg.validation_cutoff
        assert events.day.max() < day_number(cfg.test_cutoff)
        return original_snapshot(events, cutoff, horizon)

    monkeypatch.setattr(kion.protocol, 'make_snapshot', validation_snapshot)
    for variant in ['source_only', 'manual_only']:
        fit_variant(cfg, plan, variant)
        fit_variant(cfg, plan, variant)  # Resume must not fit twice.
    assert len(calls) == 2
    for variant in ['source_only', 'manual_only', 'full']:
        receipt = benchmark_variant(cfg, plan, variant)
        assert receipt['metrics']['users'] == 8 and receipt['loaded_model_verified']
        assert not receipt['test_evaluated']
    summary = summarize(cfg, plan)
    assert summary.users.eq(8).all() and summary['candidate_recall@200'].eq(7 / 8).all()
    assert np.isfinite(summary['ndcg@10']).all()
    pd.testing.assert_series_equal(
        pd.read_parquet(out / 'source_only_per_user.parquet').candidate_count,
        pd.read_parquet(out / 'manual_only_per_user.parquet').candidate_count)
    verify_plan(cfg, plan)
    with pytest.raises(ValueError, match='Only the two new subsets'):
        fit_variant(cfg, plan, 'full')
    # A changed cached input must be rejected, including when a model is already fitted.
    with events_path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Protected input'):
        fit_variant(cfg, plan, 'source_only')
