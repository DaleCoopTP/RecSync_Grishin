"""Locked refit and held-out evaluation. Test labels never enter model.fit."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import resource
import subprocess
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostRanker, Pool
from .dataset import iter_dataset, feature_columns, categorical_columns
from .io import sha256, write_json
from .metrics import BASELINES, METRICS, evaluate_baselines, evaluate_scores, bootstrap
from .pipeline import cache_snapshot, compact, context, code_fingerprint
from .protocol import load_prepared, prepared_dir, make_snapshot, day_number
from .training import model_pool


def lock_test(cfg):
    """Seal a fresh validation experiment before any held-out evaluation."""
    destination = cfg.path('results_dir') / 'final_protocol.json'
    if destination.exists():
        return load_protocol(cfg)[0]
    frozen_path = cfg.path('results_dir') / 'frozen_validation.json'
    frozen = json.loads(frozen_path.read_text())
    if frozen['code_fingerprint'] != code_fingerprint():
        raise ValueError('Pipeline changed since validation; restore matching sources before locking test')
    run = json.loads((cfg.path('results_dir') / 'validation_manifest.json').read_text())
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cfg.root, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    sources = sorted(Path(__file__).parent.glob('*.py'))
    def source_name(path):
        return str(path.relative_to(cfg.root)) if path.is_relative_to(cfg.root) else str(path)
    protocol = {'version': 1, 'locked_at_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_validation_sha256': sha256(frozen_path), 'config_fingerprint': cfg.fingerprint(),
        'selection_source_commit': commit,
        'invariant_sources': {source_name(p): sha256(p) for p in sources if p.name != 'cli.py'},
        'training_cutoffs': cfg.train_cutoffs + [cfg.validation_cutoff], 'test_cutoff': cfg.test_cutoff,
        'reference_method': max(run['baselines'], key=lambda k: run['baselines'][k]['ndcg@10']),
        'bootstrap': {'samples': cfg.bootstrap_samples, 'seed': cfg.seed, 'unit': 'user', 'confidence': 0.95, 'method': 'percentile'},
        'models': ['catboost', 'catboost_without_personal_features'], 'train_users_per_cutoff': cfg.train_users_per_cutoff,
        'selection_note': 'Parameters, comparator and analysis locked before test; no test early stopping or tuning.'}
    write_json(destination, protocol)
    load_protocol(cfg)
    return protocol


def load_protocol(cfg):
    folder = cfg.path('results_dir')
    protocol = json.loads((folder / 'final_protocol.json').read_text())
    frozen_path = folder / 'frozen_validation.json'
    frozen = json.loads(frozen_path.read_text())
    if (protocol['frozen_validation_sha256'] != sha256(frozen_path)
            or protocol['config_fingerprint'] != cfg.fingerprint()
            or frozen['config_fingerprint'] != cfg.fingerprint()):
        raise ValueError('Locked protocol/configuration mismatch')
    if (frozen['selection_split'] != 'validation' or frozen['test_evaluated']
            or frozen['validation_cutoff'] != cfg.validation_cutoff
            or frozen['test_cutoff_reserved'] != cfg.test_cutoff
            or protocol['test_cutoff'] != cfg.test_cutoff
            or protocol['training_cutoffs'] != cfg.train_cutoffs + [cfg.validation_cutoff]):
        raise ValueError('Invalid final temporal protocol')
    expected = {'loss_function': 'YetiRank', 'learning_rate': cfg.learning_rate,
                'random_seed': cfg.seed, 'thread_count': cfg.threads, 'max_ctr_complexity': 1}
    if any(frozen['parameters'][k] != v for k, v in expected.items()):
        raise ValueError('Frozen training parameters mismatch')
    if (frozen['parameters']['depth'] not in cfg.depths or frozen['parameters']['l2_leaf_reg'] not in cfg.l2_values
            or not 1 <= frozen['parameters']['iterations'] <= cfg.iterations
            or protocol['reference_method'] not in BASELINES
            or protocol['bootstrap']['samples'] != cfg.bootstrap_samples or protocol['bootstrap']['seed'] != cfg.seed):
        raise ValueError('Invalid fixed parameters or comparison plan')
    for name, checksum in protocol['invariant_sources'].items():
        if sha256(cfg.root / name) != checksum:
            raise ValueError(f'Feature/protocol source changed since selection: {name}')
    if sha256(cfg.root / frozen['model_artifact']) != frozen['model_sha256']:
        raise ValueError('Frozen validation model checksum mismatch')
    return protocol, frozen


def assert_final_training(train, cfg):
    allowed = cfg.train_cutoffs + [cfg.validation_cutoff]
    if set(train.cutoff.unique()) != set(allowed):
        raise ValueError('Final training must contain exactly the four historical cutoffs, never test')
    if any(day_number(c) + cfg.horizon_days > day_number(cfg.test_cutoff) for c in allowed):
        raise ValueError('Training labels reach the held-out test window')
    if train.label.nunique() < 2:
        raise ValueError('Final training requires positive and negative candidates')


def fit_fixed(train, cfg, frozen, out, name, ablate, signature):
    assert_final_training(train, cfg)
    columns = feature_columns(train, ablate=ablate)
    cats = categorical_columns(train, columns)
    if not ablate and (columns != frozen['features'] or cats != frozen['categorical_features']):
        raise ValueError('Final features differ from the frozen validation features')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    path = out / f'{name}.cbm'
    receipt = out / f'{name}.json'
    identity = {'parameters': frozen['parameters'], 'features': columns, 'categorical_features': cats,
                'training_cutoffs': cfg.train_cutoffs + [cfg.validation_cutoff], 'inputs': signature}
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record['identity'] != identity or sha256(path) != record['model_sha256']:
            raise ValueError('Final checkpoint identity/checksum mismatch')
        print(f'Using completed final {name}', flush=True)
        return record
    print(f'Final refit {name}: {len(train):,} rows, {len(columns)} features, {cfg.threads} threads, '
          f'{frozen["parameters"]["iterations"]} fixed trees; no eval_set', flush=True)
    start = time.perf_counter()
    model = CatBoostRanker(**frozen['parameters'], allow_writing_files=False, used_ram_limit='6gb')
    pool = model_pool(train, columns, cats, cfg.threads)
    model.fit(pool, verbose=10, use_best_model=False)
    temporary = path.with_suffix('.tmp.cbm')
    model.save_model(str(temporary)); temporary.replace(path)
    record = {'identity': identity, 'model_sha256': sha256(path), 'seconds': time.perf_counter() - start,
              'training_rows': len(train), 'training_queries': train.query_id.nunique(), 'tree_count': model.tree_count_,
              'test_labels_used': False, 'early_stopping': False}
    if model.tree_count_ != frozen['parameters']['iterations']:
        raise ValueError('Final refit did not produce the fixed number of trees')
    write_json(receipt, record)
    del pool, model
    gc.collect()
    return record


def test_cache(snapshot, users, items, cfg, out):
    path = out / f'{snapshot.cutoff}_test.parquet'; info = path.with_suffix('.json')
    if path.exists() and info.exists():
        meta = json.loads(info.read_text())
        if sha256(path) != meta['sha256']:
            raise ValueError('Test cache checksum mismatch')
        return path, meta
    started = time.perf_counter(); rows = queries = positives = 0
    temporary = path.with_suffix('.tmp'); writer = None
    print('Building held-out test candidates after both refits are complete', flush=True)
    try:
        for frame in iter_dataset(snapshot, users, items, cfg, training=False):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
            writer.write_table(table)
            rows += len(frame); queries += frame.query_id.nunique(); positives += int(frame.label.sum())
        if writer is None:
            raise ValueError('No held-out test candidates')
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)
    meta = {**snapshot.diagnostics, 'role': 'test', 'training': False, 'rows': rows, 'queries': queries,
            'positive_candidates': positives, 'sha256': sha256(path), 'seconds': time.perf_counter() - started}
    write_json(info, meta)
    return path, meta


def predict_saved(frame, cfg, path, receipt):
    if sha256(path) != receipt['model_sha256']:
        raise ValueError('Saved final model checksum mismatch')
    identity = receipt['identity']
    model = CatBoostRanker(); model.load_model(str(path))
    return model.predict(Pool(frame[identity['features']], cat_features=identity['categorical_features'],
                              thread_count=cfg.threads), thread_count=cfg.threads)


def save_test_analysis(cfg, out, frame, snapshot, protocol, receipts):
    scored = frame[['user_id', 'item_id']].copy()
    summaries, per_user = evaluate_baselines(frame[['user_id', 'item_id', *BASELINES.values()]], snapshot.targets)
    for name, receipt in receipts.items():
        scored[name] = predict_saved(frame, cfg, out / 'models' / f'{name}.cbm', receipt)
        summaries[name], per_user[name] = evaluate_scores(scored[['user_id', 'item_id', name]], snapshot.targets, name)
    scored.to_parquet(out / 'test_predictions.parquet', index=False)
    absolute = {}; paired = {}
    for name, table in per_user.items():
        table.to_parquet(out / f'{name}_per_user.parquet')
        absolute[name] = bootstrap(table, n=cfg.bootstrap_samples, seed=cfg.seed)
    for name in per_user:
        if name != 'catboost':
            paired[name] = bootstrap(per_user['catboost'], per_user[name], n=cfg.bootstrap_samples, seed=cfg.seed)
    ci = {'plan': protocol['bootstrap'], 'reference_selected_on_validation': protocol['reference_method'],
          'absolute': absolute, 'paired_catboost_minus': paired,
          'interpretation': 'Percentile intervals resample users with fixed trained models; exclude refit and temporal uncertainty.'}
    records = []
    for name, metrics in summaries.items():
        record = {'method': name, **metrics}
        for metric in METRICS:
            record[f'{metric}_lower'] = absolute[name][metric]['lower']
            record[f'{metric}_upper'] = absolute[name][metric]['upper']
        records.append(record)
    results = cfg.path('results_dir')
    pd.DataFrame(records).to_csv(results / 'test_results.csv', index=False)
    write_json(results / 'test_confidence_intervals.json', ci)
    history_counts = snapshot.history.groupby('user_id').size().reindex(per_user['catboost'].index)
    coverage = per_user['catboost']['candidate_recall@200']
    masks = {'history_1_2': history_counts <= 2, 'history_3_9': history_counts.between(3, 9),
             'history_10_plus': history_counts >= 10, 'retrieval_zero_hits': coverage == 0, 'retrieval_some_hits': coverage > 0}
    segments = []
    for segment, mask in masks.items():
        for name, table in per_user.items():
            selected = table.loc[mask]
            if len(selected):
                segments.append({'segment': segment, 'method': name, 'users': len(selected),
                                 **{m: selected[m].mean() for m in [*METRICS, 'candidate_recall@200']}})
    pd.DataFrame(segments).to_csv(results / 'test_segments.csv', index=False)
    return summaries


def run_final(cfg, tables=None):
    started = time.perf_counter()
    protocol, frozen = load_protocol(cfg)
    pa.set_cpu_count(cfg.threads); pa.set_io_thread_count(cfg.threads)
    if tables is None:
        tables = load_prepared(cfg)
        sample = json.loads((prepared_dir(cfg) / 'manifest.json').read_text())
        if sample['user_ids_sha256'] != frozen['user_ids_sha256']:
            raise ValueError('Final sample differs from validation sample')
    out, run = context(cfg, tables)
    if run['data_fingerprint'] != frozen['data_fingerprint'] or run['versions'] != frozen['versions'] or run['python'] != frozen['python']:
        raise ValueError('Final data or runtime differs from validation')
    protocol_path = cfg.path('results_dir') / 'final_protocol.json'
    existing = cfg.path('results_dir') / 'test_manifest.json'
    if existing.exists():
        result = json.loads(existing.read_text())
        verify_files(cfg, result)
        if result['data_fingerprint'] != run['data_fingerprint']:
            raise ValueError('Completed test belongs to different data')
        print('Final test already completed; using immutable results, no refitting', flush=True)
        return result
    out = out / 'final' / sha256(protocol_path)[:12]; out.mkdir(parents=True, exist_ok=True)
    if (out / 'test_manifest.json').exists():
        result = json.loads((out / 'test_manifest.json').read_text())
        verify_files(cfg, result)
        print('Final test already completed; using immutable results, no refitting', flush=True)
        return result
    events, users, items = tables
    paths = []; snapshots = []
    for cached in frozen['candidate_snapshots']:
        if cached['training']:
            path = cfg.root / cached['path']
            if sha256(path) != cached['sha256']:
                raise ValueError('Historical training cache checksum mismatch')
            paths.append(path); snapshots.append(cached)
    expected = set(cfg.train_cutoffs)
    if {s['cutoff'] for s in snapshots} != expected:
        raise ValueError('Frozen training snapshots do not match the protocol')
    addition = make_snapshot(events, cfg.validation_cutoff, cfg.horizon_days)
    path, meta = cache_snapshot(addition, users, items, cfg, out, True)
    paths.append(path); snapshots.append({**meta, 'path': str(path.relative_to(cfg.root))})
    train = compact(pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True))
    signature = [s['sha256'] for s in snapshots]
    receipts = {}
    for name, ablate in [('catboost', False), ('catboost_without_personal_features', True)]:
        receipts[name] = fit_fixed(train, cfg, frozen, out / 'models', name, ablate, signature)
    del train, addition
    gc.collect()
    # The held-out snapshot/labels are constructed only after both models have been sealed.
    snapshot = make_snapshot(events, cfg.test_cutoff, cfg.horizon_days)
    path, test_info = test_cache(snapshot, users, items, cfg, out)
    frame = compact(pd.read_parquet(path))
    summaries = save_test_analysis(cfg, out, frame, snapshot, protocol, receipts)
    trained = CatBoostRanker(); trained.load_model(str(out / 'models/catboost.cbm'))
    pd.DataFrame({'feature': frozen['features'], 'importance': trained.get_feature_importance(
        type='PredictionValuesChange', thread_count=cfg.threads)}).sort_values('importance', ascending=False).to_csv(
            cfg.path('results_dir') / 'final_feature_importance.csv', index=False)
    result = {**run, 'split': 'test', 'cutoff': cfg.test_cutoff, 'test_evaluated': True,
        'frozen_validation_sha256': sha256(cfg.path('results_dir') / 'frozen_validation.json'),
        'final_protocol_sha256': sha256(protocol_path), 'parameters': frozen['parameters'],
        'reference_method': protocol['reference_method'], 'training_snapshots': snapshots,
        'test_snapshot': {**test_info, 'path': str(path.relative_to(cfg.root))},
        'models': {name: {**receipt, 'path': str((out / 'models' / f'{name}.cbm').relative_to(cfg.root))} for name, receipt in receipts.items()},
        'metrics': summaries, 'artifact_directory': str(out.relative_to(cfg.root)),
        'predictions_sha256': sha256(out / 'test_predictions.parquet'),
        'elapsed_seconds': time.perf_counter() - started,
        'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if platform.system() == 'Darwin' else 1024),
        'evaluation_note': 'Fixed validation-selected parameters; four historical refit cutoffs; no test-driven tuning or early stopping.'}
    result['result_files'] = {name: sha256(cfg.path('results_dir') / name) for name in
        ['test_results.csv', 'test_confidence_intervals.json', 'test_segments.csv', 'final_feature_importance.csv']}
    write_json(out / 'test_manifest.json', result)
    write_json(cfg.path('results_dir') / 'test_manifest.json', result)
    return result


def verify_files(cfg, run):
    protocol, frozen = load_protocol(cfg)
    if (run['split'] != 'test' or not run['test_evaluated'] or run['cutoff'] != cfg.test_cutoff
            or run['config_fingerprint'] != cfg.fingerprint()
            or run['final_protocol_sha256'] != sha256(cfg.path('results_dir') / 'final_protocol.json')
            or run['parameters'] != frozen['parameters']):
        raise ValueError('Final result identity mismatch')
    for model in run['models'].values():
        if sha256(cfg.root / model['path']) != model['model_sha256']:
            raise ValueError('Final model checksum mismatch')
    if sha256(cfg.root / run['test_snapshot']['path']) != run['test_snapshot']['sha256']:
        raise ValueError('Test snapshot checksum mismatch')
    for name, checksum in run['result_files'].items():
        if sha256(cfg.path('results_dir') / name) != checksum:
            raise ValueError('Final results checksum mismatch')
    if sha256(cfg.root / run['artifact_directory'] / 'test_predictions.parquet') != run['predictions_sha256']:
        raise ValueError('Test predictions checksum mismatch')


def verify_final(cfg):
    run = json.loads((cfg.path('results_dir') / 'test_manifest.json').read_text())
    verify_files(cfg, run)
    pa.set_cpu_count(cfg.threads); pa.set_io_thread_count(cfg.threads)
    tables = load_prepared(cfg)
    _, actual = context(cfg, tables)
    if actual['data_fingerprint'] != run['data_fingerprint']:
        raise ValueError('Prepared test data changed')
    snapshot = make_snapshot(tables[0], cfg.test_cutoff, cfg.horizon_days)
    frame = compact(pd.read_parquet(cfg.root / run['test_snapshot']['path']))
    recorded = pd.read_parquet(cfg.root / run['artifact_directory'] / 'test_predictions.parquet')
    for name, receipt in run['models'].items():
        predictions = predict_saved(frame, cfg, cfg.root / receipt['path'], receipt)
        np.testing.assert_allclose(predictions, recorded[name], rtol=0, atol=1e-10)
        scored = frame[['user_id', 'item_id']].assign(score=predictions)
        metrics, _ = evaluate_scores(scored, snapshot.targets, 'score')
        if any(abs(metrics[k] - run['metrics'][name][k]) > 1e-10 for k in metrics):
            raise ValueError('Reloaded final metrics changed')
    return {'verified': True, 'test_cutoff': cfg.test_cutoff, 'model_sha256': run['models']['catboost']['model_sha256'],
            'ablation_sha256': run['models']['catboost_without_personal_features']['model_sha256'],
            'metrics': run['metrics']['catboost'], 'tolerance': 1e-10}
