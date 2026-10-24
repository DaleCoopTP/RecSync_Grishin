"""Fixed validation diagnostics; cached candidates and published test stay immutable."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from kion.config import Config

VARIANTS = ('source_only', 'manual_only', 'full')
LABELS = {'source_only': 'Только сигналы источников', 'manual_only': 'Только ручные признаки',
          'full': 'Полная модель'}


def configure(cfg):
    if cfg.threads > 2:
        raise ValueError('This diagnostic is limited to at most two threads')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(cfg.threads)
    import pyarrow as pa
    pa.set_cpu_count(cfg.threads)
    pa.set_io_thread_count(cfg.threads)


def feature_groups(columns):
    if len(columns) != len(set(columns)) or any(
            not c.startswith(('cand_', 'u_', 'i_', 'pair_')) and c != 'rrf_score' for c in columns):
        raise ValueError('Only declared model features are allowed; service fields must be excluded')
    source = [c for c in columns if c.startswith('cand_') or c == 'rrf_score']
    manual = [c for c in columns if c not in source]
    if not source or not manual:
        raise ValueError('Both feature families must be nonempty')
    return {'source_only': source, 'manual_only': manual, 'full': list(columns)}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def peak_rss_mb():
    divisor = 1024 ** 2 if platform.system() == 'Darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def verify_plan(cfg, plan):
    from kion.io import sha256
    if plan['config_fingerprint'] != cfg.fingerprint():
        raise ValueError('Ablation configuration differs from its sealed protocol')
    if plan['feature_groups'] != feature_groups(plan['feature_groups']['full']):
        raise ValueError('Feature families do not partition the original full model')
    for name, checksum in plan['protected_files'].items():
        if sha256(cfg.root / name) != checksum:
            raise ValueError(f'Protected input/result changed: {name}')


def make_plan(cfg):
    from kion.final import load_protocol
    from kion.io import sha256, write_json
    from kion.protocol import prepared_dir
    destination = cfg.path('results_dir') / 'feature_ablation_protocol.json'
    if destination.exists():
        plan = read_json(destination)
        verify_plan(cfg, plan)
        return plan
    locked, frozen = load_protocol(cfg)
    run = read_json(cfg.path('results_dir') / 'validation_manifest.json')
    if run['model_sha256'] != frozen['model_sha256'] or run['config_fingerprint'] != cfg.fingerprint():
        raise ValueError('Validation run differs from the frozen reference')
    versions = {p: importlib.metadata.version(p) for p in frozen['versions']}
    if versions != frozen['versions'] or platform.python_version() != frozen['python']:
        raise ValueError('Use the original baseline runtime for feature ablation')
    snapshots = frozen['candidate_snapshots']
    train = [s for s in snapshots if s['training']]
    validation = [s for s in snapshots if not s['training']]
    if (set(s['cutoff'] for s in train) != set(cfg.train_cutoffs) or len(validation) != 1
            or validation[0]['cutoff'] != cfg.validation_cutoff):
        raise ValueError('Ablation accepts only the original three train cutoffs and validation')
    events = prepared_dir(cfg) / 'events.parquet'
    sample = read_json(prepared_dir(cfg) / 'manifest.json')
    paths = [s['path'] for s in snapshots] + [frozen['model_artifact'], str(events.relative_to(cfg.root)),
             str(Path(__file__).resolve().relative_to(cfg.root))]
    for name in ('frozen_validation.json', 'validation_manifest.json', 'validation_results.csv',
                 'final_protocol.json', 'test_manifest.json', 'test_results.csv',
                 'test_confidence_intervals.json', 'test_segments.csv', 'test_reload_verification.json'):
        path = cfg.path('results_dir') / name
        if path.exists():
            paths.append(str(path.relative_to(cfg.root)))
    protected = {p: sha256(cfg.root / p) for p in paths}
    protected.update(locked['invariant_sources'])
    for snapshot in snapshots:
        if protected[snapshot['path']] != snapshot['sha256']:
            raise ValueError('Original snapshot checksum mismatch')
    if protected[str(events.relative_to(cfg.root))] != sample['events_sha256']:
        raise ValueError('Prepared events checksum mismatch')
    plan = {'version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Post-hoc validation diagnostic after baseline completion; no test evaluation or tuning',
            'config_fingerprint': cfg.fingerprint(), 'parameters': frozen['parameters'],
            'feature_groups': feature_groups(frozen['features']),
            'categorical_features': frozen['categorical_features'],
            'training_snapshots': train, 'validation_snapshot': validation[0],
            'events_path': str(events.relative_to(cfg.root)),
            'full_model': {'path': frozen['model_artifact'], 'sha256': frozen['model_sha256'],
                           'metrics': run['selected']['metrics'],
                           'historical_fit_predict_evaluate_seconds': run['selected']['seconds'],
                           'original_trained_iterations': run['selected']['trained_iterations']},
            'protected_files': protected, 'versions': versions, 'python': platform.python_version(),
            'bootstrap_samples': cfg.bootstrap_samples, 'seed': cfg.seed,
            'inference_repetitions': 3, 'test_evaluated': False,
            'limitations': ['Same personalized candidate pool and cached preprocessing in all variants',
                           'Source signals are engineered too; families share popularity information',
                           'Fixed full-model parameters, no separate tuning or early stopping for subsets',
                           'Intervals on previously used validation are descriptive, not independent confirmation',
                           'Feature generation/retrieval costs excluded; no end-to-end savings claim']}
    plan['artifact_directory'] = str((cfg.path('artifact_dir') / 'feature_ablation' / digest(plan)[:16]).relative_to(cfg.root))
    verify_plan(cfg, plan)
    write_json(destination, plan)
    write_json(cfg.root / plan['artifact_directory'] / 'protocol.json', plan)
    return plan


def fit_variant(cfg, plan, variant):
    import pandas as pd
    from catboost import CatBoostRanker
    from kion.io import sha256, write_json
    from kion.pipeline import compact
    from kion.training import model_pool
    if variant not in ('source_only', 'manual_only'):
        raise ValueError('Only the two new subsets may be fitted')
    verify_plan(cfg, plan)
    out = cfg.root / plan['artifact_directory']
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / f'{variant}_fit.json'
    model_path = out / f'{variant}.cbm'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['protocol_sha256'] != digest(plan) or sha256(model_path) != receipt['model_sha256']:
            raise ValueError('Completed ablation checkpoint differs from protocol')
        print(f'{variant}: verified checkpoint, skipping fit', flush=True)
        return receipt
    columns = plan['feature_groups'][variant]
    categorical = [c for c in plan['categorical_features'] if c in columns]
    read_columns = columns + ['label', 'query_id', 'cutoff']
    train = compact(pd.concat([pd.read_parquet(cfg.root / s['path'], columns=read_columns)
                               for s in plan['training_snapshots']], ignore_index=True))
    if set(train.cutoff.unique()) != set(cfg.train_cutoffs) or len(train) != sum(s['rows'] for s in plan['training_snapshots']):
        raise ValueError('Training rows/cutoffs differ from the original validation experiment')
    if train.label.nunique() != 2:
        raise ValueError('Training requires both labels')
    queries = train.query_id.nunique()
    started = time.perf_counter()
    pool = model_pool(train, columns, categorical, cfg.threads)
    pool_seconds = time.perf_counter() - started
    model = CatBoostRanker(**plan['parameters'], eval_metric='NDCG:top=10',
                           allow_writing_files=False, used_ram_limit='6gb')
    print(f'Fitting {variant}: {len(columns)} features, {len(train):,} pairs, '
          f'{plan["parameters"]["iterations"]} fixed trees, {cfg.threads} threads', flush=True)
    started = time.perf_counter()
    model.fit(pool, verbose=50)  # No validation labels, eval_set or early stopping.
    fit_seconds = time.perf_counter() - started
    temporary = model_path.with_suffix('.tmp')
    model.save_model(str(temporary), format='cbm')
    temporary.replace(model_path)
    receipt = {'protocol_sha256': digest(plan), 'variant': variant, 'features': columns,
               'model_sha256': sha256(model_path), 'model_path': str(model_path.relative_to(cfg.root)),
               'training_rows': len(train), 'training_queries': int(queries), 'trees': model.tree_count_,
               'pool_seconds': pool_seconds, 'fit_seconds': fit_seconds, 'fit_process_peak_rss_mb': peak_rss_mb(),
               'eval_set_used': False, 'early_stopping': False, 'test_evaluated': False}
    write_json(receipt_path, receipt)
    verify_plan(cfg, plan)
    return receipt


def benchmark_variant(cfg, plan, variant):
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRanker, Pool
    from kion.io import sha256, write_json
    from kion.metrics import evaluate_scores
    from kion.pipeline import compact
    from kion.protocol import make_snapshot, day_number
    verify_plan(cfg, plan)
    out = cfg.root / plan['artifact_directory']
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / f'{variant}_benchmark.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['protocol_sha256'] != digest(plan):
            raise ValueError('Benchmark belongs to a different protocol')
        for name, checksum in receipt['files'].items():
            if sha256(cfg.root / name) != checksum:
                raise ValueError('Completed benchmark checksum mismatch')
        return receipt
    model_info = plan['full_model'] if variant == 'full' else read_json(out / f'{variant}_fit.json')
    model_path = cfg.root / (model_info['path'] if variant == 'full' else model_info['model_path'])
    expected_sha = model_info['sha256'] if variant == 'full' else model_info['model_sha256']
    if sha256(model_path) != expected_sha:
        raise ValueError('Ablation model checksum mismatch')
    columns = plan['feature_groups'][variant]
    categorical = [c for c in plan['categorical_features'] if c in columns]
    frame = compact(pd.read_parquet(cfg.root / plan['validation_snapshot']['path'],
                                   columns=columns + ['user_id', 'item_id', 'label', 'query_id', 'cutoff']))
    if set(frame.cutoff.unique()) != {cfg.validation_cutoff}:
        raise ValueError('Only the original validation snapshot may be scored')
    model = CatBoostRanker()
    model.load_model(str(model_path))
    if model.feature_names_ != columns or model.tree_count_ != plan['parameters']['iterations']:
        raise ValueError('Loaded model features/tree count mismatch')
    started = time.perf_counter()
    pool = Pool(frame[columns], cat_features=categorical, thread_count=cfg.threads)
    pool_seconds = time.perf_counter() - started
    scores = model.predict(pool, thread_count=cfg.threads)  # Warm-up; excluded from timing.
    timings = []
    for _ in range(plan['inference_repetitions']):
        started = time.perf_counter()
        repeated = model.predict(pool, thread_count=cfg.threads)
        timings.append(time.perf_counter() - started)
        np.testing.assert_allclose(scores, repeated, rtol=0, atol=1e-10)
    inference_rss = peak_rss_mb()  # Independent process; never trained another variant.
    events = pd.read_parquet(cfg.root / plan['events_path'], columns=['user_id', 'item_id', 'day'])
    # Do not pass held-out events to snapshot/target construction.
    events = events.loc[events.day < day_number(cfg.validation_cutoff) + cfg.horizon_days]
    targets = make_snapshot(events, cfg.validation_cutoff, cfg.horizon_days).targets
    scored = frame[['user_id', 'item_id']].assign(score=scores)
    metrics, per_user = evaluate_scores(scored, targets, 'score')
    if variant == 'full' and any(abs(metrics[k] - plan['full_model']['metrics'][k]) > 1e-10 for k in metrics):
        raise ValueError('Original full-model validation metrics changed')
    predictions_path = out / f'{variant}_predictions.parquet'
    per_user_path = out / f'{variant}_per_user.parquet'
    scored.to_parquet(predictions_path, index=False)
    per_user.to_parquet(per_user_path)
    files = {str(p.relative_to(cfg.root)): sha256(p) for p in [model_path, predictions_path, per_user_path]}
    receipt = {'protocol_sha256': digest(plan), 'variant': variant, 'metrics': metrics,
               'model_sha256': expected_sha, 'model_bytes': model_path.stat().st_size,
               'features': len(columns), 'validation_rows': len(frame), 'validation_pool_seconds': pool_seconds,
               'predict_seconds_repeats': timings, 'predict_seconds_median': float(np.median(timings)),
               'inference_process_peak_rss_mb': inference_rss, 'files': files,
               'loaded_model_verified': True, 'test_evaluated': False}
    write_json(receipt_path, receipt)
    print(f'Validation {variant}: {metrics}', flush=True)
    verify_plan(cfg, plan)
    return receipt


def summarize(cfg, plan):
    import numpy as np
    import pandas as pd
    from kion.io import sha256, write_json
    from kion.metrics import METRICS, bootstrap
    verify_plan(cfg, plan)
    out = cfg.root / plan['artifact_directory']
    tables, rows, receipts = {}, [], {}
    for variant in VARIANTS:
        receipt = read_json(out / f'{variant}_benchmark.json')
        if receipt['protocol_sha256'] != digest(plan):
            raise ValueError('Mixed benchmark protocols')
        for name, checksum in receipt['files'].items():
            if sha256(cfg.root / name) != checksum:
                raise ValueError('Result/model checksum mismatch')
        receipts[variant] = receipt
        tables[variant] = pd.read_parquet(out / f'{variant}_per_user.parquet')
        fitted = read_json(out / f'{variant}_fit.json') if variant != 'full' else None
        rows.append({'variant': variant, 'features': receipt['features'], **receipt['metrics'],
                     'fit_seconds': fitted['fit_seconds'] if fitted else None,
                     'fit_process_peak_rss_mb': fitted['fit_process_peak_rss_mb'] if fitted else None,
                     'predict_seconds_median': receipt['predict_seconds_median'],
                     'inference_process_peak_rss_mb': receipt['inference_process_peak_rss_mb'],
                     'model_mib': receipt['model_bytes'] / 1024 ** 2})
    reference = tables['full']
    for table in tables.values():
        if not table.index.equals(reference.index):
            raise ValueError('Variants must evaluate identical users')
        for field in ['target_count', 'candidate_count', 'candidate_recall@200']:
            np.testing.assert_array_equal(table[field], reference[field])
    absolute = {name: bootstrap(table, n=plan['bootstrap_samples'], seed=plan['seed'])
                for name, table in tables.items()}
    paired = {f'{a}_minus_{b}': bootstrap(tables[a], tables[b], n=plan['bootstrap_samples'], seed=plan['seed'])
              for a, b in [('full', 'source_only'), ('full', 'manual_only'), ('manual_only', 'source_only')]}
    frame = pd.DataFrame(rows)
    for metric in METRICS:
        frame[f'{metric}_lower'] = [absolute[v][metric]['lower'] for v in frame.variant]
        frame[f'{metric}_upper'] = [absolute[v][metric]['upper'] for v in frame.variant]
    results = cfg.path('results_dir')
    results.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results / 'feature_ablation_results.csv', index=False)
    write_json(results / 'feature_ablation_intervals.json', {'absolute': absolute, 'paired': paired,
               'samples': plan['bootstrap_samples'], 'seed': plan['seed'], 'unit': 'user',
               'interpretation': 'Descriptive post-hoc validation intervals; fixed models, no multiplicity correction'})
    manifest = {'protocol_sha256': digest(plan), 'artifact_directory': plan['artifact_directory'],
                'benchmarks': receipts, 'protected_files_verified_unchanged': True,
                'scope': plan['scope'], 'test_evaluated': False,
                'result_files': {p.name: sha256(p) for p in [results / 'feature_ablation_results.csv',
                                                            results / 'feature_ablation_intervals.json']}}
    write_json(results / 'feature_ablation_manifest.json', manifest)
    return frame


def run(cfg, config_path):
    configure(cfg)
    plan = make_plan(cfg)
    plan_path = cfg.root / plan['artifact_directory'] / 'protocol.json'
    for operation, variants in [('fit', VARIANTS[:2]), ('benchmark', VARIANTS)]:
        for variant in variants:
            # Separate sequential processes give each variant its own RSS high-water mark.
            subprocess.run([sys.executable, '-u', '-m', 'scripts.feature_ablation', '--config', str(config_path),
                            '--plan', str(plan_path), '--worker', operation, '--variant', variant],
                           cwd=cfg.root, env=os.environ.copy(), check=True)
    frame = summarize(cfg, plan)
    print(frame.to_string(index=False), flush=True)
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.toml')
    parser.add_argument('--worker', choices=['fit', 'benchmark'])
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--plan')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = Config.load(config_path)
    configure(cfg)
    if args.worker:
        if not args.variant or not args.plan:
            parser.error('Workers require --variant and --plan')
        plan = read_json(args.plan)
        (fit_variant if args.worker == 'fit' else benchmark_variant)(cfg, plan, args.variant)
    else:
        run(cfg, config_path)
