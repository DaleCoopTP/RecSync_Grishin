"""Verify saved predictions without rewriting the recorded training timings."""
import argparse
import json
import os
from kion.config import Config


def verify(cfg):
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = str(cfg.threads)
    from catboost import CatBoostRanker, Pool
    import pandas as pd
    import pyarrow as pa
    pa.set_cpu_count(cfg.threads)
    pa.set_io_thread_count(cfg.threads)
    from kion.io import sha256
    from kion.metrics import evaluate_scores
    from kion.pipeline import code_fingerprint, compact
    from kion.protocol import load_prepared, make_snapshot
    run = json.loads((cfg.path('results_dir') / 'validation_manifest.json').read_text())
    if (run['test_evaluated'] or run['cutoff'] != cfg.validation_cutoff or run['split'] != 'validation'
            or run['code_fingerprint'] != code_fingerprint() or run['config_fingerprint'] != cfg.fingerprint()):
        raise ValueError('Validation source or protocol mismatch')
    path = cfg.root / run['model_artifact']
    if sha256(path) != run['model_sha256']:
        raise ValueError('Model checksum mismatch')
    manifest = json.loads((path.parent / 'model_manifest.json').read_text())
    cache = path.parent.parent / f'{cfg.validation_cutoff}_validation.parquet'
    cache_info = next(s for s in run['snapshots'] if not s['training'])
    if sha256(cache) != cache_info['sha256']:
        raise ValueError('Validation cache checksum mismatch')
    frame = compact(pd.read_parquet(cache))
    events, _, _ = load_prepared(cfg)
    targets = make_snapshot(events, cfg.validation_cutoff, cfg.horizon_days).targets
    model = CatBoostRanker(); model.load_model(str(path))
    scored = frame[['user_id', 'item_id']].copy()
    scored['score'] = model.predict(Pool(frame[manifest['features']], cat_features=manifest['categorical_features'],
                                         thread_count=cfg.threads), thread_count=cfg.threads)
    metrics, _ = evaluate_scores(scored, targets, 'score')
    if any(abs(metrics[key] - run['selected']['metrics'][key]) > 1e-10 for key in metrics):
        raise ValueError('Reloaded predictions do not reproduce recorded validation metrics')
    return {'model_sha256': sha256(path), 'reloaded_metrics': metrics, 'test_evaluated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.toml')
    print(json.dumps(verify(Config.load(parser.parse_args().config)), indent=2))
