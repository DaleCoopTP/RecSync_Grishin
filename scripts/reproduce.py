"""Explicitly repeat the full experiment in fresh output directories."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
from kion.config import Config


def reproduce(cfg, name):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
        raise ValueError('Use letters, digits, underscore or hyphen for the run name')
    cfg = replace(cfg, artifact_dir=f'artifacts/reproductions/{name}', results_dir=f'results/reproductions/{name}')
    results = cfg.path('results_dir')
    if results.exists():
        raise FileExistsError('Choose a new name; published/reproduced results are never overwritten')
    results.mkdir(parents=True)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = str(cfg.threads)
    lines = [f'{key} = {json.dumps(value)}' for key, value in cfg.to_dict().items()]
    # Keep relative data/artifact paths rooted at the project, just like the primary config.
    config_path = cfg.root / f'reproduction_{name}.toml'
    config_path.write_text('\n'.join(lines) + '\n')
    from kion.protocol import prepare
    from kion.pipeline import run_validation
    from kion.final import lock_test, run_final, verify_final
    from kion.io import write_json
    from scripts.freeze_validation import freeze
    from scripts.verify_validation import verify
    prepare(cfg)
    run_validation(cfg)
    write_json(results / 'reload_verification.json', verify(cfg))
    freeze(cfg)
    lock_test(cfg)
    run_final(cfg)
    write_json(results / 'test_reload_verification.json', verify_final(cfg))
    return {'results': str(results), 'config': str(config_path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.toml'); parser.add_argument('--name', required=True)
    parser.add_argument('--full', action='store_true', help='Explicitly run the expensive full validation and refit')
    args = parser.parse_args()
    if not args.full:
        parser.error('Full reproduction requires --full; use main.py verify-test for a cheap check')
    print(json.dumps(reproduce(Config.load(args.config), args.name), indent=2))
