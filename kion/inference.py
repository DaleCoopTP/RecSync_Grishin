"""Serve an explicitly saved validation or final checkpoint."""
import json
import numpy as np
from catboost import CatBoostRanker, Pool
from .candidates import CandidateRetriever
from .features import FeatureStore
from .io import sha256
from .protocol import load_prepared, make_snapshot


def recommend_saved(cfg, user_id, k=10, stage='auto'):
    if not 1 <= k <= 10 or stage not in {'auto', 'validation', 'test'}:
        raise ValueError('Supported length is 1..10; stage is auto, validation or test')
    final_path = cfg.path('results_dir') / 'test_manifest.json'
    if stage == 'auto':
        stage = 'test' if final_path.exists() else 'validation'
    if stage == 'test':
        from .final import load_protocol, verify_files
        load_protocol(cfg)
        run = json.loads(final_path.read_text()); verify_files(cfg, run)
        receipt = run['models']['catboost']
        path = cfg.root / receipt['path']; checksum = receipt['model_sha256']
        columns = receipt['identity']['features']; cats = receipt['identity']['categorical_features']
        cutoff = run['cutoff']
    else:
        frozen = json.loads((cfg.path('results_dir') / 'frozen_validation.json').read_text())
        if cfg.fingerprint() != frozen['config_fingerprint']:
            raise ValueError('Validation configuration mismatch')
        path = cfg.root / frozen['model_artifact']; checksum = frozen['model_sha256']
        columns = frozen['features']; cats = frozen['categorical_features']; cutoff = frozen['validation_cutoff']
    if sha256(path) != checksum:
        raise ValueError('Recommendation model checksum mismatch')
    events, users, items = load_prepared(cfg)
    snapshot = make_snapshot(events, cutoff, cfg.horizon_days)
    store = FeatureStore(snapshot, users, items, cfg)
    result = {'user_id': user_id, 'as_of': cutoff, 'stage': stage}
    if user_id not in store.user_index:
        order = np.lexsort((store.item_ids, -store.item_features.i_count_all.to_numpy()))[:k]
        return {**result, 'method': 'popularity_fallback', 'item_ids': store.item_ids[order].tolist()}
    pairs = CandidateRetriever(store, cfg).generate([user_id])
    if pairs.empty:
        return {**result, 'method': 'catboost', 'item_ids': []}
    frame = store.transform(pairs)
    model = CatBoostRanker(); model.load_model(str(path))
    scores = model.predict(Pool(frame[columns], cat_features=cats, thread_count=cfg.threads), thread_count=cfg.threads)
    order = np.lexsort((frame.item_id.to_numpy(), -scores))[:k]
    return {**result, 'method': 'catboost', 'item_ids': frame.item_id.to_numpy()[order].tolist(), 'scores': scores[order].tolist()}
