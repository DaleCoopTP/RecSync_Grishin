from dataclasses import replace
import json
import pandas as pd
from kion.config import Config
from kion.pipeline import run_validation


def test_validation_pipeline_and_cached_re_evaluation(tmp_path):
    cfg = replace(Config(root=tmp_path), sample_users=20, train_users_per_cutoff=20, threads=1,
                  depths=[2], l2_values=[3], iterations=4, early_stopping=2, batch_users=7)
    rows = []
    dates = [f"2021-05-{j + 1:02d}" for j in range(10)] + cfg.train_cutoffs + [cfg.validation_cutoff, cfg.test_cutoff]
    for u in range(20):
        for j, date in enumerate(dates):
            rows.append((u, (u + j) % 30, date, 600.0, 80.0))
    e = pd.DataFrame(rows, columns=["user_id", "item_id", "last_watch_dt", "total_dur", "watched_pct"])
    e["last_watch_dt"] = pd.to_datetime(e.last_watch_dt)
    e["day"] = (e.last_watch_dt - pd.Timestamp("1970-01-01")).dt.days.astype("int32")
    users = pd.DataFrame({"user_id": range(20), "age": ["young"] * 20, "income": ["low"] * 20, "sex": ["F"] * 20, "kids_flg": [0] * 20})
    items = pd.DataFrame({"item_id": range(30), "genres": ["drama, comedy"] * 30, "countries": ["RU"] * 30,
        "directors": [str(i % 3) for i in range(30)], "content_type": ["film"] * 30, "release_year": [2020] * 30,
        "for_kids": [0] * 30, "age_rating": [16] * 30})
    result = run_validation(cfg, (e, users, items))
    again = run_validation(cfg, (e, users, items), fit=False)
    assert result["selected"] == again["selected"]
    assert result["test_evaluated"] is False
    assert len(result["snapshots"]) == 4
    assert cfg.test_cutoff not in [s["cutoff"] for s in result["snapshots"]]
    assert (cfg.path("results_dir") / "validation_results.csv").exists()


def test_freeze_accepts_validation_only_and_rejects_changed_config(tmp_path):
    import pytest
    from kion.io import write_json
    from scripts.freeze_validation import freeze
    # Reuse the end-to-end synthetic experiment, then read its actual configuration.
    test_validation_pipeline_and_cached_re_evaluation(tmp_path)
    run = json.loads((tmp_path / 'results/validation_manifest.json').read_text())
    cfg = Config(**run['config'], root=tmp_path)
    write_json(tmp_path / 'results/sample_manifest.json',
               {'sample_users': cfg.sample_users, 'seed': cfg.seed, 'sample_rows': 300, 'user_ids_sha256': 'synthetic'})
    frozen = freeze(cfg)
    assert frozen['test_evaluated'] is False
    assert frozen['test_cutoff_reserved'] == cfg.test_cutoff
    assert (tmp_path / 'results/grid_results.csv').exists()
    with pytest.raises(ValueError, match='Configuration'):
        freeze(replace(cfg, iterations=cfg.iterations + 1))
