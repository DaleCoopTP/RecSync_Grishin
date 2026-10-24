"""Exercise real adapters and every research stage on a synthetic baseline."""
from dataclasses import replace
import json
import sys

import numpy as np
import pytest

from kion.config import Config
from kion.data import ARCHIVE_SHA256
from kion.io import sha256, write_json
from kion.pipeline import run_validation
from kion.protocol import prepared_dir
from kion.final import lock_test, run_final as baseline_final
from scripts.freeze_validation import freeze as freeze_baseline
from scripts.feature_ablation import make_plan, fit_variant, benchmark_variant, summarize
from test_final import synthetic_tables
from kion_autofeatures.config import ResearchConfig
from kion_autofeatures.experiment import run_validation as research_validation
from kion_autofeatures.final import freeze, run_final, verify, verify_frozen
from kion_autofeatures.runtime import execution_lock, monitor_process
from kion_autofeatures.search import run_search


@pytest.fixture
def research_baseline(tmp_path, monkeypatch):
    cfg = replace(Config(root=tmp_path), sample_users=20, train_users_per_cutoff=20,
                  threads=1, depths=[2], l2_values=[3], iterations=4,
                  early_stopping=2, batch_users=20, bootstrap_samples=20)
    tables = synthetic_tables(cfg)
    data = prepared_dir(cfg)
    data.mkdir(parents=True)
    for name, frame in zip(("events", "users", "items"), tables):
        frame.to_parquet(data / f"{name}.parquet", index=False)
    np.save(data / "user_ids.npy", np.arange(20), allow_pickle=False)
    manifest = {"sample_users": 20, "seed": 42, "sample_rows": len(tables[0]),
                "archive_sha256": ARCHIVE_SHA256, "user_ids_sha256": sha256(data / "user_ids.npy"),
                **{f"{name}_sha256": sha256(data / f"{name}.parquet") for name in ("events", "users", "items")}}
    write_json(data / "manifest.json", manifest)
    write_json(cfg.path("results_dir") / "sample_manifest.json", manifest)
    (tmp_path / "config.toml").write_text("\n".join(f"{key} = {json.dumps(value)}" for key, value in cfg.to_dict().items()))
    run_validation(cfg, tables)
    freeze_baseline(cfg)
    lock_test(cfg)
    baseline_final(cfg, tables)
    import scripts.feature_ablation as ablation_module
    from pathlib import Path
    local_script = tmp_path / "scripts" / "feature_ablation.py"
    local_script.parent.mkdir()
    local_script.write_bytes(Path(ablation_module.__file__).read_bytes())
    monkeypatch.setattr(ablation_module, "__file__", str(local_script))
    plan = make_plan(cfg)
    for variant in ("source_only", "manual_only"):
        fit_variant(cfg, plan, variant)
    for variant in ("source_only", "manual_only", "full"):
        benchmark_variant(cfg, plan, variant)
    summarize(cfg, plan)
    protocol = tmp_path / "docs" / "automatic_features_protocol.md"
    protocol.parent.mkdir()
    protocol.write_text("Synthetic protocol fixture; production uses the committed protocol.\n")
    research = ResearchConfig(root=tmp_path, threads=1, inner_users=8, screen_iterations=3,
                              screen_depth=2, max_features=4, prefix_sizes=(2, 4),
                              batch_rows=2000, prediction_repeats=1)
    return research, cfg


def test_complete_synthetic_lifecycle_freeze_refit_order_reload_and_resume(research_baseline, monkeypatch):
    import kion_autofeatures.final as final_module
    from catboost import CatBoostRanker
    research, cfg = research_baseline
    with pytest.raises(ValueError, match="Freeze"):
        run_final(research)
    search = run_search(research)
    assert 0 < search["chosen_size"] <= 4
    validation = research_validation(research)
    assert len(validation["models"]) == 7
    frozen = freeze(research)
    assert verify_frozen(research)["seal"] == frozen["seal"]
    with pytest.raises(ValueError, match="closed"):
        run_search(research)
    calls = []
    original_fit = CatBoostRanker.fit
    original_build = final_module.build_datasets

    def checked_fit(self, *args, **kwargs):
        assert "eval_set" not in kwargs and "early_stopping_rounds" not in kwargs
        calls.append(True)
        return original_fit(self, *args, **kwargs)

    def checked_build(*args, **kwargs):
        if args[3:] == ("test", "evaluation"):
            assert len(calls) == 4
        return original_build(*args, **kwargs)

    monkeypatch.setattr(CatBoostRanker, "fit", checked_fit)
    monkeypatch.setattr(final_module, "build_datasets", checked_build)
    result = run_final(research)
    assert len(calls) == 4
    assert set(result["models"]) == {"source", "source_raw", "auto", "random", "full"}
    assert result["test_evaluated"]
    assert result["snapshot_diagnostics"]["cold_active_users"] == 1
    assert all(not result["models"][n]["test_labels_used"] for n in ("source", "source_raw", "auto", "random"))
    assert run_final(research) == result
    assert len(calls) == 4
    checked = verify(research)
    assert checked["reload_verified"] and checked["test_evaluated"]
    assert checked["verified"] == ["inputs", "search", "validation", "frozen", "test"]
    with pytest.raises(ValueError, match="configuration"):
        verify_frozen(replace(research, prediction_repeats=2))
    path = research.results / "selection.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        run_final(research)


def test_memory_watchdog_and_exclusive_execution(tmp_path):
    command = [sys.executable, "-c", "import time; block=bytearray(40*1024*1024); time.sleep(10)"]
    result = monitor_process(command, tmp_path / "memory.log", 20 * 1024**2)
    assert result["memory_limit_exceeded"] and result["returncode"] != 0
    assert result["seconds"] < 8
    with execution_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with execution_lock(tmp_path):
                pass

