"""Real-data pilot with historical pseudo-test dates, isolated from the main study."""
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def prepare_workspace(root, name):
    import numpy as np
    from kion.config import Config
    from kion.io import sha256
    from kion.protocol import load_prepared, prepared_dir
    from kion_autofeatures.io import read_json, verify_files, write_json

    original = Config.load(root / "config.toml")
    events, users, items = load_prepared(original)
    chosen = np.sort(np.random.default_rng(original.seed).choice(
        np.sort(events.user_id.unique()), 1000, replace=False))
    selected = events.loc[events.user_id.isin(chosen)].copy()
    workspace = root / "artifacts" / "autofeatures" / name / "workspace"
    cfg = replace(original, root=workspace, sample_users=1000, train_users_per_cutoff=200,
                  train_cutoffs=["2021-07-12", "2021-07-19", "2021-07-26"],
                  validation_cutoff="2021-08-02", test_cutoff="2021-08-09",
                  depths=[4], l2_values=[3], iterations=30, early_stopping=5,
                  bootstrap_samples=100, batch_users=128)
    cfg.validate()
    parent = read_json(prepared_dir(original) / "manifest.json")
    identity = {"parent_events_sha256": parent["events_sha256"],
                "parent_user_ids_sha256": parent["user_ids_sha256"],
                "pilot_config": cfg.to_dict(), "users": chosen.tolist()}
    setup = workspace / "setup.json"
    if setup.exists():
        record = read_json(setup)
        if record["identity"] != identity:
            raise ValueError("Pilot workspace belongs to a different sample or configuration")
        verify_files(workspace, record["files"])
        return cfg, (selected, users.loc[users.user_id.isin(chosen)].copy(), items), record
    directory = prepared_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    tables = (selected, users.loc[users.user_id.isin(chosen)].copy(), items)
    for key, frame in zip(("events", "users", "items"), tables):
        frame.to_parquet(directory / f"{key}.parquet", index=False)
    np.save(directory / "user_ids.npy", chosen, allow_pickle=False)
    manifest = {**parent, "sample_users": 1000, "sample_rows": len(selected),
                "user_ids_sha256": sha256(directory / "user_ids.npy"),
                **{f"{key}_sha256": sha256(directory / f"{key}.parquet")
                   for key in ("events", "users", "items")},
                "sampling_note": "Uniform 1000 users from the frozen uniform 100000-user cohort; all rows retained",
                "parent_events_sha256": parent["events_sha256"],
                "snapshots": [], "test_cutoff": cfg.test_cutoff, "test_evaluated": False}
    write_json(directory / "manifest.json", manifest)
    write_json(cfg.path("results_dir") / "sample_manifest.json", manifest)
    (workspace / "config.toml").write_text("\n".join(
        f"{key} = {json.dumps(value)}" for key, value in cfg.to_dict().items()) + "\n")
    protocol = workspace / "docs" / "automatic_features_protocol.md"
    protocol.parent.mkdir()
    protocol.write_bytes((root / "docs" / "automatic_features_protocol.md").read_bytes())
    local_script = workspace / "scripts" / "feature_ablation.py"
    local_script.parent.mkdir()
    local_script.write_bytes((root / "scripts" / "feature_ablation.py").read_bytes())
    files = {str(p.relative_to(workspace)): sha256(p) for p in [
        *directory.iterdir(), workspace / "config.toml", protocol, local_script] if p.is_file()}
    record = {"identity": identity, "files": files, "history_preserved": True,
              "sample_users": len(chosen), "sample_rows": len(selected)}
    write_json(setup, record)
    return cfg, tables, record


def worker(root, name):
    from kion_autofeatures.runtime import configure
    configure(2)
    from kion.io import sha256
    from kion.pipeline import run_validation
    from kion.final import lock_test, run_final as baseline_final
    from kion_autofeatures.baseline import Baseline
    from kion_autofeatures.config import ResearchConfig
    from kion_autofeatures.experiment import run_validation as research_validation
    from kion_autofeatures.final import freeze, run_final, verify
    from kion_autofeatures.io import read_json, source_hashes, verify_files, write_json
    from kion_autofeatures.search import run_search
    from scripts.freeze_validation import freeze as baseline_freeze

    started = time.perf_counter()
    main_baseline = Baseline(ResearchConfig.load(root / "autofeatures.toml"))
    protected = main_baseline.protected_files()
    cfg, tables, setup = prepare_workspace(root, name)
    assert cfg.test_cutoff < main_baseline.cfg.test_cutoff
    assert __import__("pandas").Timestamp(cfg.test_cutoff) + __import__("pandas").Timedelta(days=7) <= __import__("pandas").Timestamp(main_baseline.cfg.test_cutoff)
    stages = {}
    def stage(key, function):
        print(f"Pilot stage: {key}", flush=True)
        before = time.perf_counter()
        value = function()
        stages[key] = time.perf_counter() - before
        return value

    if not (cfg.path("results_dir") / "validation_manifest.json").exists():
        stage("baseline_validation", lambda: run_validation(cfg, tables))
    if not (cfg.path("results_dir") / "frozen_validation.json").exists():
        stage("baseline_freeze", lambda: baseline_freeze(cfg))
    lock_test(cfg)
    stage("baseline_historical_final", lambda: baseline_final(cfg, tables))
    local = cfg.root / "scripts" / "feature_ablation.py"
    spec = importlib.util.spec_from_file_location("_pilot_feature_ablation", local)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.make_plan(cfg)
    if not (cfg.path("results_dir") / "feature_ablation_manifest.json").exists():
        for variant in ("source_only", "manual_only"):
            stage("baseline_" + variant, lambda v=variant: module.fit_variant(cfg, plan, v))
        for variant in ("source_only", "manual_only", "full"):
            stage("baseline_benchmark_" + variant, lambda v=variant: module.benchmark_variant(cfg, plan, v))
        module.summarize(cfg, plan)
    research = ResearchConfig(root=cfg.root, run_name="pilot", threads=2, inner_users=64,
                              screen_iterations=20, screen_depth=4, max_features=16,
                              prefix_sizes=(8, 16), batch_rows=20000)
    if not (research.results / "frozen.json").exists():
        stage("search", lambda: run_search(research))
    stage("validation", lambda: research_validation(research))
    stage("freeze", lambda: freeze(research))
    final = stage("historical_final", lambda: run_final(research))
    checked = stage("verify", lambda: verify(research))
    model_hashes = {str(p.relative_to(cfg.root)): sha256(p)
                    for p in (research.artifacts / "test").rglob("*.cbm")}
    repeated = stage("resume_final", lambda: run_final(research))
    assert repeated == final
    verify_files(cfg.root, model_hashes)
    verify_files(root, protected)
    selection = read_json(research.results / "selection.json")
    record = {"scope": "Real-data execution/resource pilot; quality is not used to change main study",
              "workspace": str(cfg.root.relative_to(root)), "sample_users": setup["sample_users"],
              "sample_rows": setup["sample_rows"], "history_preserved": setup["history_preserved"],
              "cohort_sha256": sha256(__import__("kion.protocol", fromlist=["prepared_dir"]).prepared_dir(cfg) / "user_ids.npy"),
              "cutoffs": {"train": cfg.train_cutoffs, "validation": cfg.validation_cutoff,
                          "historical_pseudo_test": cfg.test_cutoff},
              "research": research.to_dict(), "selected_features": len(selection["selected"]),
              "eligible_features": len(selection["eligible"]), "stages_seconds": stages,
              "total_worker_seconds": time.perf_counter() - started,
              "verification": checked, "resume_models_unchanged": True,
              "main_baseline_protected_files": protected, "main_baseline_unchanged": True,
              "main_test_evaluated": False, "research_sources": source_hashes(),
              "pilot_script_sha256": sha256(Path(__file__)),
              "main_budget_note": "Main model/grid unchanged. Pilot uses fewer users, shallower screen and shorter baseline fits; timings are not a linear forecast.",
              "main_dense_search_arrays_gib": (400000 + 200000) * 504 * 4 / 1024**3}
    write_json(cfg.root / "completed.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="pilot_23")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from kion_autofeatures.config import ResearchConfig
    ResearchConfig(run_name=args.name)
    if args.worker:
        worker(root, args.name)
        return
    from kion_autofeatures.runtime import execution_lock, monitor_process
    from kion_autofeatures.io import read_json, write_json
    directory = root / "artifacts" / "autofeatures" / args.name
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "2"
    environment["PYTHONUNBUFFERED"] = "1"
    print(f"Real-data pilot {args.name}: 1000 users, 2 threads, 6 GiB RSS limit", flush=True)
    with execution_lock(root):
        resources = monitor_process([sys.executable, "-m", "scripts.autofeatures_pilot",
                                     "--name", args.name, "--worker"],
                                    directory / "pilot.log", 6 * 1024**3, root, environment)
    write_json(directory / "resources.json", resources)
    if resources["returncode"] or resources["memory_limit_exceeded"]:
        raise RuntimeError(f"Pilot stopped; inspect {directory / 'pilot.log'}")
    record = read_json(directory / "workspace" / "completed.json")
    record["resources"] = resources
    write_json(root / "results" / "autofeatures" / f"{args.name}.json", record)
    print(json.dumps({"status": "complete", "seconds": resources["seconds"],
                      "peak_tree_rss_mb": resources["peak_tree_rss_mb"],
                      "main_test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()

