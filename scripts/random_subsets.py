"""Post-test diagnostic: rank of the frozen selection among random same-size sets.

Only the cached inner search split is used (training 19/26 July, evaluation 2 August).
The frozen method, the test week and every earlier result stay unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

SEED_OFFSET = 1000  # a stream separate from the frozen control, default_rng(seed)
METRICS = ("ndcg@10", "recall@10", "map@10")


def draw_sets(eligible, size, count, seed):
    """The frozen control's rule: uniform over eligible features, without replacement."""
    import numpy as np
    pool = sorted(eligible)
    return [np.random.default_rng(seed + SEED_OFFSET + k).choice(pool, size, replace=False).tolist()
            for k in range(1, count + 1)]


def rank_summary(values, reference):
    """Rank 1 is best; ties count against the reference."""
    import numpy as np
    values = np.asarray(values, dtype="float64")
    at_least = int((values >= reference).sum())
    return {"rank": at_least + 1, "of": len(values) + 1, "draws_at_least": at_least,
            "empirical_p_one_sided": (at_least + 1) / (len(values) + 1)}


def worker(root, draws, extra_seeds):
    from kion_autofeatures.runtime import configure
    configure(2)
    import gc
    import numpy as np
    import pandas as pd
    from kion.io import sha256
    from kion.metrics import bootstrap
    from kion.protocol import day_number, load_prepared
    from kion_autofeatures.baseline import KEYS, Baseline, attach_features
    from kion_autofeatures.config import ResearchConfig
    from kion_autofeatures.engine import frame_digest
    from kion_autofeatures.experiment import load_selection
    from kion_autofeatures.io import digest, read_json, verify_files, write_json
    from kion_autofeatures.model import fit_evaluate
    from kion_autofeatures.search import choose_users, inner_validation

    started = time.perf_counter()
    research = ResearchConfig.load(root / "autofeatures.toml")
    baseline = Baseline(research)
    selection, identity = load_selection(research, baseline)
    search = read_json(research.results / "search.json")
    verify_files(research.results, search["files"])
    protected = {name: sha256(research.results / name) for name in
                 ("search.json", "selection.json", "frozen.json", "validation.json", "test.json")}
    cfg = baseline.cfg
    selected = selection["selected"]
    sets = {"frozen_random": selection["random"]}
    sets.update({f"random_{k:02d}": d for k, d in enumerate(
        draw_sets(selection["eligible"], len(selected), draws, cfg.seed), 1)})
    if len({tuple(sorted(v)) for v in [selected, *sets.values()]}) != len(sets) + 1:
        raise ValueError("Random draws must differ from each other and from both frozen sets")

    # Rebuild the inner split exactly as run_search did, from its verified caches.
    base_columns = baseline.families["source"] + baseline.families["raw"]
    cats = [c for c in baseline.frozen["categorical_features"] if c in base_columns]
    needed = ["user_id", "item_id"] + sorted(set(selected).union(*sets.values()))
    work = research.artifacts / "search"
    train_specs, _ = baseline.snapshots()
    frames = []
    for spec in train_specs[:2]:
        frame = baseline.read_snapshot(spec, KEYS + base_columns)
        chosen = choose_users(frame.user_id, research.inner_users, cfg.seed, spec["cutoff"])
        frame = frame.loc[frame.user_id.isin(chosen)].reset_index(drop=True)
        frames.append(attach_features(frame, pd.read_parquet(work / f'{spec["cutoff"]}.parquet', columns=needed)))
    train = pd.concat(frames, ignore_index=True)
    del frames
    events, users, items = load_prepared(cfg)
    events = events.loc[events.day < day_number(cfg.validation_cutoff)].copy()
    validation, targets, _ = inner_validation(events, users, items, baseline, research)
    validation = attach_features(validation[KEYS + base_columns],
                                 pd.read_parquet(work / "inner_validation.parquet", columns=needed))
    del events, users, items
    gc.collect()
    if set(train.cutoff) != set(cfg.train_cutoffs[:2]) or set(validation.cutoff) != {cfg.train_cutoffs[2]}:
        raise ValueError("Diagnostic must stay inside the inner search dates")

    reference_dir = work / f"prefix_{len(selected)}"
    reference = read_json(reference_dir / "fit.json")
    parameters = {**baseline.frozen["parameters"], "thread_count": research.threads}
    columns = base_columns + selected
    expected = reference["identity"]
    comparability = {
        "columns": expected["columns"] == columns, "categorical": expected["categorical"] == cats,
        "parameters": expected["parameters"] == parameters,
        "train": expected["train"] == frame_digest(train[KEYS + columns]),
        "evaluation": expected["evaluation"] == frame_digest(validation[["user_id", "item_id"] + columns]),
        "targets": expected["targets"] == digest({int(u): sorted(map(int, v)) for u, v in targets.items()})}
    if not all(comparability.values()):
        raise ValueError(f"Reconstructed inner split differs from the search: {comparability}")

    context = {"diagnostic": "random_subsets_inner", "inputs": identity["inputs"],
               "search": digest(search["identity"])}
    out = research.artifacts / "diagnostics" / "random_subsets"
    runs = [("selected_refit", selected, cfg.seed)]
    runs += [(name, features, cfg.seed) for name, features in sets.items()]
    runs += [(f"selected_seed_{s}", selected, s) for s in range(cfg.seed + 1, cfg.seed + 1 + extra_seeds)]
    records, directories = {"selected": reference}, {"selected": reference_dir}
    for number, (name, features, seed) in enumerate(runs, 1):
        print(f"[{number}/{len(runs)}] {name}: seed={seed}", flush=True)
        records[name] = fit_evaluate(train, validation, targets, base_columns + features, cats,
                                     {**parameters, "random_seed": seed}, out / name, context,
                                     research.prediction_repeats)
        directories[name] = out / name
        print(f"    ndcg@10={records[name]['metrics']['ndcg@10']:.6f} fit={records[name]['fit_seconds']:.1f}s", flush=True)
    del train, validation
    gc.collect()
    refit = pd.read_parquet(out / "selected_refit" / "predictions.parquet")
    original = pd.read_parquet(reference_dir / "predictions.parquet")
    np.testing.assert_allclose(refit.score, original.score, rtol=0, atol=1e-10)

    per_user = {name: pd.read_parquet(path / "per_user.parquet") for name, path in directories.items()}
    rows = []
    for name, record in records.items():
        if name == "selected_refit":
            continue
        kind = ("selected" if name == "selected" else "selected_seed" if name.startswith("selected_seed")
                else "frozen_random" if name == "frozen_random" else "random_draw")
        features = sets.get(name, selected)
        rows.append({"variant": name, "kind": kind, "seed": record["identity"]["parameters"]["random_seed"],
                     **{m: record["metrics"][m] for m in METRICS},
                     "overlap_with_selected": len(set(features) & set(selected)),
                     "fit_seconds": record["fit_seconds"]})
    results = pd.DataFrame(rows)
    random = results.loc[results.kind == "random_draw"]
    seeds = results.loc[results.kind.isin(["selected", "selected_seed"])]
    summary = {}
    for metric in METRICS:
        value = float(results.loc[results.variant == "selected", metric].iloc[0])
        frozen_value = float(results.loc[results.variant == "frozen_random", metric].iloc[0])
        summary[metric] = {
            "selected": value, "frozen_random": frozen_value,
            "random_mean": float(random[metric].mean()), "random_std": float(random[metric].std(ddof=1)),
            "random_min": float(random[metric].min()), "random_median": float(random[metric].median()),
            "random_max": float(random[metric].max()),
            "selected_among_draws": rank_summary(random[metric], value),
            "frozen_random_among_draws": rank_summary(random[metric], frozen_value),
            "selected_seed_mean": float(seeds[metric].mean()), "selected_seed_std": float(seeds[metric].std(ddof=1)),
            "selected_seed_min": float(seeds[metric].min()), "selected_seed_max": float(seeds[metric].max())}
    draw_names = random.variant.tolist()
    mean_random = sum(per_user[n][list(METRICS)] for n in draw_names) / len(draw_names)
    paired = {name: bootstrap(per_user["selected"], per_user[name], n=1000, seed=42)
              for name in ["frozen_random", *draw_names]}
    paired["mean_of_draws"] = bootstrap(per_user["selected"], mean_random, n=1000, seed=42)
    ndcg = {name: value["ndcg@10"] for name, value in paired.items() if name != "mean_of_draws"}
    excluded = {"selected_better": sum(v["lower"] > 0 for n, v in ndcg.items() if n in draw_names),
                "random_better": sum(v["upper"] < 0 for n, v in ndcg.items() if n in draw_names)}
    if {name: sha256(research.results / name) for name in protected} != protected:
        raise ValueError("A frozen result changed during the diagnostic")
    results.to_csv(research.results / "random_subsets.csv", index=False)
    record = {
        "scope": "Post-test diagnostic on the inner search split; the frozen method and test results are unchanged",
        "protocol": {"inner_train_cutoffs": cfg.train_cutoffs[:2], "inner_evaluation_cutoff": cfg.train_cutoffs[2],
                     "inner_users_per_cutoff": research.inner_users, "size": len(selected),
                     "eligible": len(selection["eligible"]), "draws": draws,
                     "draw_rule": "numpy default_rng(seed + 1000 + k).choice(sorted(eligible), size, replace=False), k = 1..draws",
                     "frozen_random_rule": "default_rng(seed).choice(sorted(eligible), size, replace=False)",
                     "parameters": parameters, "extra_seeds_for_selected": extra_seeds,
                     "bootstrap": {"samples": 1000, "seed": 42, "unit": "user", "method": "percentile"},
                     "selected_source": f"search prefix_{len(selected)} fit, reproduced by selected_refit"},
        "comparability": comparability, "selected_refit_identical": True,
        "sets": {"selected": selected, **sets}, "results": rows, "summary": summary,
        "paired_selected_minus": paired, "paired_ndcg_intervals_excluding_zero": excluded,
        "test_evaluated": False, "frozen_results_unchanged": True,
        "script_sha256": sha256(Path(__file__)), "worker_seconds": time.perf_counter() - started}
    write_json(out / "completed.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--extra-seeds", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.draws < 2 or args.extra_seeds < 1:
        parser.error("Use at least two draws and one extra seed")
    root = Path(__file__).resolve().parents[1]
    if args.worker:
        worker(root, args.draws, args.extra_seeds)
        return
    from kion_autofeatures.config import ResearchConfig
    from kion_autofeatures.runtime import execution_lock, monitor_process
    from kion_autofeatures.io import read_json, write_json
    research = ResearchConfig.load(root / "autofeatures.toml")
    directory = research.artifacts / "diagnostics" / "random_subsets"
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = str(research.threads)
    environment["PYTHONUNBUFFERED"] = "1"
    print(f"Random-subset diagnostic: {args.draws} draws, {research.threads} threads, "
          f"{research.memory_limit_gib} GiB RSS limit; log={directory / 'random_subsets.log'}", flush=True)
    with execution_lock(root):
        resources = monitor_process([sys.executable, "-m", "scripts.random_subsets", "--draws", str(args.draws),
                                     "--extra-seeds", str(args.extra_seeds), "--worker"],
                                    directory / "random_subsets.log",
                                    int(research.memory_limit_gib * 1024**3), root, environment)
    if resources["returncode"] or resources["memory_limit_exceeded"]:
        raise RuntimeError(f"Diagnostic stopped; inspect {directory / 'random_subsets.log'}")
    record = read_json(directory / "completed.json")
    record["resources"] = resources
    write_json(research.results / "random_subsets.json", record)
    print(json.dumps({"status": "complete", "minutes": resources["seconds"] / 60,
                      "peak_tree_rss_mb": resources["peak_tree_rss_mb"],
                      "ndcg@10": record["summary"]["ndcg@10"], "test_evaluated": False},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
