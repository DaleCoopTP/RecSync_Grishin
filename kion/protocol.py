from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from .data import audit_archive, SOURCE_COMMIT, ARCHIVE_SHA256
from .io import write_json, sha256


def day_number(value):
    return int((pd.Timestamp(value) - pd.Timestamp("1970-01-01")).days)


def sample_users(events, count, seed):
    ids = np.sort(events.user_id.unique())
    if count > len(ids):
        raise ValueError(f"Requested {count} users but only {len(ids)} are present")
    chosen = np.random.default_rng(seed).choice(ids, size=count, replace=False)
    sample = events.loc[events.user_id.isin(chosen)].sort_values(["user_id", "day", "item_id"]).reset_index(drop=True)
    return sample, chosen


@dataclass
class Snapshot:
    cutoff: str
    history: pd.DataFrame
    targets: dict[int, set[int]]
    active_users: np.ndarray
    catalog: np.ndarray
    diagnostics: dict


def make_snapshot(events, cutoff, horizon=7):
    day = day_number(cutoff)
    history = events.loc[events.day < day].copy()
    future = events.loc[(events.day >= day) & (events.day < day + horizon)]
    warm = set(history.user_id.unique())
    targets = {int(u): set(map(int, g.item_id)) for u, g in future.loc[future.user_id.isin(warm)].groupby("user_id", sort=True)}
    catalog = np.sort(history.item_id.unique())
    return Snapshot(cutoff, history, targets, np.array(sorted(targets), dtype=np.int32), catalog, {
        "cutoff": cutoff, "history_rows": len(history), "history_users": len(warm), "history_items": len(catalog),
        "target_rows": len(future), "warm_active_users": len(targets),
        "cold_active_users": future.loc[~future.user_id.isin(warm)].user_id.nunique(),
        "warm_user_new_item_targets": int((~future.loc[future.user_id.isin(warm)].item_id.isin(catalog)).sum()),
    })


def prepared_dir(cfg):
    return cfg.path("data_dir") / f"sample_{cfg.sample_users}_{cfg.seed}"


def prepare(cfg, local=None):
    events, users, items, audit = audit_archive(cfg, local)
    events, selected = sample_users(events, cfg.sample_users, cfg.seed)
    out = prepared_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "user_ids.npy", selected, allow_pickle=False)
    events.to_parquet(out / "events.parquet", index=False)
    users.loc[users.user_id.isin(selected)].to_parquet(out / "users.parquet", index=False)
    items.to_parquet(out / "items.parquet", index=False)
    manifest = {
        "source_commit": SOURCE_COMMIT, "archive_sha256": ARCHIVE_SHA256, "seed": cfg.seed,
        "sample_users": cfg.sample_users, "sample_rows": len(events), "user_ids_sha256": sha256(out / "user_ids.npy"),
        "events_sha256": sha256(out / "events.parquet"), "numpy_version": np.__version__,
        "snapshots": [make_snapshot(events, s, cfg.horizon_days).diagnostics for s in cfg.train_cutoffs + [cfg.validation_cutoff]],
        "test_cutoff": cfg.test_cutoff, "test_evaluated": False,
        "sampling_note": "Uniform cohort over all released user IDs; eligibility is determined separately at each cutoff.",
        "assumptions": ["last_watch_dt is a last recorded interaction, not a complete event stream",
                        "Metadata is treated as static and available at cutoff; no historical versions exist"],
    }
    write_json(out / "manifest.json", manifest)
    write_json(cfg.path("results_dir") / "sample_manifest.json", manifest)
    return manifest


def load_prepared(cfg):
    out = prepared_dir(cfg)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["archive_sha256"] != ARCHIVE_SHA256 or manifest["sample_users"] != cfg.sample_users or manifest["seed"] != cfg.seed:
        raise ValueError("Prepared data provenance does not match configuration")
    if sha256(out / "events.parquet") != manifest["events_sha256"] or sha256(out / "user_ids.npy") != manifest["user_ids_sha256"]:
        raise ValueError("Prepared data checksum mismatch; run prepare again")
    return tuple(pd.read_parquet(out / f"{name}.parquet") for name in ("events", "users", "items"))
