from pathlib import Path
import shutil
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from .io import sha256, write_json

SOURCE_COMMIT = "f69775be31fa5779907cf0a92ddedb70037fb5ae"
ARCHIVE_SHA256 = "50fc5b7db78cbda4c3d0ee5bb5bea4c416cff41e8af58aab0b661d0d3e9dfa9b"
SOURCE_URL = f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{SOURCE_COMMIT}/data_original.zip"


def ensure_archive(cfg, local=None):
    destination = cfg.path("data_dir") / "data_original.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(local) if local else destination
    if source.exists():
        if sha256(source) != ARCHIVE_SHA256:
            raise ValueError("KION archive SHA-256 mismatch; use the pinned full data_original.zip")
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    else:
        if local:
            raise FileNotFoundError(source)
        temporary = destination.with_suffix(".download")
        try:
            request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "kion-baseline/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            if sha256(temporary) != ARCHIVE_SHA256:
                raise ValueError("Downloaded KION archive SHA-256 mismatch")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def validate_tables(events, users, items, expected_rows=None):
    requirements = [(events, {"user_id", "item_id", "last_watch_dt", "total_dur", "watched_pct"}),
                    (users, {"user_id", "age", "income", "sex", "kids_flg"}),
                    (items, {"item_id", "content_type", "genres", "countries", "directors", "release_year", "for_kids", "age_rating"})]
    for table, required in requirements:
        if required - set(table):
            raise ValueError(f"Missing required columns: {sorted(required - set(table))}")
    if expected_rows is not None and len(events) != expected_rows:
        raise ValueError(f"Expected {expected_rows} interactions, got {len(events)}")
    if events[["user_id", "item_id", "last_watch_dt"]].isna().any().any():
        raise ValueError("Missing event identifiers or dates")
    events["last_watch_dt"] = pd.to_datetime(events.last_watch_dt, format="%Y-%m-%d", errors="raise")
    if events.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Duplicate user-item pairs")
    if users.user_id.isna().any() or users.user_id.duplicated().any() or items.item_id.isna().any() or items.item_id.duplicated().any():
        raise ValueError("Metadata IDs must be present and unique")
    if events.total_dur.isna().any() or (events.total_dur < 0).any():
        raise ValueError("Invalid duration")
    if ((events.watched_pct.dropna() < 0) | (events.watched_pct.dropna() > 100)).any():
        raise ValueError("Invalid watched_pct")
    return {
        "rows": len(events), "users": events.user_id.nunique(), "items": events.item_id.nunique(),
        "min_date": str(events.last_watch_dt.min().date()), "max_date": str(events.last_watch_dt.max().date()),
        "missing_event_values": events.isna().sum().to_dict(),
        "users_without_metadata": len(set(events.user_id) - set(users.user_id)),
        "items_without_metadata": len(set(events.item_id) - set(items.item_id)),
        "missing_user_values": users.isna().sum().to_dict(), "missing_item_values": items.isna().sum().to_dict(),
    }


def load_archive(path, expected_rows=5476251):
    with zipfile.ZipFile(path) as z:
        events = pd.read_csv(z.open("data_original/interactions.csv"), dtype={
            "user_id": "int32", "item_id": "int32", "total_dur": "float32", "watched_pct": "float32"})
        users = pd.read_csv(z.open("data_original/users.csv"))
        items = pd.read_csv(z.open("data_original/items.csv"), usecols=[
            "item_id", "content_type", "genres", "countries", "directors", "release_year", "for_kids", "age_rating"])
    audit = validate_tables(events, users, items, expected_rows)
    events["day"] = (events.last_watch_dt - pd.Timestamp("1970-01-01")).dt.days.astype("int32")
    return events, users, items, audit


def audit_archive(cfg, local=None):
    path = ensure_archive(cfg, local)
    events, users, items, audit = load_archive(path)
    audit.update(source_commit=SOURCE_COMMIT, archive_sha256=ARCHIVE_SHA256)
    write_json(cfg.path("results_dir") / "data_audit.json", audit)
    return events, users, items, audit
