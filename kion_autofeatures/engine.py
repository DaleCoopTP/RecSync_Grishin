"""History-only group aggregates; candidate projection preserves pair order."""
from pathlib import Path
import time

import numpy as np
import pandas as pd

from kion.features import first_token, tokens, MISSING
from kion.io import sha256
from kion.protocol import day_number
from .baseline import assert_pairs
from .grammar import ITEM_CATEGORIES, USER_CATEGORIES, resolve_specs
from .io import checked_receipt, digest, source_hashes, write_json


def frame_digest(frame):
    import hashlib
    hasher = hashlib.sha256()
    hasher.update(str(list(frame.columns)).encode())
    hasher.update(str(list(frame.dtypes.astype(str))).encode())
    hasher.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return hasher.hexdigest()


class HistoryFeatures:
    def __init__(self, events, users, items, cutoff):
        self.day = day_number(cutoff)
        self.cutoff = cutoff
        columns = ["user_id", "item_id", "day", "total_dur", "watched_pct"]
        self.history = events.loc[events.day < self.day, columns].sort_values(
            ["user_id", "day", "item_id"]).reset_index(drop=True)
        if self.history.empty:
            raise ValueError("Automatic features require nonempty historical events")
        if self.history.duplicated(["user_id", "item_id"]).any():
            raise ValueError("Expected the KION unique last-watch user-item format")
        if (self.history.total_dur.dropna() < 0).any() or np.isinf(
                self.history[["total_dur", "watched_pct"]].to_numpy(dtype="float64")).any():
            raise ValueError("Historical numeric fields must be finite or missing; duration nonnegative")
        if users.user_id.duplicated().any() or items.item_id.duplicated().any():
            raise ValueError("Metadata keys must be unique")
        self.user_ids = pd.Index(sorted(self.history.user_id.unique()))
        self.item_ids = pd.Index(sorted(self.history.item_id.unique()))
        self.users = users.set_index("user_id").reindex(self.user_ids)[list(USER_CATEGORIES)].copy()
        self.items = items.set_index("item_id").reindex(self.item_ids)[list(ITEM_CATEGORIES)].copy()
        for col in USER_CATEGORIES:
            self.users[col] = self.users[col].astype("string").fillna(MISSING).astype(str)
        for col in ITEM_CATEGORIES:
            self.items[col] = self.items[col].map(tokens if col == "genres" else first_token)
        self.identity = {"cutoff": cutoff, "history": frame_digest(self.history),
                         "users": frame_digest(self.users.reset_index()),
                         "items": frame_digest(self.items.assign(genres=self.items.genres.map(tuple)).reset_index())}
        self._events = {}
        self._aggregates = {}

    def _path_events(self, path):
        if path in self._events:
            return self._events[path]
        root = path.split("_", 1)[0]
        frame = self.history.rename(columns={root + "_id": "root"}).copy()
        frame["weight"] = np.float32(1)
        if "_" in path:
            category = path.split("_", 1)[1]
            metadata, key = (self.items, "item_id") if root == "user" else (self.users, "user_id")
            frame["category"] = metadata[category].reindex(frame[key]).to_numpy()
            if category == "genres":
                frame["weight"] = frame.category.map(lambda x: 1 / len(x)).astype("float32")
                frame = frame.explode("category", ignore_index=True)
        frame["log_dur"] = np.log1p(frame.total_dur.to_numpy(dtype="float64"))
        self._events[path] = frame
        return frame

    def _aggregate(self, path, window):
        key = (path, window)
        if key in self._aggregates:
            return self._aggregates[key]
        frame = self._path_events(path)
        if window:
            frame = frame.loc[frame.day >= self.day - window]
        keys = ["root", "category"] if "_" in path else ["root"]
        values = frame[keys + ["day", "weight", "watched_pct"]].copy()
        values["dur_valid"] = np.isfinite(frame.log_dur).to_numpy() * frame.weight.to_numpy()
        values["dur_sum"] = frame.log_dur.fillna(0).to_numpy() * frame.weight.to_numpy()
        values["pct_valid"] = np.isfinite(frame.watched_pct).to_numpy() * frame.weight.to_numpy()
        values["pct_sum"] = frame.watched_pct.fillna(0).to_numpy() * frame.weight.to_numpy()
        result = values.groupby(keys, sort=True, observed=True).agg(
            count=("weight", "sum"), dur_valid=("dur_valid", "sum"), dur_sum=("dur_sum", "sum"),
            pct_valid=("pct_valid", "sum"), pct_sum=("pct_sum", "sum"),
            max_pct=("watched_pct", "max"), last_day=("day", "max"))
        self._aggregates[key] = result.astype("float32")
        return self._aggregates[key]

    def _project(self, pairs, path, window, column, maximum=False):
        aggregate = self._aggregate(path, window)
        root = path.split("_", 1)[0]
        if "_" not in path:
            return aggregate[column].reindex(pairs[root + "_id"]).to_numpy(dtype="float32")
        category = path.split("_", 1)[1]
        metadata, key = (self.items, "item_id") if root == "user" else (self.users, "user_id")
        values = metadata[category].reindex(pairs[key]).to_numpy()
        lookup = pd.DataFrame({"row": np.arange(len(pairs)), "root": pairs[root + "_id"].to_numpy(),
                               "category": values})
        lookup["weight"] = np.float32(1)
        if category == "genres":
            lookup["weight"] = lookup.category.map(lambda x: 1 / len(x)).astype("float32")
            lookup = lookup.explode("category", ignore_index=True)
        index = pd.MultiIndex.from_frame(lookup[["root", "category"]])
        matched = aggregate[column].reindex(index).to_numpy(dtype="float32")
        rows = lookup.row.to_numpy()
        if maximum:
            result = np.full(len(pairs), -np.inf, dtype="float32")
            valid = np.isfinite(matched)
            np.maximum.at(result, rows[valid], matched[valid])
            result[~np.isfinite(result)] = np.nan
            return result
        return np.bincount(rows, weights=np.nan_to_num(matched, nan=0) * lookup.weight.to_numpy(),
                           minlength=len(pairs)).astype("float32")

    def transform(self, pairs, specs):
        if set(pairs.columns) != {"user_id", "item_id"}:
            raise ValueError("Feature builder accepts pair IDs only, never labels or service columns")
        assert_pairs(pairs)
        if not pairs.user_id.isin(self.user_ids).all() or not pairs.item_id.isin(self.item_ids).all():
            raise ValueError("Pairs must refer to historical users and items")
        specs = resolve_specs([s.name for s in specs])
        output = {}
        # Local scalar projections are reused for derived shares/means/trends.
        projected = {}

        def project(path, window, column, maximum=False):
            key = (path, window, column, maximum)
            if key not in projected:
                projected[key] = self._project(pairs, path, window, column, maximum)
            return projected[key]

        for spec in specs:
            p, w, op = spec.path, spec.window, spec.operation
            if op in ("count", "sum_log_dur"):
                values = np.nan_to_num(project(p, w, "count" if op == "count" else "dur_sum"), nan=0)
            elif op in ("mean_log_dur", "mean_pct"):
                prefix = "dur" if op == "mean_log_dur" else "pct"
                numerator = project(p, w, prefix + "_sum")
                denominator = project(p, w, prefix + "_valid")
                values = np.divide(numerator, denominator, out=np.full(len(pairs), np.nan, dtype="float32"),
                                   where=denominator > 0)
            elif op == "max_pct":
                values = project(p, w, "max_pct", True)
            elif op == "recency":
                values = self.day - project(p, 0, "last_day", True)
            elif op == "share":
                count = np.nan_to_num(project(p, w, "count"), nan=0)
                total = np.nan_to_num(project(p.split("_", 1)[0], w, "count"), nan=0)
                values = np.divide(count, total, out=np.zeros(len(pairs), dtype="float32"), where=total > 0)
            else:
                small = np.nan_to_num(project(p, w, "count"), nan=0)
                large = np.nan_to_num(project(p, spec.long_window, "count"), nan=0)
                values = (small + 1) / (large + 1)
            output[spec.name] = np.asarray(values, dtype="float32")
        return pd.concat([pairs.reset_index(drop=True), pd.DataFrame(output, index=range(len(pairs)))], axis=1)


def cache_features(engine, pairs, specs, path, batch_rows=20000):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    receipt_path = path.with_suffix(".json")
    identity = {"history": engine.identity, "pairs": frame_digest(pairs),
                "features": [s.to_dict() for s in specs], "sources": source_hashes()}
    existing = checked_receipt(receipt_path, identity)
    if existing:
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    started = time.perf_counter()
    writer = None
    try:
        for start in range(0, max(len(pairs), 1), batch_rows):
            block = engine.transform(pairs.iloc[start:start + batch_rows], specs)
            table = pa.Table.from_pandas(block, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)
    receipt = {"identity": identity, "files": {path.name: sha256(path)}, "rows": len(pairs),
               "seconds": time.perf_counter() - started, "bytes": path.stat().st_size}
    write_json(receipt_path, receipt)
    return receipt
