"""Snapshot-only statistics. No labels are accepted by the feature builder."""
import numpy as np
import pandas as pd
from scipy import sparse
from .protocol import day_number

MISSING = "__MISSING__"


def tokens(value):
    if pd.isna(value):
        return [MISSING]
    return sorted({x.strip().lower() for x in str(value).split(",") if x.strip()}) or [MISSING]


def first_token(value):
    if pd.isna(value):
        return MISSING
    return next((x.strip().lower() for x in str(value).split(",") if x.strip()), MISSING)


def _means(index, values, size):
    valid = np.isfinite(values)
    count = np.bincount(index[valid], minlength=size)
    total = np.bincount(index[valid], weights=values[valid], minlength=size)
    return np.divide(total, count, out=np.zeros(size), where=count > 0).astype("float32"), (count == 0).astype("float32")


class FeatureStore:
    def __init__(self, snapshot, users, items, cfg):
        self.cutoff = snapshot.cutoff
        self.day = day_number(self.cutoff)
        self.history = snapshot.history.sort_values(["user_id", "day", "item_id"]).reset_index(drop=True)
        if len(self.history) == 0 or self.history.day.max() >= self.day:
            raise ValueError("Feature history must be nonempty and strictly before cutoff")
        self.user_ids = np.sort(self.history.user_id.unique())
        self.item_ids = np.sort(self.history.item_id.unique())
        self.user_index, self.item_index = pd.Index(self.user_ids), pd.Index(self.item_ids)
        self.ur = self.user_index.get_indexer(self.history.user_id)
        self.ir = self.item_index.get_indexer(self.history.item_id)
        self.shape = (len(self.user_ids), len(self.item_ids))
        self.days = self.history.day.to_numpy()
        self.ui = sparse.csr_matrix((np.ones(len(self.history), dtype="float32"), (self.ur, self.ir)), shape=self.shape)
        um = users.set_index("user_id").reindex(self.user_ids)
        im = items.set_index("item_id").reindex(self.item_ids)
        self.item_genres = [tokens(x) for x in im.genres]
        self.genres = sorted({g for gs in self.item_genres for g in gs})
        gidx = {g: n for n, g in enumerate(self.genres)}
        rows, cols, vals = [], [], []
        for n, gs in enumerate(self.item_genres):
            rows.extend([n] * len(gs)); cols.extend(gidx[g] for g in gs); vals.extend([1 / len(gs)] * len(gs))
        self.genre_matrix = sparse.csr_matrix((np.array(vals, dtype="float32"), (rows, cols)), shape=(self.shape[1], len(self.genres)))
        self.genre_dense = self.genre_matrix.toarray()
        self.user_features = pd.DataFrame(index=self.user_ids)
        self.item_features = pd.DataFrame(index=self.item_ids)
        for col in ["age", "income", "sex", "kids_flg"]:
            self.user_features[f"u_{col}"] = um[col].astype("string").fillna(MISSING).to_numpy(dtype=str)
        for col in ["content_type", "countries", "directors"]:
            self.item_features[f"i_{col}"] = im[col].map(first_token).to_numpy(dtype=str)
        self.item_features["i_genres"] = ["|".join(gs) for gs in self.item_genres]
        self.item_features["i_id"] = self.item_ids.astype(str)
        for col in ["release_year", "for_kids", "age_rating"]:
            self.item_features[f"i_{col}"] = im[col].fillna(0).to_numpy(dtype="float32")
            self.item_features[f"i_{col}_missing"] = im[col].isna().to_numpy(dtype="float32")
        self.user_genre = {}
        self.user_counts = {}
        self.window_ui = {}
        for window in [0] + list(cfg.windows):
            mask = np.ones(len(self.history), dtype=bool) if window == 0 else self.days >= self.day - window
            suffix = "all" if window == 0 else str(window)
            ui = self.ui if window == 0 else sparse.csr_matrix((np.ones(mask.sum(), dtype="float32"), (self.ur[mask], self.ir[mask])), shape=self.shape)
            self.window_ui[suffix] = ui
            uc = np.asarray(ui.sum(axis=1)).ravel().astype("float32")
            ic = np.asarray(ui.sum(axis=0)).ravel().astype("float32")
            self.user_counts[suffix] = uc
            self.user_features[f"u_count_{suffix}"] = uc
            self.item_features[f"i_count_{suffix}"] = ic
            self.user_genre[suffix] = (ui @ self.genre_matrix).toarray()
        for prefix, index, size, frame in [("u", self.ur, self.shape[0], self.user_features), ("i", self.ir, self.shape[1], self.item_features)]:
            latest = np.full(size, -1, dtype="int32")
            earliest = np.full(size, np.iinfo(np.int32).max, dtype="int32")
            np.maximum.at(latest, index, self.days)
            np.minimum.at(earliest, index, self.days)
            frame[f"{prefix}_recency"] = (self.day - latest).astype("float32")
            frame[f"{prefix}_observed_age"] = (self.day - earliest).astype("float32")
            for col in ["total_dur", "watched_pct"]:
                values = self.history[col].to_numpy(dtype="float32")
                if col == "total_dur":
                    values = np.log1p(values)
                mean, missing = _means(index, values, size)
                frame[f"{prefix}_{col}_mean"] = mean
                frame[f"{prefix}_{col}_missing"] = missing
            frame[f"{prefix}_trend_7_30"] = (frame[f"{prefix}_count_7"] + 1) / (frame[f"{prefix}_count_30"] + 1)
        self.user_features["u_active_days"] = self.history.groupby("user_id").day.nunique().reindex(self.user_ids).to_numpy(dtype="float32")
        self.user_features["u_genre_diversity"] = (self.user_genre["all"] > 0).sum(axis=1).astype("float32")
        genre_events = self.genre_matrix[self.ir].tocoo()
        self.last_genre = np.zeros((self.shape[0], len(self.genres)), dtype="int32")
        np.maximum.at(self.last_genre, (self.ur[genre_events.row], genre_events.col), self.days[genre_events.row])
        self.relations = {}
        for name in ["directors", "content_type"]:
            codes, _ = pd.factorize(self.item_features[f"i_{name}"], sort=True)
            relation = sparse.csr_matrix((np.ones(self.shape[1], dtype="float32"), (np.arange(self.shape[1]), codes)))
            counts = {suffix: (ui @ relation).tocsr() for suffix, ui in self.window_ui.items()}
            recent = pd.DataFrame({"u": self.ur, "c": codes[self.ir], "day": self.days}).groupby(["u", "c"], sort=False).day.max().reset_index()
            last = sparse.csr_matrix((recent.day.to_numpy(dtype="float32"), (recent.u, recent.c)), shape=(self.shape[0], relation.shape[1]))
            self.relations[name] = (codes, counts, last)

    def positions(self, pairs):
        u = self.user_index.get_indexer(pairs.user_id)
        i = self.item_index.get_indexer(pairs.item_id)
        if (u < 0).any() or (i < 0).any():
            raise ValueError("Feature pairs must refer to historical users and items")
        return u, i

    def base_features(self, pairs):
        u, i = self.positions(pairs)
        return pd.concat([pairs.reset_index(drop=True), self.user_features.iloc[u].reset_index(drop=True), self.item_features.iloc[i].reset_index(drop=True)], axis=1)

    def transform(self, pairs):
        frame = self.base_features(pairs)
        u, i = self.positions(pairs)
        genres = self.genre_dense[i]
        extra = {}
        for suffix, counts in self.user_genre.items():
            count = np.einsum("ij,ij->i", counts[u], genres)
            extra[f"pair_genre_count_{suffix}"] = count
            extra[f"pair_genre_share_{suffix}"] = count / np.maximum(self.user_counts[suffix][u], 1)
        last = np.max(self.last_genre[u] * (genres > 0), axis=1)
        extra["pair_genre_recency"] = np.where(last > 0, self.day - last, 999)
        extra["pair_genre_missing"] = last == 0
        for name, (codes, matrices, last_matrix) in self.relations.items():
            code = codes[i]
            for suffix, mat in matrices.items():
                count = np.asarray(mat[u, code]).ravel()
                extra[f"pair_{name}_count_{suffix}"] = count
                extra[f"pair_{name}_share_{suffix}"] = count / np.maximum(self.user_counts[suffix][u], 1)
            last = np.asarray(last_matrix[u, code]).ravel()
            extra[f"pair_{name}_recency"] = np.where(last > 0, self.day - last, 999)
            extra[f"pair_{name}_missing"] = last == 0
        return pd.concat([frame, pd.DataFrame({k: np.asarray(v, dtype="float32") for k, v in extra.items()})], axis=1)
