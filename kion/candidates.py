"""Label-independent candidate retrieval over a historical snapshot."""
import numpy as np
import pandas as pd
from scipy import sparse

SOURCES = ("pop", "recent", "knn", "genre")


def top_indices(indices, scores, item_ids, limit):
    indices = np.asarray(indices, dtype=np.int32)
    if len(indices) > limit:
        threshold = np.partition(scores[indices], -limit)[-limit]
        indices = indices[scores[indices] >= threshold]
    return indices[np.lexsort((item_ids[indices], -scores[indices]))][:limit]


class CandidateRetriever:
    def __init__(self, store, cfg):
        self.store, self.cfg = store, cfg
        h = store.history.groupby("user_id", sort=False).tail(cfg.knn_history_limit)
        u = store.user_index.get_indexer(h.user_id)
        i = store.item_index.get_indexer(h.item_id)
        self.ui = sparse.csr_matrix((np.ones(len(h), dtype="float32"), (u, i)), shape=store.shape)
        counts = np.asarray(self.ui.sum(axis=0)).ravel()
        rr, cc, vv = [], [], []
        for start in range(0, store.shape[1], 128):
            stop = min(start + 128, store.shape[1])
            co = (self.ui[:, start:stop].T @ self.ui).tocsr()
            for local in range(stop - start):
                row = start + local
                a, b = co.indptr[local:local + 2]
                cols, values = co.indices[a:b], co.data[a:b]
                valid = cols != row
                cols, values = cols[valid], values[valid]
                sim = values / np.sqrt(counts[row] * counts[cols]) * values / (values + cfg.knn_shrinkage)
                if len(cols) > cfg.knn_neighbors:
                    threshold = np.partition(sim, -cfg.knn_neighbors)[-cfg.knn_neighbors]
                    keep = sim >= threshold
                    cols, sim = cols[keep], sim[keep]
                order = np.lexsort((store.item_ids[cols], -sim))[:cfg.knn_neighbors]
                rr.append(np.full(len(order), row, dtype=np.int32)); cc.append(cols[order]); vv.append(sim[order])
        self.similarity = sparse.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(store.shape[1], store.shape[1]), dtype="float32")
        self.pop = store.item_features.i_count_all.to_numpy(dtype="float32")
        self.recent = store.item_features.i_count_7.to_numpy(dtype="float32")
        self.pop_order = np.lexsort((store.item_ids, -self.pop))
        self.recent_order = np.lexsort((store.item_ids, -self.recent))
        self.genre_pop = np.log1p(self.recent)

    def generate(self, user_ids):
        ids = np.asarray(user_ids, dtype=np.int32)
        positions = self.store.user_index.get_indexer(ids)
        if (positions < 0).any():
            raise ValueError("Candidate generation batch requires users with history")
        profiles = self.store.user_genre["all"][positions] / np.maximum(self.store.user_counts["all"][positions, None], 1)
        genre_scores = (profiles @ self.store.genre_dense.T) * self.genre_pop
        knn = (self.ui[positions] @ self.similarity).tocsr()
        rows = []
        for local, (user, pos) in enumerate(zip(ids, positions)):
            seen = self.store.ui.indices[self.store.ui.indptr[pos]:self.store.ui.indptr[pos + 1]]
            available = np.ones(self.store.shape[1], dtype=bool); available[seen] = False
            ks = np.zeros(self.store.shape[1], dtype="float32")
            ks[knn.indices[knn.indptr[local]:knn.indptr[local + 1]]] = knn.data[knn.indptr[local]:knn.indptr[local + 1]]
            score_arrays = [self.pop, self.recent, ks, genre_scores[local]]
            lists = [self.pop_order[available[self.pop_order]][:self.cfg.source_limit],
                     self.recent_order[available[self.recent_order]][:self.cfg.source_limit],
                     top_indices(np.flatnonzero(available & (ks > 0)), ks, self.store.item_ids, self.cfg.source_limit),
                     top_indices(np.flatnonzero(available & (genre_scores[local] > 0)), genre_scores[local], self.store.item_ids, self.cfg.source_limit)]
            rank_maps = [{int(item): r + 1 for r, item in enumerate(order)} for order in lists]
            union = set().union(*(set(order) for order in lists))
            rrf = {item: sum(1.0 / (self.cfg.rrf_constant + ranks[item]) for ranks in rank_maps if item in ranks) for item in union}
            preserved = set().union(*(set(order[:self.cfg.preserve_per_source]) for order in lists))
            rest = sorted(union - preserved, key=lambda item: (-rrf[item], int(self.store.item_ids[item])))
            selected = list(preserved) + rest[:max(0, self.cfg.candidate_limit - len(preserved))]
            for item in sorted(selected, key=lambda j: int(self.store.item_ids[j])):
                values = [int(user), int(self.store.item_ids[item]), rrf[item]]
                for scores, ranks in zip(score_arrays, rank_maps):
                    values.extend([float(scores[item]), float(ranks.get(item, self.cfg.source_limit + 1)), float(item in ranks)])
                rows.append(values)
        columns = ["user_id", "item_id", "rrf_score"]
        for source in SOURCES:
            columns.extend([f"cand_{source}_score", f"cand_{source}_rank", f"cand_{source}_present"])
        frame = pd.DataFrame(rows, columns=columns)
        for col in columns:
            frame[col] = frame[col].astype("int32" if col in ["user_id", "item_id"] else "float32")
        return frame

    def popular(self, user_id, k=10):
        pos = self.store.user_index.get_indexer([user_id])[0]
        seen = set() if pos < 0 else set(self.store.ui.indices[self.store.ui.indptr[pos]:self.store.ui.indptr[pos + 1]])
        return [int(self.store.item_ids[i]) for i in self.pop_order if i not in seen][:k]
