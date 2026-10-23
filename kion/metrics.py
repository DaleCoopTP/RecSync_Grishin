import numpy as np
import pandas as pd

METRICS = ("ndcg@10", "recall@10", "map@10")
BASELINES = {"popularity_all": "cand_pop_score", "popularity_7d": "cand_recent_score", "itemknn": "cand_knn_score", "rrf": "rrf_score"}


def user_metrics(recommended, relevant, k=10):
    if not relevant:
        raise ValueError("Ranking metrics require at least one relevant item")
    if len(recommended) != len(set(recommended)):
        raise ValueError("Duplicate recommended items")
    hits = np.array([item in relevant for item in recommended[:k]], dtype="float64")
    ideal = np.sum(1.0 / np.log2(np.arange(2, min(k, len(relevant)) + 2)))
    dcg = np.sum(hits / np.log2(np.arange(2, len(hits) + 2)))
    ap = np.sum(hits * np.cumsum(hits) / np.arange(1, len(hits) + 1)) / min(k, len(relevant))
    return {f"ndcg@{k}": float(dcg / ideal), f"recall@{k}": float(hits.sum() / len(relevant)), f"map@{k}": float(ap)}


def evaluate_scores(frame, targets, score_column, k=10):
    if len(frame) and (frame.duplicated(["user_id", "item_id"]).any() or not np.isfinite(frame[score_column]).all()):
        raise ValueError("Evaluation requires unique pairs and finite scores")
    groups = {int(uid): g for uid, g in frame.groupby("user_id", sort=False)}
    rows = []
    for user, relevant in sorted(targets.items()):
        g = groups.get(user)
        if g is None:
            ranked, candidates = [], set()
        else:
            ids = g.item_id.to_numpy()
            order = np.lexsort((ids, -g[score_column].to_numpy()))
            ranked, candidates = ids[order][:k].tolist(), set(ids)
        rows.append({"user_id": user, **user_metrics(ranked, relevant, k),
                     "candidate_recall@200": len(candidates & relevant) / len(relevant),
                     "target_count": len(relevant), "candidate_count": len(candidates)})
    per_user = pd.DataFrame(rows).set_index("user_id")
    summary = {c: float(per_user[c].mean()) for c in [f"ndcg@{k}", f"recall@{k}", f"map@{k}", "candidate_recall@200"]}
    summary["users"] = len(per_user)
    return summary, per_user


def evaluate_baselines(frame, targets):
    summary, per_user = {}, {}
    for name, col in BASELINES.items():
        summary[name], per_user[name] = evaluate_scores(frame, targets, col)
    return summary, per_user


def bootstrap(per_user, reference=None, n=1000, seed=42):
    if reference is not None:
        if not per_user.index.equals(reference.index):
            raise ValueError("Paired bootstrap requires identical user order")
        values = per_user[list(METRICS)].to_numpy() - reference[list(METRICS)].to_numpy()
    else:
        values = per_user[list(METRICS)].to_numpy()
    if len(values) == 0:
        raise ValueError("No users to bootstrap")
    rng = np.random.default_rng(seed)
    draws = np.empty((n, len(METRICS)))
    for j in range(n):
        draws[j] = values[rng.integers(0, len(values), len(values))].mean(axis=0)
    return {metric: {"mean": float(values[:, j].mean()), "lower": float(np.quantile(draws[:, j], 0.025)),
                     "upper": float(np.quantile(draws[:, j], 0.975))} for j, metric in enumerate(METRICS)}
