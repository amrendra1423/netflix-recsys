"""
Evaluation: rating-prediction accuracy (RMSE, MAE) and Top-K ranking quality
(MAP@K, Precision@K, Recall@K, NDCG@K, Hit-Rate, Coverage).

Methodology (documented here and in the README)
-----------------------------------------------
* Train/test split: per-user random hold-out of TEST_FRACTION of each user's
  ratings (see data_processing.per_user_split).
* Relevance: a held-out (test) movie is *relevant* to a user iff the user's
  actual rating is >= RELEVANCE_THRESHOLD (3.5 stars), per the problem statement.
* Top-K generation: for each evaluated user we score every catalogue item,
  remove items the user already rated in train, and take the K highest-scoring
  items.
* MAP@K: average precision at K, where a recommended item is a "hit" if it is
  one of the user's relevant test items; AP@K = (1/min(R,K)) * sum_{n=1..K}
  Precision@n * rel_n, and MAP@K is the mean of AP@K over evaluated users.
  Recommended items that are not in the user's test set are treated as
  non-relevant (standard offline assumption).
"""
from __future__ import annotations

import time

import numpy as np

import config


# --------------------------------------------------------------------------- #
# Rating accuracy
# --------------------------------------------------------------------------- #
def rmse(model, u, i, r) -> float:
    pred = model.predict_pairs(u, i)
    return float(np.sqrt(np.mean((pred - r) ** 2)))


def mae(model, u, i, r) -> float:
    pred = model.predict_pairs(u, i)
    return float(np.mean(np.abs(pred - r)))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def group_by_user(users, items, n_users):
    """Return a list `g` with g[u] = array of item indices for user u."""
    order = np.argsort(users, kind="stable")
    su, si = users[order], items[order]
    counts = np.bincount(su, minlength=n_users)
    splits = np.cumsum(counts)[:-1]
    return np.split(si, splits)


# --------------------------------------------------------------------------- #
# Ranking evaluation
# --------------------------------------------------------------------------- #
def ranking_eval(model, ds, k: int = config.TOP_K,
                 n_eval_users: int = config.N_EVAL_USERS,
                 relevance_threshold: float = config.RELEVANCE_THRESHOLD,
                 seed: int = config.RANDOM_SEED,
                 batch_users: int = 1000):
    """Compute MAP@K, Precision@K, Recall@K, NDCG@K, Hit-Rate and Coverage."""
    n_users, n_items = ds.n_users, ds.n_items
    train_by_user = group_by_user(ds.train_u, ds.train_i, n_users)
    rel = ds.test_r >= relevance_threshold
    rel_by_user = group_by_user(ds.test_u[rel], ds.test_i[rel], n_users)

    eligible = np.array([u for u in range(n_users)
                         if len(rel_by_user[u]) > 0 and len(train_by_user[u]) > 0])
    rng = np.random.default_rng(seed)
    if len(eligible) > n_eval_users:
        eval_users = rng.choice(eligible, size=n_eval_users, replace=False)
    else:
        eval_users = eligible

    idcg = np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))   # idcg[r-1] for r relevant
    ap_sum = prec_sum = rec_sum = ndcg_sum = hit_sum = 0.0
    recommended_items = set()

    for s in range(0, len(eval_users), batch_users):
        batch = eval_users[s:s + batch_users]
        scores = np.asarray(model.score_users(batch), dtype=np.float64)
        # mask train-seen items
        for row, u in enumerate(batch):
            scores[row, train_by_user[u]] = -np.inf
        # top-k (unsorted) then order by score desc
        topk_part = np.argpartition(scores, -k, axis=1)[:, -k:]
        part_scores = np.take_along_axis(scores, topk_part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        topk = np.take_along_axis(topk_part, order, axis=1)     # (m, k) item indices

        for row, u in enumerate(batch):
            recs = topk[row]
            relevant = rel_by_user[u]
            rel_set = set(relevant.tolist())
            hits = np.array([it in rel_set for it in recs], dtype=np.float64)
            n_hits = int(hits.sum())
            R = len(relevant)

            recommended_items.update(recs.tolist())
            # precision / recall / hit-rate
            prec_sum += n_hits / k
            rec_sum += n_hits / R
            hit_sum += 1.0 if n_hits > 0 else 0.0
            # average precision @ k
            if n_hits > 0:
                ranks = np.arange(1, k + 1)
                cum_hits = np.cumsum(hits)
                precision_at = cum_hits / ranks
                ap = np.sum(precision_at * hits) / min(R, k)
                ap_sum += ap
                # ndcg
                dcg = np.sum(hits / np.log2(ranks + 1))
                ndcg_sum += dcg / idcg[min(R, k) - 1]

    m = len(eval_users)
    return {
        f"MAP@{k}": ap_sum / m,
        f"Precision@{k}": prec_sum / m,
        f"Recall@{k}": rec_sum / m,
        f"NDCG@{k}": ndcg_sum / m,
        f"HitRate@{k}": hit_sum / m,
        "Coverage": len(recommended_items) / n_items,
        "n_eval_users": m,
    }


def evaluate_model(model, ds, k: int = config.TOP_K, verbose: bool = True):
    """Full metric suite for one model."""
    res = {"model": model.name}
    t0 = time.time()
    res["RMSE"] = rmse(model, ds.test_u, ds.test_i, ds.test_r)
    res["MAE"] = mae(model, ds.test_u, ds.test_i, ds.test_r)
    res.update(ranking_eval(model, ds, k=k))
    res["eval_seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"  {model.name}: RMSE={res['RMSE']:.4f} MAE={res['MAE']:.4f} "
              f"MAP@{k}={res[f'MAP@{k}']:.4f} P@{k}={res[f'Precision@{k}']:.4f} "
              f"NDCG@{k}={res[f'NDCG@{k}']:.4f} Cov={res['Coverage']:.3f}")
    return res
