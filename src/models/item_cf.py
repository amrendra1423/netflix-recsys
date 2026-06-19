"""
Item-based collaborative filtering (centered-cosine / Pearson similarity).

Two movies are similar if users who rated both tended to rate them the same way
relative to each movie's average. We:
  1. center ratings by item mean:  x_ui = r_ui - mean_i
  2. compute item-item cosine similarity on centered vectors (= Pearson over
     co-rating users), with shrinkage (trust higher-overlap pairs) and a minimum
     co-rating support floor;
  3. keep each item's top-K neighbours.

Two scoring modes:
  * predict_pairs (rating prediction / RMSE):
        r_hat_ui = mean_i + sum_j sim(i,j)(r_uj - mean_j) / sum_j |sim(i,j)|
  * score_users (Top-N ranking): the weighted-similarity aggregate
        s_ui = sum_{j rated by u} sim(i,j)(r_uj - mean_j)
    which is the standard item-based recommendation score. (Ranking by the
    predicted rating instead is dominated by niche high-mean items and ranks
    poorly; the aggregate score ranks far better — see the report.)

Heavy steps are dense matrix products. The centered user-item matrix is dense
(n_users x n_items); for the default subset (~40k x 2k) this is ~320 MB.
"""
from __future__ import annotations

import numpy as np

import config


class ItemCFModel:
    name = "Item-based CF"

    def __init__(self, k=config.ITEMCF_K, shrinkage=config.ITEMCF_SHRINKAGE,
                 min_support=config.ITEMCF_MIN_SUPPORT, verbose=True):
        self.k = k
        self.shrinkage = shrinkage
        self.min_support = min_support
        self.verbose = verbose

    def fit(self, u, i, r, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items
        r = r.astype(np.float32)
        self.global_mean = float(r.mean())

        counts = np.bincount(i, minlength=n_items)
        sums = np.bincount(i, weights=r, minlength=n_items)
        self.item_mean = np.where(counts > 0, sums / np.maximum(counts, 1),
                                  self.global_mean).astype(np.float32)

        self._build_profiles(u, i, r)

        G = self.X.T @ self.X                            # similarity numerator
        ind = (self.X != 0).astype(np.float32)
        C = ind.T @ ind                                  # co-rating counts
        norm = np.sqrt(np.maximum(np.diag(G), 1e-8)).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            S = G / np.outer(norm, norm)
        S = np.nan_to_num(S, nan=0.0)
        S *= C / (C + self.shrinkage)
        S[C < self.min_support] = 0.0
        np.fill_diagonal(S, 0.0)

        K = min(self.k, n_items - 1)
        part = np.argpartition(np.abs(S), -K, axis=0)[-K:]
        mask = np.zeros_like(S, dtype=bool)
        cols = np.broadcast_to(np.arange(n_items), (K, n_items))
        mask[part, cols] = True
        self.S = np.where(mask, S, 0.0).astype(np.float32)
        self.absS = np.abs(self.S)
        if self.verbose:
            print(f"  [ItemCF] similarity built: {n_items} items, top-{K} "
                  f"neighbours, {int((self.S != 0).sum()):,} nonzero sims")
        return self

    def _build_profiles(self, u, i, r):
        """Dense centered user-item matrix (0 = not rated)."""
        self.X = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        self.X[u, i] = (r - self.item_mean[i]).astype(np.float32)

    def attach_profiles(self, u, i, r, n_users):
        """Rebuild user profiles from train ratings (needed after load())."""
        self.n_users = n_users
        self._build_profiles(u, i, r.astype(np.float32))
        return self

    # ------------------------------------------------------------------ #
    def _num_den(self, users):
        Xu = self.X[users]
        num = Xu @ self.S
        den = (Xu != 0).astype(np.float32) @ self.absS
        return num, den

    def predict_pairs(self, u, i, batch_users=8000):
        """Predicted ratings for arbitrary (u, i) pairs (batched over users)."""
        u = np.asarray(u); i = np.asarray(i)
        uniq = np.unique(u)
        pos = {int(uid): k for k, uid in enumerate(uniq)}
        u_local = np.array([pos[int(x)] for x in u], dtype=np.int64)
        preds = np.empty(len(u), dtype=np.float32)
        for s in range(0, len(uniq), batch_users):
            block = uniq[s:s + batch_users]
            num, den = self._num_den(block)
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = self.item_mean[None, :] + np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
            in_block = (u_local >= s) & (u_local < s + len(block))
            preds[in_block] = rate[u_local[in_block] - s, i[in_block]]
        return np.clip(preds, 1.0, 5.0)

    def score_users(self, users):
        """Top-N ranking score: weighted-similarity aggregate (masked)."""
        num, den = self._num_den(np.asarray(users))
        return np.where(den > 0, num, -1e9)

    def similar_items(self, item_idx, top=10):
        col = self.S[:, item_idx]
        order = np.argsort(col)[::-1][:top]
        return [(int(j), float(col[j])) for j in order if col[j] > 0]

    def save(self, path):
        np.savez_compressed(path, S=self.S, item_mean=self.item_mean,
                            global_mean=self.global_mean)

    def load(self, path):
        z = np.load(path)
        self.S = z["S"]; self.absS = np.abs(self.S)
        self.item_mean = z["item_mean"]; self.global_mean = float(z["global_mean"])
        self.n_items = len(self.item_mean)
        return self
