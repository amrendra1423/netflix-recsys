"""
Self-contained, memory-light recommender for single-process hosting
(e.g. Streamlit Community Cloud, where there is no separate API backend).

Unlike the API's service (which builds the full dense ~320 MB user-item matrix
for batch scoring), this scores **one user at a time** straight from the saved
item-item similarity matrix S (~16 MB) plus that user's train ratings. Total
memory stays well under a few hundred MB, so it fits free-tier limits.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import data_processing as dp
import recommend as rc
from models import SVDModel, BaselineModel, PopularityModel

RANKERS = ("itemcf", "svd", "hybrid", "popularity", "baseline")
RATING_MODELS = ("svd", "itemcf", "baseline")


def _titles_path():
    for p in (config.MOVIE_TITLES_FILE,
              os.path.join(config.PROCESSED_DIR, "movie_titles.csv")):
        if os.path.exists(p):
            return p
    return None


class CloudRecommender:
    def __init__(self, dataset_path=None, model_dir=None):
        dataset_path = dataset_path or os.path.join(config.PROCESSED_DIR, "dataset.npz")
        model_dir = model_dir or config.MODEL_DIR
        self.ds = dp.Dataset.load(dataset_path)
        tp = _titles_path()
        self.titles = dp.load_movie_titles(tp) if tp else {}
        self.idx = rc.build_user_index(self.ds)

        z = np.load(os.path.join(model_dir, "itemcf.npz"))
        self.S = z["S"]; self.item_mean = z["item_mean"]
        self.svd = SVDModel().load(os.path.join(model_dir, "svd.npz"))
        self.baseline = BaselineModel().load(os.path.join(model_dir, "baseline.npz"))
        self.pop = PopularityModel().load(os.path.join(model_dir, "popularity.npz"))

        lp = np.log1p(self.ds.item_popularity).astype(np.float32)
        self.logpopz = (lp - lp.mean()) / (lp.std() + 1e-9)
        self.user_raw2idx = {int(r): i for i, r in enumerate(self.ds.raw_user_ids)}
        self.movie_raw2idx = {int(r): i for i, r in enumerate(self.ds.raw_movie_ids)}

    # ---- scoring (one user) ---- #
    def _itemcf_num_den(self, u):
        rated = self.idx["train_i"][u]
        if len(rated) == 0:
            z = np.zeros(self.ds.n_items, np.float32)
            return z, z
        x = (self.idx["train_r"][u] - self.item_mean[rated]).astype(np.float32)
        rows = self.S[rated]                       # (|rated| x n_items)
        return x @ rows, np.abs(rows).sum(0)

    def _scores(self, u, model):
        if model == "itemcf":
            num, den = self._itemcf_num_den(u)
            return np.where(den > 0, num, -1e9).astype(np.float32)
        if model == "svd":
            return self.svd.score_users(np.array([u]))[0]
        if model == "hybrid":
            s = self.svd.score_users(np.array([u]))[0].astype(np.float32)
            s = (s - s.mean()) / (s.std() + 1e-9)
            return s + self.logpopz
        if model == "popularity":
            return self.ds.item_popularity.astype(np.float32)
        if model == "baseline":
            return self.baseline.mu + self.baseline.b_u[u] + self.baseline.b_i
        raise KeyError(f"unknown model '{model}'")

    # ---- public API ---- #
    def info(self):
        return {"n_users": int(self.ds.n_users), "n_movies": int(self.ds.n_items),
                "models": list(RANKERS), "rating_models": list(RATING_MODELS),
                "relevance_threshold": config.RELEVANCE_THRESHOLD}

    def _title(self, j):
        return rc.title_of(self.ds, self.titles, j)

    def sample_user_ids(self, n=25, seed=0):
        n_train = np.array([len(x) for x in self.idx["train_i"]])
        n_rel = np.array([int((self.idx["test_r"][u] >= config.RELEVANCE_THRESHOLD).sum())
                          for u in range(self.ds.n_users)])
        elig = np.where((n_train >= 20) & (n_rel >= 5))[0]
        rng = np.random.default_rng(seed)
        pick = rng.choice(elig, size=min(n, len(elig)), replace=False)
        return [int(self.ds.raw_user_ids[i]) for i in sorted(pick)]

    def search_movies(self, query, limit=20):
        q = query.lower().strip(); out = []
        for i in range(self.ds.n_items):
            t = self._title(i)
            if q in t.lower():
                out.append({"movie_id": int(self.ds.raw_movie_ids[i]), "title": t})
                if len(out) >= limit:
                    break
        return out

    def user_profile(self, user_id, n=10):
        u = self.user_raw2idx[int(user_id)]
        liked = rc.liked_history(self.ds, u, self.titles, self.idx, n=n)
        return {"user_id": int(user_id),
                "n_train_ratings": int(len(self.idx["train_i"][u])),
                "likes": [{"title": t, "rating": r} for t, r in liked]}

    def recommend(self, user_id, model="itemcf", k=10):
        u = self.user_raw2idx[int(user_id)]
        sc = self._scores(u, model).astype(np.float64).copy()
        sc[self.idx["train_i"][u]] = -np.inf
        part = np.argpartition(sc, -k)[-k:]
        order = part[np.argsort(-sc[part])]
        rel = set(self.idx["test_i"][u][self.idx["test_r"][u] >= config.RELEVANCE_THRESHOLD].tolist())
        return [{"rank": n + 1, "movie_id": int(self.ds.raw_movie_ids[j]),
                 "title": self._title(j), "score": round(float(sc[j]), 4),
                 "held_out_match": int(j) in rel} for n, j in enumerate(order)]

    def similar(self, movie_id, k=10):
        mi = self.movie_raw2idx[int(movie_id)]
        col = self.S[:, mi]; order = np.argsort(col)[::-1][:k]
        return {"movie_id": int(movie_id), "title": self._title(mi),
                "similar": [{"movie_id": int(self.ds.raw_movie_ids[j]),
                             "title": self._title(j), "similarity": round(float(col[j]), 4)}
                            for j in order if col[j] > 0]}

    def predict(self, user_id, movie_id, model="svd"):
        u = self.user_raw2idx[int(user_id)]; mi = self.movie_raw2idx[int(movie_id)]
        if model == "itemcf":
            num, den = self._itemcf_num_den(u)
            val = self.item_mean[mi] + (num[mi] / den[mi] if den[mi] > 0 else 0.0)
        elif model == "baseline":
            val = self.baseline.mu + self.baseline.b_u[u] + self.baseline.b_i[mi]
        else:
            val = float(self.svd.predict_pairs(np.array([u]), np.array([mi]))[0])
        return {"user_id": int(user_id), "movie_id": int(movie_id),
                "title": self._title(mi), "model": model,
                "predicted_rating": round(float(np.clip(val, 1, 5)), 3)}
