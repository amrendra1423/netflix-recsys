"""
Shared inference service for the Netflix recommender.

Loads the saved dataset + models once and answers recommendation queries. Used
by both the FastAPI app (deploy/api.py) and the Streamlit dashboard
(deploy/dashboard.py). Pure NumPy/pandas under the hood.

Note on ids: the public API speaks the original Netflix user/movie ids; this
service maps them to the contiguous internal indices the models use.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# make the repo importable whether run from repo root or deploy/
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import data_processing as dp
import recommend as rc
from models import (BaselineModel, SVDModel, ItemCFModel, PopularityModel,
                    PopBlendModel)

RANKERS = ("itemcf", "svd", "hybrid", "popularity", "baseline")
RATING_MODELS = ("svd", "itemcf", "baseline")


def _titles_path():
    for p in (config.MOVIE_TITLES_FILE,
              os.path.join(config.PROCESSED_DIR, "movie_titles.csv")):
        if os.path.exists(p):
            return p
    return None


class RecommenderService:
    def __init__(self, dataset_path=None, model_dir=None):
        dataset_path = dataset_path or os.path.join(config.PROCESSED_DIR, "dataset.npz")
        model_dir = model_dir or config.MODEL_DIR
        self.ds = dp.Dataset.load(dataset_path)
        tp = _titles_path()
        self.titles = dp.load_movie_titles(tp) if tp else {}
        self.idx = rc.build_user_index(self.ds)

        # id <-> index maps
        self.user_raw2idx = {int(r): i for i, r in enumerate(self.ds.raw_user_ids)}
        self.movie_raw2idx = {int(r): i for i, r in enumerate(self.ds.raw_movie_ids)}

        # models
        self.models = {}
        self.models["popularity"] = PopularityModel().load(os.path.join(model_dir, "popularity.npz"))
        self.models["baseline"] = BaselineModel().load(os.path.join(model_dir, "baseline.npz"))
        self.models["svd"] = SVDModel().load(os.path.join(model_dir, "svd.npz"))
        icf = ItemCFModel().load(os.path.join(model_dir, "itemcf.npz"))
        icf.attach_profiles(self.ds.train_u, self.ds.train_i, self.ds.train_r, self.ds.n_users)
        self.models["itemcf"] = icf
        self.models["hybrid"] = PopBlendModel(self.models["svd"], self.ds.item_popularity, alpha=1.0)

    # ------------------------------------------------------------------ #
    def _title(self, item_idx):
        return rc.title_of(self.ds, self.titles, item_idx)

    def info(self):
        return {
            "n_users": int(self.ds.n_users),
            "n_movies": int(self.ds.n_items),
            "models": list(self.models.keys()),
            "rating_models": list(RATING_MODELS),
            "relevance_threshold": config.RELEVANCE_THRESHOLD,
        }

    def sample_user_ids(self, n=20, seed=0):
        n_train = np.array([len(x) for x in self.idx["train_i"]])
        n_rel = np.array([int((self.idx["test_r"][u] >= config.RELEVANCE_THRESHOLD).sum())
                          for u in range(self.ds.n_users)])
        elig = np.where((n_train >= 20) & (n_rel >= 5))[0]
        rng = np.random.default_rng(seed)
        pick = rng.choice(elig, size=min(n, len(elig)), replace=False)
        return [int(self.ds.raw_user_ids[i]) for i in sorted(pick)]

    def search_movies(self, query, limit=20):
        q = query.lower().strip()
        out = []
        for i in range(self.ds.n_items):
            t = self._title(i)
            if q in t.lower():
                out.append({"movie_id": int(self.ds.raw_movie_ids[i]), "title": t,
                            "n_ratings": int(self.ds.item_popularity[i])})
                if len(out) >= limit:
                    break
        return out

    def user_profile(self, user_id, n=10):
        if int(user_id) not in self.user_raw2idx:
            raise KeyError(f"unknown user_id {user_id}")
        u = self.user_raw2idx[int(user_id)]
        liked = rc.liked_history(self.ds, u, self.titles, self.idx, n=n)
        return {"user_id": int(user_id),
                "n_train_ratings": int(len(self.idx["train_i"][u])),
                "likes": [{"title": t, "rating": r} for t, r in liked]}

    def recommend(self, user_id, model="itemcf", k=10):
        if model not in self.models:
            raise KeyError(f"unknown model '{model}'")
        if int(user_id) not in self.user_raw2idx:
            raise KeyError(f"unknown user_id {user_id}")
        u = self.user_raw2idx[int(user_id)]
        recs = rc.recommend(self.models[model], u, self.ds, self.titles, self.idx,
                            k=k, relevance_threshold=config.RELEVANCE_THRESHOLD)
        for n, r in enumerate(recs, 1):
            r["rank"] = n
            r["movie_id"] = int(self.ds.raw_movie_ids[r.pop("item_idx")])
            r["held_out_match"] = r.pop("hit")     # demo-only signal
        return {"user_id": int(user_id), "model": model, "recommendations": recs}

    def similar(self, movie_id, k=10):
        if int(movie_id) not in self.movie_raw2idx:
            raise KeyError(f"unknown movie_id {movie_id}")
        mi = self.movie_raw2idx[int(movie_id)]
        sims = self.models["itemcf"].similar_items(mi, top=k)
        return {"movie_id": int(movie_id), "title": self._title(mi),
                "similar": [{"movie_id": int(self.ds.raw_movie_ids[j]),
                             "title": self._title(j), "similarity": round(s, 4)}
                            for j, s in sims]}

    def predict(self, user_id, movie_id, model="svd"):
        if model not in RATING_MODELS:
            raise KeyError(f"'{model}' is not a rating model {RATING_MODELS}")
        if int(user_id) not in self.user_raw2idx:
            raise KeyError(f"unknown user_id {user_id}")
        if int(movie_id) not in self.movie_raw2idx:
            raise KeyError(f"unknown movie_id {movie_id}")
        u = self.user_raw2idx[int(user_id)]
        mi = self.movie_raw2idx[int(movie_id)]
        pred = float(self.models[model].predict_pairs(np.array([u]), np.array([mi]))[0])
        return {"user_id": int(user_id), "movie_id": int(movie_id),
                "title": self._title(mi), "model": model,
                "predicted_rating": round(pred, 3)}


_SINGLETON = None


def get_service():
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = RecommenderService()
    return _SINGLETON
