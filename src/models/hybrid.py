"""
Hybrid ranker: personalised model + popularity (optional task / ensemble).

Rating-prediction models (SVD, Item-CF) are tuned for RMSE, and ranking purely
by predicted rating ignores how *popular* an item is - which dominates offline
Top-N quality. This hybrid keeps the base model's rating predictions (so RMSE is
unchanged) but, for ranking, blends the base model's per-user scores with a
log-popularity prior (both standardised):

    rank_score = z(base_score) + alpha * z(log(1 + popularity))

It demonstrates the rating-accuracy vs ranking trade-off and is a simple,
effective ensemble that lifts the MF model's MAP@10 substantially.
"""
from __future__ import annotations

import numpy as np


class PopBlendModel:
    def __init__(self, base, item_popularity, alpha=1.0, name=None):
        self.base = base
        self.alpha = alpha
        pop = np.log1p(item_popularity).astype(np.float32)
        self.popz = (pop - pop.mean()) / (pop.std() + 1e-9)
        self.name = name or f"Hybrid ({base.name} + popularity)"
        self.n_items = len(item_popularity)

    def predict_pairs(self, u, i):
        return self.base.predict_pairs(u, i)        # RMSE == base model's RMSE

    def score_users(self, users):
        s = np.asarray(self.base.score_users(users), dtype=np.float32)
        s = (s - s.mean()) / (s.std() + 1e-9)
        return s + self.alpha * self.popz[None, :]
