"""
Popularity reference model.

A non-personalised baseline that ranks every user by global movie popularity
(number of training ratings). It is a deliberately simple reference, but on
offline Top-N evaluation popularity is a notoriously strong baseline because
popular movies appear in many users' held-out sets. Including it makes the
ranking comparison honest and contextualises the personalised models.

For rating prediction it falls back to the per-item mean.
"""
from __future__ import annotations

import numpy as np


class PopularityModel:
    name = "Popularity (reference)"

    def fit(self, u, i, r, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items
        r = r.astype(np.float32)
        self.global_mean = float(r.mean())
        counts = np.bincount(i, minlength=n_items)
        sums = np.bincount(i, weights=r, minlength=n_items)
        self.item_mean = np.where(counts > 0, sums / np.maximum(counts, 1),
                                  self.global_mean).astype(np.float32)
        self.popularity = counts.astype(np.float32)
        return self

    def predict_pairs(self, u, i):
        return np.clip(self.item_mean[i], 1.0, 5.0)

    def score_users(self, users):
        return np.repeat(self.popularity[None, :], len(users), axis=0)

    def save(self, path):
        np.savez(path, item_mean=self.item_mean, popularity=self.popularity,
                 global_mean=self.global_mean)

    def load(self, path):
        z = np.load(path)
        self.item_mean = z["item_mean"]; self.popularity = z["popularity"]
        self.global_mean = float(z["global_mean"]); self.n_items = len(self.item_mean)
        return self
