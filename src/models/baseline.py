"""
Bias baseline model.

Predicts r_ui = mu + b_u + b_i, where mu is the global mean and b_u / b_i are
regularised user and item biases. Biases are estimated with the standard
alternating closed-form updates (Koren 2009):

    b_i = sum_{u}(r_ui - mu - b_u) / (lambda_i + |R(i)|)
    b_u = sum_{i}(r_ui - mu - b_i) / (lambda_u + |R(u)|)

This is a deceptively strong baseline: it captures "this user rates high" and
"this movie is widely liked" effects, and every serious model should beat it.
All updates are fully vectorised with ``np.bincount``.
"""
from __future__ import annotations

import numpy as np

import config


class BaselineModel:
    name = "Baseline (bias)"

    def __init__(self, reg_user: float = config.BASELINE_REG_USER,
                 reg_item: float = config.BASELINE_REG_ITEM,
                 n_iter: int = config.BASELINE_ITERS):
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.n_iter = n_iter

    def fit(self, u, i, r, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items
        r = r.astype(np.float64)
        self.mu = float(r.mean())
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        u_counts = np.bincount(u, minlength=n_users)
        i_counts = np.bincount(i, minlength=n_items)

        for _ in range(self.n_iter):
            # item biases given user biases
            resid = r - self.mu - self.b_u[u]
            self.b_i = np.bincount(i, weights=resid, minlength=n_items) / (
                self.reg_item + i_counts)
            # user biases given item biases
            resid = r - self.mu - self.b_i[i]
            self.b_u = np.bincount(u, weights=resid, minlength=n_users) / (
                self.reg_user + u_counts)
        return self

    def predict_pairs(self, u, i):
        pred = self.mu + self.b_u[u] + self.b_i[i]
        return np.clip(pred, 1.0, 5.0)

    def score_users(self, users):
        """(len(users), n_items) score matrix used for ranking."""
        return self.mu + self.b_u[users][:, None] + self.b_i[None, :]

    def save(self, path):
        np.savez(path, mu=self.mu, b_u=self.b_u, b_i=self.b_i)

    def load(self, path):
        z = np.load(path)
        self.mu = float(z["mu"]); self.b_u = z["b_u"]; self.b_i = z["b_i"]
        self.n_users, self.n_items = len(self.b_u), len(self.b_i)
        return self
