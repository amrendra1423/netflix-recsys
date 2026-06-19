"""
SVD / biased matrix factorisation (Funk-SVD), trained with mini-batch gradient
descent.

Model:   r_ui = mu + b_u + b_i + p_u . q_i
Loss:    sum (r_ui - r_hat)^2 + reg * (b_u^2 + b_i^2 + ||p_u||^2 + ||q_i||^2)

This is the model family that won the Netflix Prize, implemented in pure NumPy.
Per mini-batch we apply the *mean* gradient for each user/item: biases via
``np.bincount`` and latent factors via a sort + ``np.add.reduceat`` segment
reduction. Using the mean (not the sum) keeps steps stable even though popular
items recur many times per batch, and avoids the slow ``np.add.at`` scatter.
"""
from __future__ import annotations

import numpy as np

import config


class SVDModel:
    name = "SVD (matrix factorisation)"

    def __init__(self, n_factors=config.SVD_FACTORS, n_epochs=config.SVD_EPOCHS,
                 lr=config.SVD_LR, reg=config.SVD_REG, batch=config.SVD_BATCH,
                 seed=config.RANDOM_SEED, verbose=True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr0 = lr
        self.reg = reg
        self.batch = batch
        self.seed = seed
        self.verbose = verbose

    def _bias_step(self, b, idx, grad, lr, n):
        cnt = np.bincount(idx, minlength=n).astype(np.float32)
        s = np.bincount(idx, weights=grad, minlength=n).astype(np.float32)
        nz = cnt > 0
        b[nz] += lr * (s[nz] / cnt[nz])

    def _factor_step(self, F, idx, grad, lr):
        order = np.argsort(idx)
        uniq, starts = np.unique(idx[order], return_index=True)
        cnt = np.diff(np.append(starts, len(idx))).astype(np.float32)
        summed = np.add.reduceat(grad[order], starts, axis=0)
        F[uniq] += lr * (summed / cnt[:, None])

    def fit(self, u, i, r, n_users, n_items):
        rng = np.random.default_rng(self.seed)
        self.n_users, self.n_items = n_users, n_items
        u = u.astype(np.int32); i = i.astype(np.int32)
        r = r.astype(np.float32)
        self.mu = np.float32(r.mean())

        f = self.n_factors
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)
        self.P = (0.1 * rng.standard_normal((n_users, f))).astype(np.float32)
        self.Q = (0.1 * rng.standard_normal((n_items, f))).astype(np.float32)

        n = len(r)
        reg = np.float32(self.reg)
        for epoch in range(self.n_epochs):
            lr = np.float32(self.lr0 * (0.98 ** epoch))
            perm = rng.permutation(n)
            for s in range(0, n, self.batch):
                b = perm[s:s + self.batch]
                ub, ib, rb = u[b], i[b], r[b]
                pu, qi = self.P[ub], self.Q[ib]
                pred = self.mu + self.b_u[ub] + self.b_i[ib] + np.einsum("ij,ij->i", pu, qi)
                err = (rb - pred).astype(np.float32)

                self._bias_step(self.b_u, ub, err - reg * self.b_u[ub], lr, n_users)
                self._bias_step(self.b_i, ib, err - reg * self.b_i[ib], lr, n_items)
                self._factor_step(self.P, ub, err[:, None] * qi - reg * pu, lr)
                self._factor_step(self.Q, ib, err[:, None] * pu - reg * qi, lr)

            if self.verbose and (epoch % 5 == 4 or epoch == self.n_epochs - 1):
                pr = self.mu + self.b_u[u] + self.b_i[i] + np.einsum(
                    "ij,ij->i", self.P[u], self.Q[i])
                rmse = float(np.sqrt(np.mean((r - pr) ** 2)))
                print(f"  [SVD] epoch {epoch + 1:2d}/{self.n_epochs}  "
                      f"train RMSE={rmse:.4f}  lr={lr:.4f}", flush=True)
        return self

    def predict_pairs(self, u, i):
        pred = (self.mu + self.b_u[u] + self.b_i[i]
                + np.einsum("ij,ij->i", self.P[u], self.Q[i]))
        return np.clip(pred, 1.0, 5.0)

    def score_users(self, users):
        return (self.mu + self.b_u[users][:, None] + self.b_i[None, :]
                + self.P[users] @ self.Q.T)

    def save(self, path):
        np.savez(path, mu=self.mu, b_u=self.b_u, b_i=self.b_i, P=self.P, Q=self.Q)

    def load(self, path):
        z = np.load(path)
        self.mu = np.float32(z["mu"]); self.b_u = z["b_u"]; self.b_i = z["b_i"]
        self.P = z["P"]; self.Q = z["Q"]
        self.n_users, self.n_items = self.P.shape[0], self.Q.shape[0]
        self.n_factors = self.P.shape[1]
        return self
