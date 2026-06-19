"""
Recommendation generation: Top-K lists, item-item "more like this", and simple
explanations.

A model exposes ``score_users(users) -> (len(users), n_items)``. To recommend
for a user we score every item, mask the ones already rated in train, and take
the K highest-scoring items.
"""
from __future__ import annotations

import numpy as np


def build_user_index(ds):
    """Return dicts: train and (relevant) test items/ratings grouped by user."""
    def group(uu, ii, rr):
        order = np.argsort(uu, kind="stable")
        uu, ii, rr = uu[order], ii[order], rr[order]
        counts = np.bincount(uu, minlength=ds.n_users)
        splits = np.cumsum(counts)[:-1]
        items = np.split(ii, splits)
        rats = np.split(rr, splits)
        return items, rats
    tr_i, tr_r = group(ds.train_u, ds.train_i, ds.train_r)
    te_i, te_r = group(ds.test_u, ds.test_i, ds.test_r)
    return {"train_i": tr_i, "train_r": tr_r, "test_i": te_i, "test_r": te_r}


def title_of(ds, titles, item_idx):
    return titles.get(int(ds.raw_movie_ids[item_idx]), f"movie#{int(ds.raw_movie_ids[item_idx])}")


def top_k(model, user_idx, seen_items, k=10):
    scores = np.asarray(model.score_users([user_idx]))[0].astype(np.float64)
    scores[seen_items] = -np.inf
    part = np.argpartition(scores, -k)[-k:]
    order = part[np.argsort(-scores[part])]
    return order, scores[order]


def recommend(model, user_idx, ds, titles, idx, k=10, relevance_threshold=3.5):
    seen = idx["train_i"][user_idx]
    rel_test = set(idx["test_i"][user_idx][idx["test_r"][user_idx] >= relevance_threshold].tolist())
    items, scores = top_k(model, user_idx, seen, k)
    recs = []
    for it, sc in zip(items, scores):
        recs.append({
            "item_idx": int(it),
            "title": title_of(ds, titles, it),
            "score": round(float(sc), 4),
            "hit": int(it) in rel_test,
        })
    return recs


def liked_history(ds, user_idx, titles, idx, n=8):
    items = idx["train_i"][user_idx]; rats = idx["train_r"][user_idx]
    order = np.argsort(-rats)[:n]
    return [(title_of(ds, titles, items[o]), float(rats[o])) for o in order]


def similar_movies(icf, item_idx, ds, titles, n=8):
    out = []
    for j, sim in icf.similar_items(item_idx, top=n):
        out.append((title_of(ds, titles, j), round(sim, 3)))
    return out


def explain_itemcf(icf, user_idx, rec_item_idx, ds, titles, idx, n=3):
    """Why was rec_item recommended? The user's rated movies most similar to it."""
    seen = idx["train_i"][user_idx]
    sims = icf.S[seen, rec_item_idx]                 # similarity of rec to each rated item
    order = np.argsort(-sims)[:n]
    return [(title_of(ds, titles, seen[o]), round(float(sims[o]), 3))
            for o in order if sims[o] > 0]
