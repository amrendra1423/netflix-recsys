"""
Exploratory data analysis for the Netflix Prize dataset.

Computes the headline characteristics the problem statement asks for - rating
distribution, user activity, content popularity (long tail), sparsity and
temporal trends - and renders figures. Aggregates are cached so figures can be
re-rendered without re-parsing the raw data.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_aggregates(u, m, r, d):
    """Return small aggregate arrays from raw (user, movie, rating, date)."""
    rating_counts = np.bincount(r.astype(np.int64), minlength=6)[1:6]
    user_counts = np.bincount(u.astype(np.int64))
    user_counts = user_counts[user_counts > 0]
    movie_counts = np.bincount(m.astype(np.int64))
    movie_counts = movie_counts[movie_counts > 0]
    months, month_vol = np.unique((d // 100).astype(np.int64), return_counts=True)
    return {
        "n_ratings": np.int64(len(r)),
        "n_users": np.int64(len(user_counts)),
        "n_movies": np.int64(len(movie_counts)),
        "rating_counts": rating_counts,
        "user_counts": user_counts.astype(np.int32),
        "movie_counts": movie_counts.astype(np.int32),
        "months": months, "month_vol": month_vol,
        "mean_rating": np.float64((np.arange(1, 6) * rating_counts).sum() / rating_counts.sum()),
    }


def save_aggregates(agg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **agg)


def load_aggregates(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def summarize(agg):
    n_r, n_u, n_m = int(agg["n_ratings"]), int(agg["n_users"]), int(agg["n_movies"])
    uc, mc = agg["user_counts"], agg["movie_counts"]
    rc = agg["rating_counts"]
    sparsity = 1.0 - n_r / (n_u * n_m)
    sorted_mc = np.sort(mc)[::-1]
    cum = np.cumsum(sorted_mc) / sorted_mc.sum()
    top1pct = sorted_mc[:max(1, n_m // 100)].sum() / sorted_mc.sum()
    top20pct_share = cum[max(0, int(0.2 * n_m) - 1)]
    return {
        "n_ratings": n_r, "n_users": n_u, "n_movies": n_m,
        "sparsity_pct": round(sparsity * 100, 3),
        "mean_rating": round(float(agg["mean_rating"]), 3),
        "rating_distribution": {str(i + 1): int(rc[i]) for i in range(5)},
        "ratings_per_user_median": int(np.median(uc)),
        "ratings_per_user_mean": round(float(uc.mean()), 1),
        "ratings_per_user_max": int(uc.max()),
        "ratings_per_movie_median": int(np.median(mc)),
        "ratings_per_movie_mean": round(float(mc.mean()), 1),
        "ratings_per_movie_max": int(mc.max()),
        "share_of_ratings_top20pct_movies": round(float(top20pct_share), 3),
        "share_of_ratings_top1pct_movies": round(float(top1pct), 3),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
BLUE = "#3b6ea5"; ORANGE = "#e07b39"


def fig_rating_distribution(agg, path):
    rc = agg["rating_counts"]; pct = rc / rc.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(np.arange(1, 6), pct, color=BLUE, edgecolor="white")
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.4, f"{p:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("Star rating"); ax.set_ylabel("% of ratings")
    ax.set_title("Rating distribution"); ax.set_xticks(range(1, 6))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_user_activity(agg, path):
    uc = agg["user_counts"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(uc, bins=np.logspace(0, np.log10(uc.max()), 40), color=ORANGE, edgecolor="white")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("# ratings by a user (log)"); ax.set_ylabel("# users (log)")
    ax.set_title("User activity distribution (long tail)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_movie_popularity(agg, path):
    mc = np.sort(agg["movie_counts"])[::-1]
    cum = np.cumsum(mc) / mc.sum()
    x = np.arange(1, len(mc) + 1) / len(mc) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(np.arange(1, len(mc) + 1), mc, color=BLUE)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("Movie rank (log)"); a1.set_ylabel("# ratings (log)")
    a1.set_title("Content popularity (long tail)")
    a2.plot(x, cum * 100, color=ORANGE)
    a2.axhline(80, ls="--", c="grey", lw=1)
    a2.set_xlabel("% of movies (most popular first)"); a2.set_ylabel("% of all ratings")
    a2.set_title("Cumulative rating share")
    for a in (a1, a2): a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fig_ratings_over_time(agg, path):
    months = agg["months"]; vol = agg["month_vol"]
    order = np.argsort(months); months, vol = months[order], vol[order]
    labels = [f"{int(m)//100}-{int(m)%100:02d}" for m in months]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(months)), vol, color=BLUE)
    step = max(1, len(months) // 10)
    ax.set_xticks(range(0, len(months), step))
    ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("# ratings"); ax.set_title("Rating volume over time")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def make_all_figures(agg, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    fig_rating_distribution(agg, os.path.join(fig_dir, "rating_distribution.png"))
    fig_user_activity(agg, os.path.join(fig_dir, "user_activity.png"))
    fig_movie_popularity(agg, os.path.join(fig_dir, "movie_popularity.png"))
    fig_ratings_over_time(agg, os.path.join(fig_dir, "ratings_over_time.png"))
