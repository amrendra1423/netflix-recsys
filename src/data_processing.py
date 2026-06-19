"""
Data processing pipeline for the Netflix Prize dataset.

Parses the raw combined_data_*.txt files (which interleave a movie header line
"<movie_id>:" with rating lines "<user_id>,<rating>,<date>"), builds a dense
subset of popular movies + active users, re-indexes ids to 0..N-1, and produces
a reproducible per-user train/test split. NumPy + pandas only.
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    pd.set_option("future.no_silent_downcasting", True)
except Exception:
    pass


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_file(path, chunksize=1_000_000):
    """Stream one combined_data file into numeric arrays.

    Returns user(int32), movie(int16), rating(int8), date(int32 YYYYMMDD).
    Only the id column is read as string; ratings are read as float so movie
    headers become NaN and are trivially detected. Keeps per-chunk memory low.
    """
    users, movies, ratings, dates = [], [], [], []
    carry = np.int32(-1)
    seen = 0
    reader = pd.read_csv(
        path, header=None, names=["c0", "c1", "c2"],
        dtype={"c0": str}, chunksize=chunksize, engine="c",
    )
    for chunk in reader:
        seen += len(chunk)
        if seen % 5_000_000 < chunksize:
            print(f"    ...{seen:,} lines read", flush=True)
        c0 = chunk["c0"].values
        rating_col = chunk["c1"].values
        is_hdr = np.isnan(rating_col)

        hdr_pos = np.flatnonzero(is_hdr)
        seg_ids = np.array([int(s[:-1]) for s in c0[hdr_pos]], dtype=np.int32)
        seg = np.searchsorted(hdr_pos, np.arange(len(c0)), side="right") - 1
        movie_full = np.where(seg >= 0, seg_ids[seg.clip(min=0)], carry).astype(np.int32)
        if len(seg_ids):
            carry = seg_ids[-1]

        keep = ~is_hdr
        users.append(c0[keep].astype(np.int32))
        movies.append(movie_full[keep].astype(np.int16))
        ratings.append(rating_col[keep].astype(np.int8))
        d = chunk["c2"].values[keep].astype("U10")
        dates.append(np.char.replace(d, "-", "").astype(np.int32))

    return (np.concatenate(users), np.concatenate(movies),
            np.concatenate(ratings), np.concatenate(dates))


def parse_raw(files):
    """Parse a list of combined_data files into concatenated arrays."""
    u_parts, m_parts, r_parts, d_parts = [], [], [], []
    for path in files:
        t0 = time.time()
        u, m, r, d = _parse_file(path)
        u_parts.append(u); m_parts.append(m); r_parts.append(r); d_parts.append(d)
        print(f"  parsed {os.path.basename(path)}: {len(u):,} ratings "
              f"({time.time() - t0:.1f}s)")
    return (np.concatenate(u_parts), np.concatenate(m_parts),
            np.concatenate(r_parts), np.concatenate(d_parts))


# --------------------------------------------------------------------------- #
# Subset construction
# --------------------------------------------------------------------------- #
def build_subset(user, movie, rating, date,
                 top_n_movies, min_user_ratings, n_sample_users, seed):
    """Filter raw ratings to popular movies + a sample of active users."""
    movie_counts = np.bincount(movie.astype(np.int64))
    user_counts = np.bincount(user.astype(np.int64))

    n_keep = min(top_n_movies, np.count_nonzero(movie_counts))
    top_movies = np.argsort(movie_counts)[::-1][:n_keep]
    movie_keep = np.zeros(movie_counts.shape[0], dtype=bool)
    movie_keep[top_movies] = True

    active = np.where(user_counts >= min_user_ratings)[0]
    rng = np.random.default_rng(seed)
    if len(active) > n_sample_users:
        active = rng.choice(active, size=n_sample_users, replace=False)
    user_keep = np.zeros(user_counts.shape[0], dtype=bool)
    user_keep[active] = True

    mask = movie_keep[movie] & user_keep[user]
    return user[mask], movie[mask], rating[mask], date[mask]


# --------------------------------------------------------------------------- #
# Re-indexing + split
# --------------------------------------------------------------------------- #
def per_user_split(u_idx, i_idx, rating, date, n_users,
                   test_fraction, min_train, seed):
    """Hold out a random test_fraction of each user's ratings (vectorised),
    guaranteeing at least min_train ratings remain in train."""
    rng = np.random.default_rng(seed)
    n = len(u_idx)
    rand = rng.random(n)
    order = np.lexsort((rand, u_idx))
    sorted_u = u_idx[order]

    counts = np.bincount(sorted_u, minlength=n_users)
    start = np.zeros(n_users, dtype=np.int64)
    start[1:] = np.cumsum(counts)[:-1]
    within_rank = np.arange(n) - start[sorted_u]

    n_test = np.floor(test_fraction * counts).astype(np.int64)
    n_test = np.clip(np.minimum(n_test, counts - min_train), 0, None)
    is_test_sorted = within_rank < n_test[sorted_u]

    is_test = np.zeros(n, dtype=bool)
    is_test[order] = is_test_sorted
    return ~is_test, is_test


@dataclass
class Dataset:
    n_users: int
    n_items: int
    raw_user_ids: np.ndarray
    raw_movie_ids: np.ndarray
    train_u: np.ndarray
    train_i: np.ndarray
    train_r: np.ndarray
    train_d: np.ndarray
    test_u: np.ndarray
    test_i: np.ndarray
    test_r: np.ndarray
    test_d: np.ndarray
    item_popularity: np.ndarray

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            n_users=self.n_users, n_items=self.n_items,
            raw_user_ids=self.raw_user_ids, raw_movie_ids=self.raw_movie_ids,
            train_u=self.train_u, train_i=self.train_i,
            train_r=self.train_r, train_d=self.train_d,
            test_u=self.test_u, test_i=self.test_i,
            test_r=self.test_r, test_d=self.test_d,
            item_popularity=self.item_popularity,
        )

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return Dataset(
            n_users=int(z["n_users"]), n_items=int(z["n_items"]),
            raw_user_ids=z["raw_user_ids"], raw_movie_ids=z["raw_movie_ids"],
            train_u=z["train_u"], train_i=z["train_i"],
            train_r=z["train_r"].astype(np.float32), train_d=z["train_d"],
            test_u=z["test_u"], test_i=z["test_i"],
            test_r=z["test_r"].astype(np.float32), test_d=z["test_d"],
            item_popularity=z["item_popularity"],
        )


def build_or_load_subset(cache_path, n_data_files, top_n_movies,
                         min_user_ratings, n_sample_users, seed, verbose=True):
    """Parse + subset, caching the small (raw-id) subset arrays to disk so the
    expensive ~2 GB parse only happens once."""
    if cache_path and os.path.exists(cache_path):
        if verbose:
            print(f"Loading cached subset <- {cache_path}")
        z = np.load(cache_path)
        return z["user"], z["movie"], z["rating"], z["date"]

    files = config.combined_data_files(n_data_files)
    if verbose:
        print(f"Parsing {n_data_files} raw file(s)...")
    user, movie, rating, date = parse_raw(files)
    if verbose:
        print(f"Total raw ratings: {len(user):,}")
    user, movie, rating, date = build_subset(
        user, movie, rating, date,
        top_n_movies, min_user_ratings, n_sample_users, seed)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, user=user, movie=movie, rating=rating, date=date)
        if verbose:
            print(f"Cached subset -> {cache_path}")
    return user, movie, rating, date


def prepare_dataset(
    n_data_files=config.N_DATA_FILES,
    top_n_movies=config.TOP_N_MOVIES,
    min_user_ratings=config.MIN_USER_RATINGS,
    n_sample_users=config.N_SAMPLE_USERS,
    test_fraction=config.TEST_FRACTION,
    min_train=config.MIN_TRAIN_RATINGS,
    seed=config.RANDOM_SEED,
    cache_subset=True,
    verbose=True,
):
    """End-to-end: parse -> subset -> reindex -> split -> Dataset."""
    cache_path = (os.path.join(config.PROCESSED_DIR, "subset_raw.npz")
                  if cache_subset else None)
    user, movie, rating, date = build_or_load_subset(
        cache_path, n_data_files, top_n_movies, min_user_ratings,
        n_sample_users, seed, verbose)
    if verbose:
        print(f"Subset ratings: {len(user):,} "
              f"({len(np.unique(user)):,} users x {len(np.unique(movie)):,} movies)")

    raw_user_ids, u_idx = np.unique(user, return_inverse=True)
    raw_movie_ids, i_idx = np.unique(movie, return_inverse=True)
    n_users, n_items = len(raw_user_ids), len(raw_movie_ids)
    u_idx = u_idx.astype(np.int32)
    i_idx = i_idx.astype(np.int32)
    rating = rating.astype(np.float32)

    train_mask, test_mask = per_user_split(
        u_idx, i_idx, rating, date, n_users, test_fraction, min_train, seed)

    item_pop = np.bincount(i_idx[train_mask], minlength=n_items).astype(np.int64)

    ds = Dataset(
        n_users=n_users, n_items=n_items,
        raw_user_ids=raw_user_ids, raw_movie_ids=raw_movie_ids,
        train_u=u_idx[train_mask], train_i=i_idx[train_mask],
        train_r=rating[train_mask], train_d=date[train_mask],
        test_u=u_idx[test_mask], test_i=i_idx[test_mask],
        test_r=rating[test_mask], test_d=date[test_mask],
        item_popularity=item_pop,
    )
    if verbose:
        print(f"Train ratings: {len(ds.train_r):,} | Test ratings: {len(ds.test_r):,}")
    return ds


# --------------------------------------------------------------------------- #
# Movie titles
# --------------------------------------------------------------------------- #
def load_movie_titles(path=config.MOVIE_TITLES_FILE):
    """Map raw movie id -> 'Title (Year)'. Robust to commas in titles."""
    titles = {}
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(",", 2)
            if len(parts) < 3:
                continue
            mid, year, title = parts
            try:
                mid = int(mid)
            except ValueError:
                continue
            year = year if year and year != "NULL" else "?"
            titles[mid] = f"{title} ({year})"
    return titles
