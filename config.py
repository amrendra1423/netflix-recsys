"""
Central configuration for the Netflix Prize recommendation system.

All tunable parameters (paths, subset definition, model hyper-parameters and
evaluation settings) live here so experiments are reproducible and easy to tweak.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory containing the raw Netflix Prize files (combined_data_1..4.txt,
# movie_titles.csv, probe.txt, qualifying.txt). By default we assume the repo
# lives *inside* the dataset folder, so the raw files are one level up.
# Override with the NETFLIX_DATA_DIR environment variable if your layout differs.
DATA_DIR = os.environ.get("NETFLIX_DATA_DIR", os.path.dirname(REPO_ROOT))

OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
REC_DIR = os.path.join(OUTPUT_DIR, "recommendations")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data")

MOVIE_TITLES_FILE = os.path.join(DATA_DIR, "movie_titles.csv")


def combined_data_files(n_files):
    """Return the list of combined_data_*.txt paths to ingest."""
    return [os.path.join(DATA_DIR, f"combined_data_{i}.txt") for i in range(1, n_files + 1)]


# --------------------------------------------------------------------------- #
# Subset definition
# --------------------------------------------------------------------------- #
# The full dataset is ~100M ratings / ~2 GB. As the problem statement explicitly
# allows, we train on a dense subset of popular movies + active users. This
# removes extreme cold-start noise, keeps the item-item similarity matrix in
# memory, and lets us compare three models quickly and reproducibly.
# Scale up by raising N_DATA_FILES / TOP_N_MOVIES / N_SAMPLE_USERS.
N_DATA_FILES = 1          # how many combined_data_*.txt files to read (1..4)
TOP_N_MOVIES = 2000       # keep the N most-rated movies
MIN_USER_RATINGS = 20     # a user must have >= this many ratings to be "active"
N_SAMPLE_USERS = 40000    # randomly sample this many active users (caps subset size)
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Train / test split
# --------------------------------------------------------------------------- #
TEST_FRACTION = 0.20      # fraction of each user's ratings held out for testing
MIN_TRAIN_RATINGS = 5     # users need >= this many train ratings to be kept

# --------------------------------------------------------------------------- #
# Relevance (for ranking metrics): item is relevant iff actual rating >= 3.5
# --------------------------------------------------------------------------- #
RELEVANCE_THRESHOLD = 3.5

# --------------------------------------------------------------------------- #
# Model hyper-parameters
# --------------------------------------------------------------------------- #
# Baseline (global mean + regularised user/item biases)
BASELINE_REG_USER = 10.0
BASELINE_REG_ITEM = 25.0
BASELINE_ITERS = 12

# SVD / biased matrix factorisation (Funk-SVD, mini-batch SGD).
# Small batches recover SGD-like convergence; mean-gradient updates keep it
# stable. ~11 epochs reaches test RMSE ~0.89 on the default subset.
SVD_FACTORS = 40
SVD_EPOCHS = 11
SVD_LR = 0.03
SVD_REG = 0.02
SVD_BATCH = 20000

# Item-based collaborative filtering
ITEMCF_K = 40             # number of neighbours used in prediction
ITEMCF_SHRINKAGE = 50.0   # similarity shrinkage to down-weight low-overlap pairs
ITEMCF_MIN_SUPPORT = 5    # ignore item pairs co-rated by fewer than this many users

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
TOP_K = 10                # K for Top-K recommendation / MAP@K, Precision@K, ...
N_EVAL_USERS = 5000       # users sampled for (expensive) ranking metrics
