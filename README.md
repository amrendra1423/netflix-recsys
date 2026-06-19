# Netflix Prize — Recommendation System for Personalized Content Discovery

> **🎬 Live demo (Streamlit Cloud):** https://amrendra1423-netflix-recsys.streamlit.app
>
> [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amrendra1423-netflix-recsys.streamlit.app)

A reproducible recommendation engine built on the **Netflix Prize dataset**
(~100M ratings, 480K users, 17,770 movies). It learns user preferences, predicts
ratings for unseen movies, and generates personalized Top-N recommendations.

Three recommendation approaches are implemented **from scratch in NumPy** and
compared, plus a popularity reference and a popularity-blend hybrid:

| Model | Type | What it captures |
|-------|------|------------------|
| **Baseline (bias)** | `mu + b_u + b_i` | global + user + item rating tendencies |
| **SVD** | biased matrix factorization (Funk-SVD, SGD) | latent user/item factors |
| **Item-based CF** | centered-cosine item-item neighborhood | "users who liked X also liked Y" |
| Popularity | most-rated movies | non-personalized ranking reference |
| Hybrid (SVD + popularity) | ensemble re-ranking | rating accuracy + popularity prior |

The dependency surface is intentionally tiny (`numpy`, `pandas`, `matplotlib`) so
results are easy to reproduce.

---

## Headline results (default subset: 40,000 users × 2,000 movies)

| Model | RMSE ↓ | MAE ↓ | MAP@10 ↑ | Precision@10 | NDCG@10 | HitRate@10 | Coverage |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Popularity (reference) | 1.003 | 0.804 | 0.098 | 0.138 | 0.186 | 0.649 | 0.03 |
| Baseline (bias) | 0.920 | 0.724 | 0.006 | 0.022 | 0.021 | 0.186 | 0.01 |
| SVD (matrix factorization) | 0.893 | 0.696 | 0.005 | 0.012 | 0.014 | 0.107 | 0.36 |
| **Item-based CF** | **0.890** | **0.681** | **0.106** | 0.139 | 0.190 | 0.603 | **0.81** |
| Hybrid (SVD + popularity) | 0.893 | 0.696 | 0.105 | **0.144** | **0.196** | **0.669** | 0.10 |

**Key takeaways**
1. **Matrix factorization and item-CF clearly win rating prediction** (RMSE 0.89
   vs 0.92 baseline); popularity (a mean predictor) is worst at RMSE.
2. **Rating accuracy ≠ ranking quality.** SVD has the 2nd-best RMSE but among the
   *worst* MAP@10: ranking purely by predicted rating surfaces niche high-rated
   titles (e.g. cult TV box-sets) that users rarely have in their held-out set.
3. **Item-based CF is the best all-rounder** — top RMSE *and* top MAP@10 *and*
   the highest catalog coverage (0.81 → diverse, personalized lists).
4. **A simple popularity blend lifts SVD's MAP@10 ~20×** (0.005 → 0.105) with
   *identical* RMSE — a concrete demonstration of the accuracy/ranking trade-off
   and an effective lightweight ensemble.
5. Offline Top-N is **popularity-biased**: a pure popularity ranker scores
   MAP@10 0.098, so personalized models must justify themselves on *coverage*
   and *quality of explanations*, not MAP alone.

Full numbers: `outputs/results/metrics.csv`. Sample recommendations and
"more like this" lists: `outputs/recommendations/sample_recommendations.md`.

---

## Repository structure

```
netflix_recsys/
├── config.py                 # all paths & hyper-parameters (single source of truth)
├── requirements.txt
├── src/
│   ├── data_processing.py    # parse raw files, build subset, train/test split
│   ├── eda.py                # aggregates + figures
│   ├── evaluation.py         # RMSE, MAE, MAP@10, P/R/NDCG@10, HitRate, Coverage
│   ├── recommend.py          # Top-K, item-item similarity, explanations
│   └── models/
│       ├── baseline.py       # global + user + item bias
│       ├── svd.py            # Funk-SVD (mini-batch SGD)
│       ├── item_cf.py        # item-based collaborative filtering
│       ├── popularity.py     # popularity reference
│       └── hybrid.py         # SVD + popularity blend
├── scripts/
│   ├── 01_prepare_data.py    # -> data/dataset.npz
│   ├── 02_run_eda.py         # -> outputs/figures/, outputs/results/eda_stats.json
│   ├── 03_train.py           # -> outputs/models/*.npz
│   ├── 04_evaluate.py        # -> outputs/results/metrics.{json,csv}
│   └── 05_recommend.py       # -> outputs/recommendations/*
├── notebooks/
│   └── walkthrough.ipynb     # end-to-end narrative (EDA + results + examples)
└── outputs/                  # figures, models, results, recommendations
```

---

## Setup & reproduction

**1. Data.** Download the Netflix Prize data
(https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) and place
`combined_data_1..4.txt`, `movie_titles.csv`, `probe.txt`, `qualifying.txt` in the
dataset folder. By default the code expects the repo to live *inside* that folder
(raw files one level up). Otherwise set the data directory explicitly:

```bash
export NETFLIX_DATA_DIR="/path/to/netflix-prize-data"
```

**2. Install dependencies.**

```bash
pip install -r requirements.txt
```

**3. Run the pipeline** (each step caches its output, so steps are independent):

```bash
python scripts/01_prepare_data.py     # parse + subset + split  -> data/dataset.npz
python scripts/02_run_eda.py          # EDA figures + stats
python scripts/03_train.py            # train & save all models
python scripts/04_evaluate.py         # RMSE + MAP@10 + ranking metrics
python scripts/05_recommend.py        # sample Top-10 recommendations & examples
```

Everything is seeded (`config.RANDOM_SEED = 42`) and reproducible.

---

## Methodology

### Data processing
The raw files interleave a movie-header line (`<movie_id>:`) with rating lines
(`<user_id>,<rating>,<date>`). `data_processing.py` streams them in chunks
(reading only the id column as a string so movie headers are detected as NaN
ratings), reconstructs the per-row movie id with a vectorized segment fill, and
emits compact numeric arrays.

### Subset (why, and how)
The full dataset is ~100M ratings / ~2 GB. The problem statement explicitly
allows training on a subset under compute constraints, so we build a **dense
subset**: the **top 2,000 most-rated movies** and a random sample of **40,000
active users** (≥ 20 ratings). This removes extreme cold-start noise, keeps the
item-item similarity matrix in memory, and lets all models train/compare in
minutes. Scale up via `N_DATA_FILES`, `TOP_N_MOVIES`, `N_SAMPLE_USERS` in
`config.py`. Resulting subset: **2,686,283 train / 651,893 test ratings**,
95.8% sparse.

### Train/test split
A **per-user random hold-out**: for each user, 20% of their ratings are held out
for test (guaranteeing ≥ 5 train ratings remain). This evaluates the model's
ability to predict each user's *own* unseen ratings, which is the right setting
for personalization. Implemented vectorized in `per_user_split`.

### Relevance definition
Per the problem statement, a held-out (test) movie is **relevant** to a user iff
the user's actual rating is **≥ 3.5 stars**.

### Top-10 generation procedure
For a user we (1) score every one of the 2,000 movies with the model, (2) remove
movies the user already rated in *train*, and (3) take the 10 highest-scoring
remaining movies. Each model uses its natural scoring function: predicted rating
(baseline, SVD), weighted item-similarity aggregate (Item-CF), popularity
(Popularity), or standardized blend (Hybrid).

### MAP@10 methodology
For each evaluated user, average precision at 10:
`AP@10 = (1 / min(R,10)) * Σ_{n=1..10} Precision@n · rel_n`, where `rel_n = 1`
if the n-th recommendation is one of the user's relevant test movies, and `R` is
the user's number of relevant test movies. **MAP@10** is the mean of `AP@10`
over evaluated users (5,000 sampled users with ≥ 1 relevant test item). Items
recommended that are not in the test set are treated as non-relevant (standard
offline assumption). We additionally report MAE, Precision@10, Recall@10,
NDCG@10, Hit-Rate@10 and catalog Coverage.

### Models
- **Baseline** — regularized biases solved by alternating closed-form updates.
- **SVD** — `r = mu + b_u + b_i + p_u·q_i`, trained with mini-batch SGD using
  mean-gradient updates (stable) and a sort+`reduceat` scatter (fast, pure NumPy);
  ~11 epochs, 40 factors.
- **Item-based CF** — item-item centered-cosine (Pearson) similarity with
  shrinkage and a min-support floor, top-40 neighbors. Predicts ratings via the
  neighbor-weighted formula; ranks via the weighted-similarity aggregate.
- **Popularity / Hybrid** — references for the ranking discussion.

---

## EDA highlights (from `combined_data_1.txt`, 24.1M ratings)
- **470,758 users · 4,499 movies · 98.86% sparse.**
- **Mean rating ≈ 3.6** with a strong positivity bias (66% of ratings are 4–5).
- **Heavy-tailed activity:** median user rates 24 movies, but the mostactive rates 4,467.
- **Extreme content long tail:** the top 20% of movies receive **89%** of all
  ratings (top 1% → 23%). Figures in `outputs/figures/`.

---

## Future improvements
- Train on the full 100M ratings (the pipeline scales via config) and add
  time-aware splits.
- Learning-to-rank objectives (BPR / WARP) to optimize Top-N directly rather
  than RMSE, closing the accuracy↔ranking gap.
- Incorporate movie metadata (year, and external genre data) for a content-aware
  hybrid and better cold-start handling.
- Neural collaborative filtering / two-tower models for richer interactions.

## Notes
The models, metrics and figures in `outputs/` were generated by the commands
above on the default subset. Implemented with NumPy/pandas/matplotlib only — no
specialized RecSys libraries — so the algorithms are fully transparent.
