# Deployment Guide

Serve the recommender as a **REST API** (FastAPI) and an **interactive dashboard**
(Streamlit). Both share one inference layer (`deploy/service.py`) that loads the
trained models from `outputs/models/` and the split from `data/dataset.npz`.

> Serving needs only the small artifacts (`dataset.npz` ~16 MB, models ~8 MB,
> bundled `movie_titles.csv`). The ~2 GB raw Netflix text files are **not** needed
> at serving time. Make sure you have run `scripts/01` and `scripts/03` once so
> those artifacts exist.

```
            ┌────────────┐      HTTP       ┌──────────────────┐
  user ───▶ │ Dashboard  │ ───────────────▶│  FastAPI service │
            │ (Streamlit)│   /recommend…   │  models in RAM   │
            └────────────┘                 └──────────────────┘
                 :8501                            :8000
```

## Option A — Docker (recommended, one command)

```bash
# from the repo root
docker compose -f deploy/docker-compose.yml up --build
```

- API:        http://localhost:8000  (interactive docs at **/docs**)
- Dashboard:  http://localhost:8501

The dashboard waits for the API healthcheck, then talks to it at
`http://api:8000`. Stop with `Ctrl-C` (or `docker compose ... down`).

## Option B — Local (no Docker)

```bash
pip install -r deploy/requirements.txt

# terminal 1 — API
uvicorn deploy.api:app --host 0.0.0.0 --port 8000

# terminal 2 — dashboard
API_URL=http://localhost:8000 streamlit run deploy/dashboard.py
```

If your raw files live elsewhere, set `NETFLIX_DATA_DIR` (titles fall back to the
bundled `data/movie_titles.csv` automatically).

## API reference

| Method & path | Purpose |
|---|---|
| `GET /` | dataset + model info |
| `GET /health` | liveness probe |
| `GET /users/sample?n=20` | valid demo user ids |
| `GET /users/{user_id}/profile?n=10` | a user's most-liked movies |
| `GET /recommend?user_id=&model=itemcf&k=10` | Top-K recommendations |
| `GET /movies/search?q=matrix` | find movies by title |
| `GET /similar?movie_id=468&k=10` | "more like this" |
| `GET /predict?user_id=&movie_id=&model=svd` | predicted rating |

```bash
curl "http://localhost:8000/recommend?user_id=713249&model=itemcf&k=5"
curl "http://localhost:8000/similar?movie_id=3925&k=5"          # Matrix Reloaded
curl "http://localhost:8000/predict?user_id=713249&movie_id=457&model=svd"
```

`model` accepts `itemcf` (best ranking), `svd`, `hybrid`, `popularity`,
`baseline`; `/predict` accepts `svd`, `itemcf`, `baseline`.

## Production & scaling notes

- **Precompute for scale.** At high QPS, don't score on every request — run
  `python deploy/precompute.py --model itemcf --k 20` to dump per-user Top-N and
  per-movie similar lists to JSON, and serve them as O(1) lookups (the standard
  batch-recommendation pattern). Refresh the job on a schedule as new ratings
  arrive.
- **Memory.** The API holds the Item-CF user-profile matrix (~320 MB for the
  default subset) for live scoring. To shrink the footprint, serve **SVD only**
  (a few MB of factors) or switch to the precomputed lookups above. Budget
  ~1 GB RAM for the default live setup.
- **Concurrency.** The API is stateless after startup — scale horizontally
  behind a load balancer, e.g. `uvicorn ... --workers 4` or `gunicorn -k
  uvicorn.workers.UvicornWorker`.
- **Cloud.** The image runs on any container host — Google Cloud Run, AWS
  ECS/Fargate, Fly.io, Render, or a plain VM. Push `netflix-recsys` to a registry
  and deploy; point the dashboard's `API_URL` at the API's public URL. The
  Streamlit app can also go to Streamlit Community Cloud.
- **Cold start.** New users/movies aren't in the trained subset; fall back to the
  `popularity` model for cold users and to title/metadata search for new movies
  until they accrue ratings.
