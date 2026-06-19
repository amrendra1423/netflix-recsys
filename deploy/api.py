"""
FastAPI REST service for the Netflix recommender.

Run (from the repo root, with deps installed):
    uvicorn deploy.api:app --host 0.0.0.0 --port 8000
or from this folder:
    cd deploy && uvicorn api:app --port 8000

Then open http://localhost:8000/docs for interactive Swagger docs.

Endpoints
---------
GET /                         service + dataset info
GET /health                   liveness probe
GET /users/sample?n=20        a few valid demo user ids
GET /users/{user_id}/profile  a user's most-liked movies
GET /recommend?user_id=&model=itemcf&k=10
GET /movies/search?q=matrix
GET /similar?movie_id=468&k=10
GET /predict?user_id=&movie_id=&model=svd
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from service import get_service, RANKERS, RATING_MODELS

app = FastAPI(
    title="Netflix Prize Recommender API",
    version="1.0",
    description="Personalized movie recommendations (Item-CF, SVD, hybrid).",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_svc = None


@app.on_event("startup")
def _load():
    global _svc
    _svc = get_service()    # loads dataset + models once


@app.get("/")
def root():
    return _svc.info()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/users/sample")
def users_sample(n: int = Query(20, ge=1, le=200)):
    return {"user_ids": _svc.sample_user_ids(n)}


@app.get("/users/{user_id}/profile")
def user_profile(user_id: int, n: int = Query(10, ge=1, le=50)):
    try:
        return _svc.user_profile(user_id, n)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/recommend")
def recommend(user_id: int,
              model: str = Query("itemcf", enum=list(RANKERS)),
              k: int = Query(10, ge=1, le=50)):
    try:
        return _svc.recommend(user_id, model, k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/movies/search")
def movies_search(q: str, limit: int = Query(20, ge=1, le=100)):
    return {"query": q, "results": _svc.search_movies(q, limit)}


@app.get("/similar")
def similar(movie_id: int, k: int = Query(10, ge=1, le=50)):
    try:
        return _svc.similar(movie_id, k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/predict")
def predict(user_id: int, movie_id: int,
            model: str = Query("svd", enum=list(RATING_MODELS))):
    try:
        return _svc.predict(user_id, movie_id, model)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
