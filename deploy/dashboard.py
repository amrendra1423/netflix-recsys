"""
Streamlit dashboard for the Netflix recommender.

A thin client over the FastAPI service: pick a user and model, view their liked
history and Top-N recommendations, explore "more like this", and predict a
rating for any user-movie pair.

Run (with the API already running):
    API_URL=http://localhost:8000 streamlit run deploy/dashboard.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="wide")


@st.cache_data(ttl=60)
def api_get(path, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def check_api():
    try:
        api_get("/health")
        return True
    except Exception as e:
        st.error(f"Cannot reach the API at {API_URL}. Start it first "
                 f"(`uvicorn deploy.api:app --port 8000`).\n\n{e}")
        return False


st.title("🎬 Netflix Prize — Personalized Recommendations")
st.caption(f"Backend: {API_URL}")

if not check_api():
    st.stop()

info = api_get("/")
models = info["models"]

# ----------------------------- sidebar ---------------------------------- #
with st.sidebar:
    st.header("Controls")
    sample = api_get("/users/sample", n=25)["user_ids"]
    user_choice = st.selectbox("Demo user", sample, index=0)
    user_manual = st.text_input("...or enter a user id", value=str(user_choice))
    user_id = int(user_manual) if user_manual.strip().isdigit() else user_choice
    model = st.selectbox("Ranking model", models,
                         index=models.index("itemcf") if "itemcf" in models else 0)
    k = st.slider("How many recommendations", 5, 30, 10)
    st.markdown("---")
    st.metric("Users", f"{info['n_users']:,}")
    st.metric("Movies", f"{info['n_movies']:,}")

col1, col2 = st.columns([1, 1.3])

# --------------------------- user profile ------------------------------- #
with col1:
    st.subheader("👤 This user likes")
    try:
        prof = api_get(f"/users/{user_id}/profile", n=10)
        st.caption(f"User {user_id} — {prof['n_train_ratings']} ratings in history")
        st.dataframe(pd.DataFrame(prof["likes"]).rename(
            columns={"title": "Movie", "rating": "★"}),
            hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"No profile: {e}")

# --------------------------- recommendations ---------------------------- #
with col2:
    st.subheader(f"⭐ Top-{k} recommendations · `{model}`")
    try:
        rec = api_get("/recommend", user_id=user_id, model=model, k=k)
        df = pd.DataFrame(rec["recommendations"])
        df["match"] = df["held_out_match"].map({True: "✓", False: ""})
        st.dataframe(
            df[["rank", "title", "score", "match"]].rename(
                columns={"rank": "#", "title": "Movie", "score": "Score",
                         "match": "Held-out hit"}),
            hide_index=True, use_container_width=True)
        st.caption("‘Held-out hit’ = movie is in this user's hidden test set "
                   "with rating ≥ 3.5 (offline demo signal).")
    except Exception as e:
        st.warning(f"No recommendations: {e}")

st.markdown("---")
c3, c4 = st.columns(2)

# --------------------------- more like this ----------------------------- #
with c3:
    st.subheader("🔎 More like this")
    q = st.text_input("Search a movie", value="The Matrix")
    if q.strip():
        results = api_get("/movies/search", q=q, limit=15)["results"]
        if results:
            label = {f"{r['title']}  (id {r['movie_id']})": r["movie_id"] for r in results}
            pick = st.selectbox("Pick a movie", list(label.keys()))
            sim = api_get("/similar", movie_id=label[pick], k=10)
            st.dataframe(pd.DataFrame(sim["similar"]).rename(
                columns={"title": "Similar movie", "similarity": "Similarity"})[
                ["Similar movie", "Similarity"]],
                hide_index=True, use_container_width=True)
        else:
            st.info("No movies matched.")

# --------------------------- predict rating ----------------------------- #
with c4:
    st.subheader("🎯 Predict a rating")
    rmodel = st.selectbox("Rating model", info["rating_models"], index=0)
    q2 = st.text_input("Movie to score", value="Pulp Fiction")
    res = api_get("/movies/search", q=q2, limit=15)["results"] if q2.strip() else []
    if res:
        label2 = {f"{r['title']}  (id {r['movie_id']})": r["movie_id"] for r in res}
        pick2 = st.selectbox("Pick", list(label2.keys()), key="predmovie")
        if st.button("Predict"):
            try:
                pr = api_get("/predict", user_id=user_id, movie_id=label2[pick2], model=rmodel)
                st.metric(f"Predicted rating · {pr['title']}", f"{pr['predicted_rating']} ★")
            except Exception as e:
                st.warning(str(e))
