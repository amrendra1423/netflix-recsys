"""
Standalone Streamlit app for Streamlit Community Cloud (or `streamlit run`).

Self-contained: loads the trained models in-process via CloudRecommender (no
separate API needed) and is memory-light enough for the free tier. Deploy by
pointing Streamlit Cloud at this file:  deploy/streamlit_app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_recommender import CloudRecommender

st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading models…")
def load_recommender():
    return CloudRecommender()


rec = load_recommender()
info = rec.info()

st.title("🎬 Netflix Prize — Personalized Recommendations")
st.caption(f"{info['n_users']:,} users · {info['n_movies']:,} movies · "
           "models: Item-CF, SVD, Hybrid, Popularity, Baseline")

with st.sidebar:
    st.header("Controls")
    sample = rec.sample_user_ids(25)
    user_choice = st.selectbox("Demo user", sample, index=0)
    user_manual = st.text_input("…or enter a user id", value=str(user_choice))
    user_id = int(user_manual) if user_manual.strip().isdigit() else user_choice
    model = st.selectbox("Ranking model", info["models"], index=0)
    k = st.slider("How many recommendations", 5, 30, 10)
    st.markdown("---")
    st.caption("Item-CF gives the best ranking; SVD is best at rating "
               "prediction; Hybrid = SVD + popularity.")

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("👤 This user likes")
    try:
        prof = rec.user_profile(user_id, 10)
        st.caption(f"User {user_id} — {prof['n_train_ratings']} ratings in history")
        st.dataframe(pd.DataFrame(prof["likes"]).rename(
            columns={"title": "Movie", "rating": "★"}),
            hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Unknown user: {e}")

with col2:
    st.subheader(f"⭐ Top-{k} recommendations · `{model}`")
    try:
        recs = rec.recommend(user_id, model, k)
        df = pd.DataFrame(recs)
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

with c3:
    st.subheader("🔎 More like this")
    q = st.text_input("Search a movie", value="The Matrix")
    if q.strip():
        results = rec.search_movies(q, 15)
        if results:
            label = {f"{r['title']}  (id {r['movie_id']})": r["movie_id"] for r in results}
            pick = st.selectbox("Pick a movie", list(label.keys()))
            sim = rec.similar(label[pick], 10)
            if sim["similar"]:
                st.dataframe(pd.DataFrame(sim["similar"]).rename(
                    columns={"title": "Similar movie", "similarity": "Similarity"})[
                    ["Similar movie", "Similarity"]],
                    hide_index=True, use_container_width=True)
            else:
                st.info("No strong neighbours for this title.")
        else:
            st.info("No movies matched.")

with c4:
    st.subheader("🎯 Predict a rating")
    rmodel = st.selectbox("Rating model", info["rating_models"], index=0)
    q2 = st.text_input("Movie to score", value="Pulp Fiction")
    res = rec.search_movies(q2, 15) if q2.strip() else []
    if res:
        label2 = {f"{r['title']}  (id {r['movie_id']})": r["movie_id"] for r in res}
        pick2 = st.selectbox("Pick", list(label2.keys()), key="predmovie")
        if st.button("Predict"):
            pr = rec.predict(user_id, label2[pick2], rmodel)
            st.metric(f"Predicted rating · {pr['title']}", f"{pr['predicted_rating']} ★")
