# Deploy on Streamlit Community Cloud (+ GitHub)

Streamlit Community Cloud runs **one** Streamlit app straight from a GitHub repo
(no separate backend). So we use the self-contained app **`deploy/streamlit_app.py`**,
which loads the trained models in-process via `deploy/cloud_recommender.py`. It is
memory-light (~200 MB — scores one user at a time from the saved similarity
matrix, not the full 320 MB matrix), so it fits the free tier.

> The FastAPI + `dashboard.py` + Docker setup is for self-hosting. For Streamlit
> Cloud, use `deploy/streamlit_app.py`.

## 0. (Optional) Run it locally first
```powershell
pip install -r requirements.txt
streamlit run deploy/streamlit_app.py
```
→ opens http://localhost:8501. No API needed — it's all in one process.

## 1. Put the project on GitHub

A partial `.git` folder may exist in this directory (the tool that generated the
project couldn't finalize it). **Delete it first**, then init cleanly:

```powershell
cd "D:\Projects\Recommendation  System for Personalized Content Discovery\netflix_recsys"
Remove-Item -Recurse -Force .\.git    # only if a .git folder is present
git init -b main
git add .
git commit -m "Netflix Prize recommender: EDA, models, evaluation, Streamlit app"
```

The trained artifacts the app needs (`data/dataset.npz`, `outputs/models/*.npz`,
`data/movie_titles.csv`) are committed on purpose (~25 MB total — `.gitignore`
is already set up for this). The ~2 GB raw Netflix text files are excluded.

Confirm the artifacts are staged:
```powershell
git ls-files | findstr "dataset.npz models/ movie_titles.csv"
```
If that prints nothing, force-add them: `git add -f data/dataset.npz outputs/models/*.npz data/movie_titles.csv`.

Create the GitHub repo and push — either with the GitHub CLI:
```powershell
gh repo create netflix-recsys --public --source . --remote origin --push
```
…or via the website: create an empty repo at github.com/new (name it e.g.
`netflix-recsys`, no README), then:
```powershell
git remote add origin https://github.com/<your-username>/netflix-recsys.git
git push -u origin main
```

## 2. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and **sign in with GitHub** (authorize it).
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/netflix-recsys`
   - **Branch:** `main`
   - **Main file path:** `deploy/streamlit_app.py`
4. (Optional) **Advanced settings →** Python 3.11. Dependencies are read from
   the root `requirements.txt` automatically.
5. Click **Deploy**. First build takes a few minutes; then you get a public URL
   like `https://<your-app>.streamlit.app`.

## 3. Troubleshooting

- **`ModuleNotFoundError`** → make sure the root `requirements.txt` is committed
  (it lists `streamlit`, `numpy`, `pandas`).
- **`FileNotFoundError: dataset.npz` / `itemcf.npz`** → the model artifacts
  weren't pushed (likely git-ignored). Run the `git add -f …` line above, commit,
  and push again. On the cloud the app expects `data/dataset.npz`,
  `outputs/models/*.npz`, and `data/movie_titles.csv` in the repo.
- **Memory** → the app peaks around ~200 MB, within the ~1 GB free tier. If you
  later scale the dataset way up, raise resources or precompute Top-N lists
  (`deploy/precompute.py`).
- **Reboot/redeploy** → from the app's menu on Streamlit Cloud, or just push a
  new commit — it redeploys automatically.
