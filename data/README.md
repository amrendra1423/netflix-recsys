# data/

Place the raw Netflix Prize files in the **dataset folder** (by default the
parent of this repo, or wherever `NETFLIX_DATA_DIR` points):

```
combined_data_1.txt   combined_data_2.txt   combined_data_3.txt   combined_data_4.txt
movie_titles.csv      probe.txt             qualifying.txt
```

Download: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

Generated caches (created by the pipeline; safe to delete and regenerate):
- `subset_raw.npz` — parsed + subsetted raw ratings (skips the ~2 GB re-parse)
- `dataset.npz`    — re-indexed train/test split used by all models
- `eda_agg.npz`    — cached EDA aggregates
