"""Step 4 - evaluate all models (RMSE, MAE, MAP@10, P/R/NDCG@10, HitRate,
Coverage) and write outputs/results/metrics.{json,csv}."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd

import config
import data_processing as dp
import evaluation as ev
from models import (BaselineModel, SVDModel, ItemCFModel, PopularityModel,
                    PopBlendModel)


def load_models(ds):
    md = config.MODEL_DIR
    pop = PopularityModel().load(os.path.join(md, "popularity.npz"))
    base = BaselineModel().load(os.path.join(md, "baseline.npz"))
    svd = SVDModel().load(os.path.join(md, "svd.npz"))
    icf = ItemCFModel().load(os.path.join(md, "itemcf.npz"))
    icf.attach_profiles(ds.train_u, ds.train_i, ds.train_r, ds.n_users)   # rebuild X
    hybrid = PopBlendModel(svd, ds.item_popularity, alpha=1.0)
    # ordered for the report
    return [pop, base, svd, icf, hybrid]


def main():
    ds = dp.Dataset.load(os.path.join(config.PROCESSED_DIR, "dataset.npz"))
    models = load_models(ds)
    print(f"Evaluating on {len(ds.test_r):,} test ratings "
          f"(relevance >= {config.RELEVANCE_THRESHOLD})\n")

    rows = []
    for m in models:
        t0 = time.time()
        res = ev.evaluate_model(m, ds, k=config.TOP_K, verbose=True)
        rows.append(res)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(os.path.join(config.RESULTS_DIR, "metrics.csv"))
    print("\n" + df.round(4).to_string())
    print(f"\nSaved -> {config.RESULTS_DIR}/metrics.json , metrics.csv")


if __name__ == "__main__":
    main()
