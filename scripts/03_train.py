"""Step 3 - train models and save them to outputs/models/.

Usage:
  python scripts/03_train.py                 # train any models not yet cached
  python scripts/03_train.py --force         # retrain all
  python scripts/03_train.py --models svd    # train a subset
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config
import data_processing as dp
from models import BaselineModel, SVDModel, ItemCFModel, PopularityModel

BUILDERS = {
    "popularity": PopularityModel,
    "baseline": BaselineModel,
    "svd": SVDModel,
    "itemcf": ItemCFModel,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="popularity,baseline,svd,itemcf")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ds = dp.Dataset.load(os.path.join(config.PROCESSED_DIR, "dataset.npz"))
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    print(f"Dataset: {ds.n_users:,} users x {ds.n_items:,} items, "
          f"train={len(ds.train_r):,}")

    for name in args.models.split(","):
        name = name.strip()
        path = os.path.join(config.MODEL_DIR, f"{name}.npz")
        if os.path.exists(path) and not args.force:
            print(f"[{name}] cached -> skip (use --force to retrain)")
            continue
        model = BUILDERS[name]()
        t0 = time.time()
        model.fit(ds.train_u, ds.train_i, ds.train_r, ds.n_users, ds.n_items)
        model.save(path)
        print(f"[{name}] trained & saved in {time.time() - t0:.1f}s -> {path}")


if __name__ == "__main__":
    main()
