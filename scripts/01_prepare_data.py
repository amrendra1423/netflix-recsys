"""Step 1 — parse raw data, build the subset, split, and cache to data/dataset.npz."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import data_processing as dp


def main():
    t0 = time.time()
    ds = dp.prepare_dataset()
    out = os.path.join(config.PROCESSED_DIR, "dataset.npz")
    ds.save(out)

    sparsity = 1.0 - (len(ds.train_r) + len(ds.test_r)) / (ds.n_users * ds.n_items)
    print("-" * 60)
    print(f"Users:            {ds.n_users:,}")
    print(f"Items:            {ds.n_items:,}")
    print(f"Train ratings:    {len(ds.train_r):,}")
    print(f"Test ratings:     {len(ds.test_r):,}")
    print(f"Matrix sparsity:  {sparsity * 100:.3f}% empty")
    print(f"Saved dataset ->  {out}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
