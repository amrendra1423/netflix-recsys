"""Step 2 - exploratory data analysis: compute aggregates, save stats + figures.

Uses combined_data_1.txt (a representative slice of the full catalogue) so the
long-tail / sparsity characteristics reflect the unfiltered data, not the dense
modelling subset. Aggregates are cached to data/eda_agg.npz for fast re-runs.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config
import data_processing as dp
import eda


def main():
    agg_path = os.path.join(config.PROCESSED_DIR, "eda_agg.npz")
    if os.path.exists(agg_path):
        print(f"Loading cached EDA aggregates <- {agg_path}")
        agg = eda.load_aggregates(agg_path)
    else:
        print("Parsing combined_data_1.txt for EDA...")
        t0 = time.time()
        u, m, r, d = dp.parse_raw(config.combined_data_files(1))
        print(f"  parsed {len(u):,} ratings in {time.time()-t0:.1f}s")
        agg = eda.compute_aggregates(u, m, r, d)
        eda.save_aggregates(agg, agg_path)
        print(f"  cached aggregates -> {agg_path}")

    stats = eda.summarize(agg)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "eda_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))

    eda.make_all_figures(agg, config.FIG_DIR)
    print(f"\nFigures -> {config.FIG_DIR}")
    for f in sorted(os.listdir(config.FIG_DIR)):
        print("  ", f)


if __name__ == "__main__":
    main()
