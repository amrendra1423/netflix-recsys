"""
(Optional, for scaling) Precompute Top-N recommendations for every user and
"similar items" for every movie, and dump them to JSON. A production API can
then serve recommendations as O(1) lookups instead of scoring at request time -
the standard pattern for batch recommendation serving.

    python deploy/precompute.py --model itemcf --k 20
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from service import RecommenderService

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="itemcf")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "precomputed"))
    args = ap.parse_args()

    svc = RecommenderService()
    ds = svc.ds
    model = svc.models[args.model]
    idx = svc.idx
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # ---- per-user Top-N (batched scoring, mask train-seen) ----
    recs = {}
    for s in range(0, ds.n_users, args.batch):
        users = np.arange(s, min(s + args.batch, ds.n_users))
        scores = np.asarray(model.score_users(users), dtype=np.float64)
        for row, u in enumerate(users):
            scores[row, idx["train_i"][u]] = -np.inf
        part = np.argpartition(scores, -args.k, axis=1)[:, -args.k:]
        for row, u in enumerate(users):
            top = part[row][np.argsort(-scores[row, part[row]])]
            recs[int(ds.raw_user_ids[u])] = [int(ds.raw_movie_ids[j]) for j in top]
    with open(os.path.join(args.out, f"top_{args.k}_{args.model}.json"), "w") as fh:
        json.dump(recs, fh)

    # ---- per-movie similar items (Item-CF) ----
    icf = svc.models["itemcf"]
    similar = {}
    for mi in range(ds.n_items):
        similar[int(ds.raw_movie_ids[mi])] = [int(ds.raw_movie_ids[j])
                                              for j, _ in icf.similar_items(mi, top=args.k)]
    with open(os.path.join(args.out, "similar_items.json"), "w") as fh:
        json.dump(similar, fh)

    # ---- title lookup so a thin server needs nothing else ----
    titles = {int(ds.raw_movie_ids[i]): svc._title(i) for i in range(ds.n_items)}
    with open(os.path.join(args.out, "titles.json"), "w") as fh:
        json.dump(titles, fh)

    print(f"Precomputed Top-{args.k} for {len(recs):,} users and similar items "
          f"for {len(similar):,} movies in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
