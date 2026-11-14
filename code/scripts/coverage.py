"""Union-of-windows coverage per K' for one source encoding. CPU only; no GPU, no re-encoding.

  python scripts/coverage.py --docs enc/MiniLM/doc.npy --queries enc/MiniLM/q.npy \
      --kprime 50,100,200,500,1000,2000 --out coverage/MiniLM.json
  # restrict to the queries that can actually be served (replay v3's arrival population):
  python scripts/coverage.py --docs ... --queries ... --qids sub1m/query_sets/servable.txt

--exclude-qids and --weights are applied in that order and they are not the same filter.
--exclude-qids drops the ids you name whatever their weight; --weights drops only the ids whose
weight is 0. So under a weighted workload the two can disagree with sweep.py, which builds its
servable set from `all_query_weights() > 0` alone: a query you exclude by id but which carries
weight > 0 is dropped here and kept there. That is intended -- the exclusion list is for queries
that must not be served at all (the evaluation-only secondary set), the weights for queries the
workload never draws -- but if you pass both, pass the same exclusion list to sweep.py's pair
config too, or the two coverage numbers will differ by those queries' windows.

      coverage(K') = |union over the queries q of source top-K'(q)| / N

Why this is worth its own script, run before anything else:

 1. It is the CEILING on the effect. Under replay v3 a window strategy must encode at least
    this union to serve the query population, and a full rebuild costs exactly N, so no window
    strategy can save more than 1 - coverage(K') of the encoding work. On the first 1M run that
    union was 86-88% of N at K'=500 over ~7k queries, i.e. a ceiling of 12-14%, and the paper
    has to state it or a reviewer reads "10% cheaper" without knowing 14% was the maximum.

 2. It decides whether `coverage` can be a map feature at all. The query set and K' are fixed
    across the pairs of one workload, so coverage varies between pairs only through the SOURCE
    model's window concentration. If every candidate source lands at the same value it predicts
    nothing across pairs (DESIGN.md Claim A), and it is an explanatory quantity instead.

 3. Together with G(K') from measure.py it gives the predicted optimal window depth without any
    replay: the smallest K' whose G(K') <= eps, whose achievable saving is 1 - coverage(K').
    Deeper windows buy the quality that makes a strategy admissible and cost coverage, which is
    the saving - that trade-off is the paper's economics. scripts/predict.py joins the two.

Memory: documents stay memory-mapped (via `pair.load_vectors`, which only normalises - and so
only copies into RAM - when the vectors are not already unit-norm; `encode.py` writes them
normalised, and the script prints which path it took). Queries are processed in blocks, and one
boolean mask of N is kept per K' (6 MB for six depths at N = 1e6), never a Q x K' index matrix.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap.pair import load_vectors  # noqa: E402
from reusemap.search import topk_ip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--docs", required=True, help="doc.npy of the SOURCE model (as written by encode.py)")
ap.add_argument("--queries", required=True, help="q.npy of the SOURCE model")
ap.add_argument("--kprime", default="50,100,200,500,1000,2000", help="comma-separated window depths")
ap.add_argument("--qids", default=None,
                help="file of query row indices to use (one per line). Default: every query. Pass "
                     "the servable set (traffic pool + primary eval, no TREC-DL) to match replay v3")
ap.add_argument("--exclude-qids", default=None,
                help="file of query row indices to DROP (e.g. sub1m/query_sets/trecdl.txt, which is "
                     "evaluation-only and never served, so it must not count towards coverage)")
ap.add_argument("--weights", default=None,
                help="workload weights .npy. Queries with weight 0 are dropped, because the v3 "
                     "stream can never serve them. Pass this for a weighted workload (ORCAS, the "
                     "Zipf code sweep) or this script and scripts/sweep.py will be measuring "
                     "different query sets: sweep drops zero-weight queries via all_query_weights, "
                     "and without --weights this script does not")
ap.add_argument("--block", type=int, default=512, help="queries per matrix product")
ap.add_argument("--out", default=None)
ap.add_argument("--name", default=None, help="label for the JSON (default: the docs directory)")
a = ap.parse_args()

ks = sorted({int(k) for k in a.kprime.split(",") if k.strip()})
if not ks or ks[0] < 1:
    sys.exit("--kprime must be positive integers")

# load_vectors, not normalize_rows: normalize_rows divides, which materialises the whole matrix
# in RAM (3 GB at 1M x 768, 10 GB at 1M x 2560). load_vectors checks a sample of norms and keeps
# the memmap when the vectors are already unit-norm, which is what encode.py writes - and
# topk_ip takes the blockwise memmap path for exactly that case.
X = load_vectors(a.docs)
Q = load_vectors(a.queries)
N = X.shape[0]
if N == 0 or Q.shape[0] == 0:
    sys.exit(f"empty input: {N} documents, {Q.shape[0]} queries (a failed encode writes a 0-row .npy)")
print(f"documents: {type(X).__name__} {X.shape} {X.dtype}"
      + ("" if isinstance(X, np.memmap) else "  (NOT memory-mapped: the vectors were not "
                                             "unit-norm, so they were normalised into RAM)"))
if Q.shape[1] != X.shape[1]:
    sys.exit(f"dimension mismatch: queries {Q.shape[1]} vs documents {X.shape[1]} - are these the "
             f"same model's encodings?")
qids = (np.loadtxt(a.qids, dtype=np.int64, ndmin=1) if a.qids else np.arange(Q.shape[0]))
bad = qids[(qids < 0) | (qids >= Q.shape[0])]
if len(bad):
    sys.exit(f"--qids has {len(bad)} index/indices outside [0, {Q.shape[0]}) (e.g. {bad[:3].tolist()})")
qids = np.unique(qids)
if a.exclude_qids:
    drop = np.loadtxt(a.exclude_qids, dtype=np.int64, ndmin=1)
    before = len(qids)
    qids = np.setdiff1d(qids, drop)
    print(f"excluded {before - len(qids)} of {before} queries listed in {a.exclude_qids}")
if a.weights:
    wv = np.asarray(np.load(a.weights), float)
    if len(wv) < Q.shape[0]:
        sys.exit(f"--weights has {len(wv)} entries but there are {Q.shape[0]} queries")
    if len(wv) > Q.shape[0]:
        print(f"warning: --weights has {len(wv)} entries for {Q.shape[0]} queries; the trailing "
              f"{len(wv) - Q.shape[0]} are ignored. If that file came from a larger query set, "
              f"the weights may not line up with these rows at all.")
    before = len(qids)
    qids = qids[wv[qids] > 0]
    print(f"excluded {before - len(qids)} of {before} queries with weight 0 in {a.weights}")
if len(qids) == 0:
    sys.exit("no queries left after --qids / --exclude-qids / --weights")
kmax = min(max(ks), N)   # the deepest usable window, i.e. clamped to N -- not max(ks)
ks = [k for k in ks if k <= N] or [kmax]

masks = {k: np.zeros(N, bool) for k in ks}
t0 = time.time()
for a0 in range(0, len(qids), a.block):
    blk = qids[a0:a0 + a.block]
    top, _ = topk_ip(Q[blk], X, kmax)
    for k in ks:
        idx = np.unique(top[:, :k])
        assert (idx >= 0).all(), "padded (-1) ids in the source top-k: depth exceeds N"
        masks[k][idx] = True
    if a0 // a.block % 20 == 0:
        print(f"  {a0 + len(blk)}/{len(qids)} queries  ({time.time() - t0:.0f}s)", flush=True)

rows = []
for k in ks:
    cov = float(masks[k].sum()) / N
    # What the union would be if each window were a uniformly random K'-subset. The measured
    # value is lower when windows concentrate on the same documents; the gap is the thing that
    # can differ between source models.
    indep = float(1.0 - (1.0 - min(k, N) / N) ** len(qids))
    rows.append({"kprime": int(k), "covered": int(masks[k].sum()), "coverage": cov,
                 "ceiling_frac": 1.0 - cov, "coverage_if_independent": indep,
                 "concentration": (indep - cov)})

out = {"name": a.name or os.path.basename(os.path.dirname(os.path.abspath(a.docs))),
       "weights_file": os.path.abspath(a.weights) if a.weights else None,
       "exclude_qids_file": os.path.abspath(a.exclude_qids) if a.exclude_qids else None,
       "docs": os.path.abspath(a.docs), "queries": os.path.abspath(a.queries),
       "N": int(N), "n_queries": int(len(qids)), "dim": int(X.shape[1]),
       "qids_file": os.path.abspath(a.qids) if a.qids else None,
       "seconds": time.time() - t0, "rows": rows}

print(f"\n{'K-prime':>8s} {'covered':>12s} {'coverage':>9s} {'ceiling':>8s} {'if indep':>9s} {'concentr':>9s}")
for r in rows:
    print(f"{r['kprime']:8d} {r['covered']:12d} {r['coverage']:9.4f} {r['ceiling_frac']:8.4f} "
          f"{r['coverage_if_independent']:9.4f} {r['concentration']:9.4f}")
print(f"\n{len(qids)} queries over N={N}. 'ceiling' is the largest share of the encoding work any "
      f"window strategy can save against a full rebuild at that depth.")
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)
