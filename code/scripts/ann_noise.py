"""P0 control: ANN noise floor. Same model, HNSW index built twice with different
insertion orders; reports each build's recall of the exact top-K' and the overlap
between the two builds. UNTESTED here (needs faiss-cpu).

  python scripts/ann_noise.py --doc enc/bge-base/doc.npy --q enc/bge-base/q.npy --kprime 500 1000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reusemap.search import normalize_rows, topk_ip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--doc", required=True)
ap.add_argument("--q", required=True)
ap.add_argument("--kprime", type=int, nargs="+", default=[500])
ap.add_argument("--n-queries", type=int, default=2000)
ap.add_argument("--M", type=int, default=32)
ap.add_argument("--ef-construction", type=int, default=200)
ap.add_argument("--ef-search", type=int, default=1000)
a = ap.parse_args()

import faiss  # noqa: E402

X = normalize_rows(np.load(a.doc))
Q = normalize_rows(np.load(a.q))
rng = np.random.default_rng(0)
Q = Q[rng.choice(len(Q), min(a.n_queries, len(Q)), replace=False)]
kmax = max(a.kprime)
exact, _ = topk_ip(Q, X, kmax)


def build(seed):
    perm = np.random.default_rng(seed).permutation(len(X))
    idx = faiss.IndexHNSWFlat(X.shape[1], a.M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = a.ef_construction
    idx.add(X[perm])
    idx.hnsw.efSearch = max(a.ef_search, kmax)
    _, I = idx.search(Q, kmax)
    return np.where(I >= 0, perm[np.clip(I, 0, None)], -1)


A, B = build(1), build(2)
for kp in sorted(a.kprime):
    ra = np.mean([len(np.intersect1d(A[i, :kp], exact[i, :kp])) / kp for i in range(len(Q))])
    rb = np.mean([len(np.intersect1d(B[i, :kp], exact[i, :kp])) / kp for i in range(len(Q))])
    ab = np.mean([len(np.intersect1d(A[i, :kp], B[i, :kp])) / kp for i in range(len(Q))])
    print(f"K'={kp}: recall(build1)={ra:.4f} recall(build2)={rb:.4f} overlap(build1,build2)={ab:.4f}")
