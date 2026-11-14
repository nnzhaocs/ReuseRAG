"""Synthetic model pairs for testing the pipeline (NOT for any paper number).

A latent vector u (dim L) defines relevance. The source model sees the first `src_dims`
latent coordinates plus noise; the target sees `tgt_dims` coordinates plus noise, through
its own random projection. A noisy low-dimensional view plays BM25. Varying the views
yields pairs with different compatibility and residual miss.
"""
import argparse
import json
import os

import numpy as np


def norm(A):
    return (A / np.linalg.norm(A, axis=1, keepdims=True)).astype(np.float32)


def make_pair(out, N=4000, Q=400, L=24, src_dims=16, tgt_dims=24, ds=32, dt=48,
              src_noise=0.3, tgt_noise=0.2, identity=False, seed=0, bm25_depth=40):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((N, L))
    Uq = U[rng.integers(0, N, Q)] + 0.6 * rng.standard_normal((Q, L))  # queries near docs

    # one projection per model, shared by its documents and queries
    Ps = np.random.default_rng(seed + 1).standard_normal((src_dims, ds))
    doc_src = norm(U[:, :src_dims] @ Ps + src_noise * rng.standard_normal((N, ds)))
    q_src = norm(Uq[:, :src_dims] @ Ps + src_noise * rng.standard_normal((Q, ds)))
    if identity:
        doc_tgt, q_tgt = doc_src.copy(), q_src.copy()
    else:
        Pt = np.random.default_rng(seed + 2).standard_normal((tgt_dims, dt))
        doc_tgt = norm(U[:, :tgt_dims] @ Pt + tgt_noise * rng.standard_normal((N, dt)))
        q_tgt = norm(Uq[:, :tgt_dims] @ Pt + tgt_noise * rng.standard_normal((Q, dt)))
    # relevance from the full latent space: top-1 rel=2, next 2 rel=1
    S = Uq @ U.T
    assert N >= 3, f"make_pair writes 3 qrels per query, so it needs N >= 3; got {N}"
    top = np.argsort(-S, axis=1)[:, :3]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "qrels.tsv"), "w") as f:
        for qi in range(Q):
            for r, d in enumerate(top[qi]):
                f.write(f"{qi}\t{d}\t{2 if r == 0 else 1}\n")
    # BM25 stand-in: a noisy view of 8 latent dims
    Pb = rng.standard_normal((8, 16))
    db = norm(U[:, :8] @ Pb + 0.8 * rng.standard_normal((N, 16)))
    qb = norm(Uq[:, :8] @ Pb + 0.8 * rng.standard_normal((Q, 16)))
    bm = np.argsort(-(qb @ db.T), axis=1)[:, :bm25_depth]
    np.save(os.path.join(out, "doc_src.npy"), doc_src)
    np.save(os.path.join(out, "doc_tgt.npy"), doc_tgt)
    np.save(os.path.join(out, "q_src.npy"), q_src)
    np.save(os.path.join(out, "q_tgt.npy"), q_tgt)
    np.save(os.path.join(out, "bm25.npy"), bm.astype(np.int64))
    np.save(os.path.join(out, "doc_len.npy"), rng.lognormal(4, 0.6, N))
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"synthetic": True, "src_dims": src_dims, "tgt_dims": tgt_dims,
                   "src_noise": src_noise, "tgt_noise": tgt_noise, "identity": identity}, fh)


SUITE = {
    "syn_identity": dict(identity=True),
    "syn_rotation": dict(src_dims=24, src_noise=0.05, tgt_noise=0.05),  # same info, new projection
    "syn_near": dict(src_dims=24, src_noise=0.2),
    "syn_close": dict(src_dims=22, src_noise=0.15),
    "syn_mid": dict(src_dims=16, src_noise=0.3),
    "syn_far": dict(src_dims=10, src_noise=0.5),
    "syn_veryfar": dict(src_dims=6, src_noise=0.8),
    "syn_mid2": dict(src_dims=18, src_noise=0.45, seed=7),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--Q", type=int, default=400)
    a = ap.parse_args()
    for name, kw in SUITE.items():
        make_pair(os.path.join(a.out, name), N=a.N, Q=a.Q, **kw)
        print("wrote", name)
