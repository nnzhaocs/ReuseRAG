"""BM25 top-Kb per query with bm25s (pure Python; much faster than rank_bm25 at 1M docs).
UNTESTED here. Parameters default to Anserini's defaults (k1=0.9, b=0.4); state them as such.

  pip install bm25s
  python scripts/bm25_run.py --corpus corpus.tsv --queries queries.tsv --k 1000 --out bm25.npy
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--queries", required=True)
ap.add_argument("--k", type=int, default=1000)
ap.add_argument("--k1", type=float, default=0.9)
ap.add_argument("--b", type=float, default=0.4)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import bm25s  # noqa: E402

docs = [line.rstrip("\n").split("\t")[-1] for line in open(a.corpus, encoding="utf-8")]
qs = [line.rstrip("\n").split("\t")[-1] for line in open(a.queries, encoding="utf-8")]
retr = bm25s.BM25(k1=a.k1, b=a.b)
retr.index(bm25s.tokenize(docs, stopwords="en"))
idx, _ = retr.retrieve(bm25s.tokenize(qs, stopwords="en"), k=a.k)
ix = np.asarray(idx, dtype=np.int64)
np.save(a.out, ix)
print("wrote", a.out, ix.shape)
if ix.ndim == 2 and ix.shape[1] < a.k:
    print(f"warning: only {ix.shape[1]} of the {a.k} requested BM25 columns exist "
          f"({len(docs)} documents indexed). Any config with bm25_depth > {ix.shape[1]} "
          f"will fail later inside heat.py, which cannot say that the cause was here.")
