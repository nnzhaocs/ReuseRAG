"""Assemble a pair directory from two models' encodings plus shared qrels/BM25/lengths.

  python scripts/assemble_pair.py --src enc/MiniLM --tgt enc/bge-base --data sub1m \\
      --bm25 sub1m/bm25.npy --out pairs/P2 --meta '{"source":"all-MiniLM-L6-v2","target":"bge-base-en-v1.5"}'
Each encoding dir holds doc.npy and q.npy. --data is the build_subsample.py output.
Files are symlinked, not copied.
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--tgt", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--bm25")
ap.add_argument("--weights", help="optional workload weights .npy (e.g. ORCAS frequency)")
ap.add_argument("--out", required=True)
ap.add_argument("--meta", default="{}")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def link(src, name):
    dst = os.path.join(a.out, name)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def rows(path):
    return int(np.load(path, mmap_mode="r").shape[0])  # memory-mapped: the header only


def lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


# Refuse a mismatch here, with the numbers, instead of letting it surface later as a bare
# assertion inside Pair.__init__ - or, worse, as a silently truncated score (e5-base reported
# 0.000 on 6,980 of 7,077 queries because nothing compared these counts).
want_docs = lines(os.path.join(a.data, "corpus.tsv"))
want_q = lines(os.path.join(a.data, "queries.tsv"))
problems = []
for role, d in (("src", a.src), ("tgt", a.tgt)):
    for fn, want, what in (("doc.npy", want_docs, "corpus.tsv lines"),
                           ("q.npy", want_q, "queries.tsv lines")):
        path = os.path.join(d, fn)
        if not os.path.exists(path):
            problems.append(f"{role}: {path} is missing")
            continue
        got = rows(path)
        if got != want:
            problems.append(f"{role}: {path} has {got} rows but {a.data} has {want} {what}")
if problems:
    sys.exit("refusing to assemble a pair with mismatched inputs:\n  " + "\n  ".join(problems)
             + "\nRe-encode the offending side under its contract (scripts/encode.py), then "
               "check it with scripts/encode_check.py.")

link(os.path.join(a.src, "doc.npy"), "doc_src.npy")
link(os.path.join(a.src, "q.npy"), "q_src.npy")
link(os.path.join(a.tgt, "doc.npy"), "doc_tgt.npy")
link(os.path.join(a.tgt, "q.npy"), "q_tgt.npy")
link(os.path.join(a.data, "qrels.tsv"), "qrels.tsv")
if a.bm25:
    link(a.bm25, "bm25.npy")
if a.weights:
    link(a.weights, "weights.npy")
### encoding is explicit: build_code_corpus.py writes utf-8, and reading it back under a
### non-utf-8 system locale (some CI images) fails on the first non-ASCII identifier
with open(os.path.join(a.data, "corpus.tsv"), encoding="utf-8") as fh:
    lens = [len(line.rstrip("\n").split("\t")[-1]) for line in fh]
np.save(os.path.join(a.out, "doc_len.npy"), np.array(lens))
with open(os.path.join(a.out, "meta.json"), "w") as fh:
    json.dump(json.loads(a.meta), fh, indent=1)
print("assembled", a.out)
