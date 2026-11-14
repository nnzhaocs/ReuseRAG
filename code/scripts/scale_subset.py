"""Nested intermediate corpora (e.g. 2M, 4M) cut from the full collection, with no re-encoding.

    python scripts/scale_subset.py --full subfull --small sub1m --enc /root/autodl-tmp/enc_full \
        --models all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5 \
        --sizes 2000000,4000000 --out-root /root/autodl-tmp --tag s --seed 0

The 1M corpus (`--small`) holds every judged passage plus a uniform fill, in collection order, and
the full corpus (`--full`) is the whole collection in collection order, so every 1M row is a full
row. Each intermediate corpus is the 1M corpus plus a uniform random draw of further passages, and
the draws are nested (the 2M corpus is inside the 4M corpus): growing the corpus only adds
unjudged passages, which is the comparison the scale study needs. Rows keep collection order.

For each size S this writes <out-root>/<tag><S/1e6>m/ with corpus.tsv, queries.tsv,
query_sets/ and qrels.tsv (document indices remapped), rows.npy (full-corpus row of each
document), and <out-root>/enc_<tag><S/1e6>m/<model>/doc.npy (the encoder's full-corpus vectors
at those rows) with q.npy linked to the full encoding's (the queries are identical). It refuses
to start a size whose vectors would not fit on the disk.
"""
import argparse
import os
import shutil
import sys

import numpy as np


def pids(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.split("\t", 2)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--small", required=True)
    ap.add_argument("--enc", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tag", default="s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200000)
    a = ap.parse_args()
    tokens = [x.strip() for x in a.sizes.split(",") if x.strip()]
    if not tokens:
        print("no intermediate sizes requested; nothing to do")
        return
    sizes = sorted({int(x) for x in tokens})
    if open(os.path.join(a.full, "queries.tsv"), "rb").read() != open(os.path.join(a.small, "queries.tsv"), "rb").read():
        sys.exit("queries.tsv differs between --full and --small; the corpora would not share queries")

    # pass 1: which full rows are the 1M rows (both files are in collection order)
    small = list(pids(os.path.join(a.small, "corpus.tsv")))
    n_full, j = 0, 0
    marks = []
    for i, pid in enumerate(pids(os.path.join(a.full, "corpus.tsv"))):
        if j < len(small) and pid == small[j]:
            marks.append(i); j += 1
        n_full = i + 1
    if j != len(small):
        sys.exit(f"only {j} of {len(small)} 1M passages were found in order in the full corpus")
    in_small = np.zeros(n_full, bool); in_small[np.array(marks, np.int64)] = True
    rest = np.flatnonzero(~in_small)
    extra = np.random.default_rng(a.seed).permutation(rest)
    print(f"full {n_full:,} rows; 1M rows found in order: {len(small):,}")

    # qrels of the full corpus, by full row
    fq = np.loadtxt(os.path.join(a.full, "qrels.tsv"), dtype=np.int64, ndmin=2)
    if not in_small[fq[:, 1]].all():
        sys.exit("a judged passage of the full corpus is not in the 1M corpus")

    out_rows = {}
    for S in sizes:
        if S <= len(small) or S > n_full:
            sys.exit(f"size {S} must lie between {len(small)} and {n_full}")
        rows = np.sort(np.concatenate([np.flatnonzero(in_small), extra[:S - len(small)]]))
        lab = f"{a.tag}{S / 1e6:g}m"
        d = os.path.join(a.out_root, lab)
        os.makedirs(os.path.join(d, "query_sets"), exist_ok=True)
        np.save(os.path.join(d, "rows.npy"), rows)
        shutil.copy(os.path.join(a.full, "queries.tsv"), os.path.join(d, "queries.tsv"))
        for f in os.listdir(os.path.join(a.full, "query_sets")):
            shutil.copy(os.path.join(a.full, "query_sets", f), os.path.join(d, "query_sets", f))
        new = np.searchsorted(rows, fq[:, 1])
        assert (rows[new] == fq[:, 1]).all()
        np.savetxt(os.path.join(d, "qrels.tsv"), np.column_stack([fq[:, 0], new, fq[:, 2]]), fmt="%d", delimiter="\t")
        out_rows[S] = (rows, d, lab)

    # pass 2: corpus.tsv for every size in one read of the full corpus
    keep = {S: np.zeros(n_full, bool) for S in sizes}
    for S in sizes:
        keep[S][out_rows[S][0]] = True
    fhs = {S: open(os.path.join(out_rows[S][1], "corpus.tsv"), "w", encoding="utf-8") for S in sizes}
    cnt = {S: 0 for S in sizes}
    with open(os.path.join(a.full, "corpus.tsv"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            for S in sizes:
                if keep[S][i]:
                    _, pid, text = line.rstrip("\n").split("\t", 2)
                    fhs[S].write(f"{cnt[S]}\t{pid}\t{text}\n"); cnt[S] += 1
    for S in sizes:
        fhs[S].close()
        assert cnt[S] == S, (cnt[S], S)
        print(f"{out_rows[S][2]}: corpus.tsv {S:,} rows, qrels {len(fq):,}")

    # vectors: one model at a time, refuse a size that would not fit
    for S in sizes:
        rows, d, lab = out_rows[S]
        for m in a.models:
            src = os.path.join(a.enc, m)
            X = np.load(os.path.join(src, "doc.npy"), mmap_mode="r")
            if X.shape[0] != n_full:
                sys.exit(f"{src}/doc.npy has {X.shape[0]} rows, the full corpus {n_full}")
            od = os.path.join(a.out_root, f"enc_{lab}", m); os.makedirs(od, exist_ok=True)
            path = os.path.join(od, "doc.npy")
            if os.path.exists(path) and np.load(path, mmap_mode="r").shape == (S, X.shape[1]):
                print(f"{lab} {m}: present"); continue
            need = S * X.shape[1] * 4 / 2 ** 30 + 1
            free = shutil.disk_usage(od).free / 2 ** 30
            if free < need:
                sys.exit(f"{lab} {m}: needs {need:.1f} GB, {free:.1f} GB free; stop (smaller sizes are done)")
            tmp = path + ".partial.npy"
            Y = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(S, X.shape[1]))
            for b in range(0, S, a.chunk):
                Y[b:b + a.chunk] = X[rows[b:b + a.chunk]]
            Y.flush(); del Y
            os.replace(tmp, path)
            for f in ("q.npy", "q.npy.json", "q.npy.txt"):
                if os.path.exists(os.path.join(src, f)):
                    t = os.path.join(od, f)
                    if os.path.lexists(t):
                        os.remove(t)
                    os.symlink(os.path.realpath(os.path.join(src, f)), t)
            print(f"{lab} {m}: {S:,} x {X.shape[1]} written")


if __name__ == "__main__":
    main()
