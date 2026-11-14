"""Build an MS MARCO passage subsample that KEEPS EVERY JUDGED PASSAGE, plus random fill.

A uniform random subsample drops most judged passages and makes M_T and G(K')
meaningless. UNTESTED here (no MS MARCO in the dev sandbox) -- run the sanity
printout at the end and check the judged-coverage line says 100%.

Inputs
  --collection collection.tsv            "pid<TAB>text"
  --qrels FILE:TAG ...                   TREC-style "qid 0 pid rel" (MS MARCO dev qrels use
                                         the same 4 columns); TAG names the query set
  --queries FILE:TAG ...                 "qid<TAB>text"; TAG must match a qrels TAG
Outputs (in --out)
  corpus.tsv    "docidx<TAB>pid<TAB>text"
  queries.tsv   "qidx<TAB>qid<TAB>tag<TAB>text"
  qrels.tsv     "qidx<TAB>docidx<TAB>rel"      (reusemap format)
  query_sets/<TAG>.txt   query indices per set (e.g. trecdl -> secondary_eval_qids)
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--collection", required=True)
ap.add_argument("--qrels", nargs="+", required=True)
ap.add_argument("--queries", nargs="+", required=True)
ap.add_argument("--N", type=int, default=1_000_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "query_sets"), exist_ok=True)

qrels = []  # (tag, qid, pid, rel)
for spec in a.qrels:
    path, tag = spec.rsplit(":", 1)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                qrels.append((tag, parts[0], parts[2], int(parts[3])))
judged = {pid for _, _, pid, _ in qrels}
print(f"judged passages: {len(judged)}")

# pass 1: count lines
with open(a.collection, encoding="utf-8") as f:
    n_total = sum(1 for _ in f)
print(f"collection lines: {n_total}")
if a.N < len(judged):
    raise SystemExit("N smaller than the judged set")

# pass 2: decide which non-judged lines to keep (uniform), keep all judged
rng = np.random.default_rng(a.seed)
need = a.N - len(judged)
is_judged_line = np.zeros(n_total, bool)
with open(a.collection, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if line.split("\t", 1)[0] in judged:
            is_judged_line[i] = True
pool = np.flatnonzero(~is_judged_line)
if need > len(pool):
    raise SystemExit(f"N={a.N} needs {need} non-judged lines to fill out {len(judged)} judged "
                     f"ones, but the collection has only {len(pool)}. Lower --N, or use the "
                     f"whole collection if {n_total} is close enough to what you wanted.")

def flat(s):
    """Collapse tabs, CR and LF to spaces.

    corpus.tsv is `idx TAB pid TAB text` and queries.tsv `idx TAB qid TAB tag TAB text`, and
    encode.py reads the text as split("\t")[-1]. So one tab inside a passage silently drops
    everything before it from what gets encoded -- wrong vectors, with every row count still
    matching and no check anywhere catching it. build_code_corpus.py and build_nq_prefix.py
    already sanitise; this is the same guard for the MS MARCO path.
    """
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()

keep = np.zeros(n_total, bool)
keep[is_judged_line] = True
keep[rng.choice(pool, size=need, replace=False)] = True

### A duplicate pid in the collection is refused, not tolerated. `pid2idx[pid] = len(pid2idx)`
### re-assigns an existing key without growing the dict, so a single duplicate writes two rows
### that share one index and then shifts every later pid's index down by one. Nothing downstream
### notices: the row written for the duplicate and the row for the NEXT pid both carry that
### index, so a qrel resolves to whichever comes first, and the judged-coverage self-check below
### still prints 100%. That is silently wrong M_T and G, which is worse than any crash.
pid2idx = {}
rows_written = 0
with open(a.collection, encoding="utf-8") as f, \
        open(os.path.join(a.out, "corpus.tsv"), "w", encoding="utf-8") as out:
    for i, line in enumerate(f):
        if keep[i]:
            if "\t" not in line:
                raise SystemExit(f"{a.collection} line {i + 1}: no tab; expected pid<TAB>text")
            pid, text = line.rstrip("\n").split("\t", 1)
            text = flat(text)
            if pid in pid2idx:
                raise SystemExit(
                    f"{a.collection} line {i + 1}: duplicate pid {pid!r} (first seen at index "
                    f"{pid2idx[pid]}). Two rows would share one document index and every later "
                    f"pid would shift down by one, so qrels would point at the wrong passages "
                    f"while 'judged coverage' still read 100%. Deduplicate the collection first.")
            pid2idx[pid] = len(pid2idx)
            out.write(f"{pid2idx[pid]}\t{pid}\t{text}\n")
            rows_written += 1
### belt and braces: the index column must be exactly 0..n-1, one row each, or the vectors
### encode.py produces will not line up with the ids qrels uses
assert rows_written == len(pid2idx) == sum(keep), (rows_written, len(pid2idx), int(sum(keep)))
assert set(pid2idx.values()) == set(range(rows_written)), "document indices are not contiguous"

qid2idx, sets = {}, {}
with open(os.path.join(a.out, "queries.tsv"), "w", encoding="utf-8") as out:
    for spec in a.queries:
        path, tag = spec.rsplit(":", 1)
        judged_q = {q for t, q, _, _ in qrels if t == tag}
        with open(path, encoding="utf-8") as f:
            for n_, line in enumerate(f):
                if "\t" not in line:
                    raise SystemExit(f"{path} line {n_ + 1}: no tab; expected qid<TAB>text")
                qid, text = line.rstrip("\n").split("\t", 1)
                text = flat(text)
                if qid not in judged_q or (tag, qid) in qid2idx:
                    continue
                qid2idx[(tag, qid)] = len(qid2idx)
                sets.setdefault(tag, []).append(qid2idx[(tag, qid)])
                out.write(f"{qid2idx[(tag, qid)]}\t{qid}\t{tag}\t{text}\n")

missing = 0
with open(os.path.join(a.out, "qrels.tsv"), "w", encoding="utf-8") as out:
    for tag, qid, pid, rel in qrels:
        if (tag, qid) not in qid2idx:
            continue
        if pid not in pid2idx:
            missing += 1
            continue
        out.write(f"{qid2idx[(tag, qid)]}\t{pid2idx[pid]}\t{rel}\n")
for tag, idxs in sets.items():
    np.savetxt(os.path.join(a.out, "query_sets", f"{tag}.txt"), np.array(idxs), fmt="%d")
print(f"corpus N={len(pid2idx)}  queries={len(qid2idx)}  sets={ {t: len(v) for t, v in sets.items()} }")
print(f"judged coverage: {100 * (1 - missing / max(len(qrels), 1)):.2f}%  (must be 100%; "
      f"<100% means judged pids are absent from the collection file)")
