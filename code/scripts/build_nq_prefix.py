"""Build a BEIR Natural Questions subset for the P5' comparison/calibration (team script).

NOTE: none of the constructions tried so far reproduces EmbedFlow's "canonical prefix"
(MiniLM M_S 0.5875 prefix / 0.8426 random vs the registry's 0.7458). Get the real
construction from EmbedFlow's builder or its author before calling P5' a calibration.

  python scripts/build_nq_prefix.py --out subnq [--prefix 100000] [--n-queries 1000]

Inputs are fetched from the HF mirror (huggingface.co is unreachable from the GPU box):
  BeIR/nq          corpus-00000-of-00001.parquet, queries-00000-of-00001.parquet
  BeIR/nq-qrels    test.tsv

Construction (documented because the registry record pins the *sizes* but not the rule):
  * documents  = the first `--prefix` rows of the corpus parquet, in file order;
  * queries    = the first `--n-queries` rows of the queries parquet, in file order;
  * qrels      = the test qrels restricted to those queries;
  * any judged document that falls outside the prefix is appended, so judged coverage is
    100% by construction (the count is printed and recorded in meta.json).

Output is the same layout as build_subsample.py, so every downstream script works unchanged:
  corpus.tsv / queries.tsv / qrels.tsv / query_sets/*.txt / doc_len.npy / meta.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--prefix", type=int, default=100000)
ap.add_argument("--n-queries", type=int, default=1000)
ap.add_argument("--cache-dir", default="/dev/shm/nq", help="where the parquet files are staged")
ap.add_argument("--corpus-order", choices=["prefix", "random"], default="prefix",
                help="prefix = first N rows in file order; random = uniform sample of N (seeded)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--text", choices=["title_text", "text"], default="title_text",
                help="document text: title + body, or body only")
ap.add_argument("--query-select", choices=["first", "njudged"], default="first",
                help="first N queries in file order, or the N queries with the most judgments")
ap.add_argument("--outside", choices=["append", "drop"], default="append",
                help="what to do with judged documents that fall outside the prefix: append them "
                     "to the corpus (judged coverage stays 100%) or drop those qrels rows. The "
                     "registry record pins the sizes, not this rule, so both are reported.")
a = ap.parse_args()
if a.prefix <= 0 or a.n_queries <= 0:
    sys.exit("--prefix and --n-queries must be positive")


def doc_text(title, text):
    """One rule for every document's text, prefix rows and appended judged rows alike."""
    return (title or "") + " " + (text or "") if a.text == "title_text" else (text or "")

os.makedirs(os.path.join(a.out, "query_sets"), exist_ok=True)
os.makedirs(a.cache_dir, exist_ok=True)


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"cached {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.0f} MB)")
        return dest
    if os.system(f'curl -sSL -m 3600 -o "{dest}" "{url}"') != 0:
        raise SystemExit(f"download failed: {url} (otherwise this surfaces later as a "
                         f"parquet parse error on a truncated file)")
    print(f"downloaded {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.0f} MB)")
    return dest


M = "https://hf-mirror.com/datasets"
corpus_pq = fetch(f"{M}/BeIR/nq/resolve/main/corpus/corpus-00000-of-00001.parquet",
                  os.path.join(a.cache_dir, "nq-corpus.parquet"))
queries_pq = fetch(f"{M}/BeIR/nq/resolve/main/queries/queries-00000-of-00001.parquet",
                   os.path.join(a.cache_dir, "nq-queries.parquet"))
qrels_txt = fetch(f"{M}/BeIR/nq-qrels/resolve/main/test.tsv",
                  os.path.join(a.cache_dir, "nq-qrels-test.tsv"))

# ---- documents: first `prefix` rows, in file order -------------------------------
docs, order = [], []
seen = 0                      # rows scanned, for the reservoir below
_res_rng = np.random.default_rng(a.seed)
for batch in pq.ParquetFile(corpus_pq).iter_batches(batch_size=10000,
                                                     columns=["_id", "title", "text"]):
    for did, title, text in zip(batch.column("_id").to_pylist(),
                                batch.column("title").to_pylist(),
                                batch.column("text").to_pylist()):
        if a.corpus_order == "prefix" and len(docs) >= a.prefix:
            break
        body = doc_text(title, text)
        ### reservoir sampling (Algorithm R) for --corpus-order random. Collecting every row
        ### and sampling afterwards held the WHOLE corpus in memory -- BeIR/nq is about 2.7M
        ### documents of title+text, several GB -- because the break above only fires in
        ### prefix mode. The reservoir is bounded at --prefix and one pass still suffices.
        if a.corpus_order == "random":
            seen += 1
            if len(docs) < a.prefix:
                docs.append((did, body))
            else:
                j = int(_res_rng.integers(0, seen))
                if j < a.prefix:
                    docs[j] = (did, body)
            continue
        docs.append((did, body))
        order.append(did)
    if a.corpus_order == "prefix" and len(docs) >= a.prefix:
        break
if a.corpus_order == "random":
    order = [d[0] for d in docs]
    print(f"corpus docs scanned: {seen}; reservoir sample: {len(docs)} docs (seed {a.seed})")
else:
    print(f"corpus docs read: {len(docs)}")

# ---- queries: first `n_queries` rows --------------------------------------------
qid2text, qorder = {}, []
for batch in pq.ParquetFile(queries_pq).iter_batches(batch_size=1000, columns=["_id", "text"]):
    for qid, text in zip(batch.column("_id").to_pylist(), batch.column("text").to_pylist()):
        if len(qorder) >= a.n_queries:
            break
        qid2text[qid] = text
        qorder.append(qid)
    if len(qorder) >= a.n_queries:
        break
print(f"queries: {len(qorder)}")

# ---- qrels ----------------------------------------------------------------------
qrels_all = {}
with open(qrels_txt, encoding="utf-8") as f:
    next(f)
    for line in f:
        q, d, r = line.rstrip("\n").split("\t")
        if q in qid2text:
            qrels_all.setdefault(q, {})[d] = int(r)
qrels = qrels_all
if a.query_select == "njudged":
    ranked = sorted(qrels_all, key=lambda q: (-len(qrels_all[q]), qorder.index(q)))
    keep_q = set(ranked[:a.n_queries])
    qorder = [q for q in qorder if q in keep_q]
    qid2text = {q: qid2text[q] for q in qorder}
    qrels = {q: qrels_all[q] for q in qorder}
print(f"qrels rows {sum(len(v) for v in qrels.values())} over {len(qrels)} queries")
print(f"query selection: {a.query_select}")

# judged docs that fall outside the prefix
order_set = set(order)
outside = sorted({d for rel in qrels.values() for d in rel if d not in order_set})
print(f"judged docs outside the prefix: {len(outside)} -> policy '{a.outside}'")
if a.outside == "drop":
    for rel in qrels.values():
        for d in list(rel):
            if d not in order_set:
                del rel[d]
    qrels = {q: rel for q, rel in qrels.items() if rel}
    print(f"queries with at least one in-prefix judged doc: {len(qrels)}")
all_ids = order + outside
idx = {d: i for i, d in enumerate(all_ids)}
texts = dict(docs)
if outside:
    got, outside_set = set(), set(outside)
    for batch in pq.ParquetFile(corpus_pq).iter_batches(batch_size=10000,
                                                        columns=["_id", "title", "text"]):
        for did, title, text in zip(batch.column("_id").to_pylist(),
                                    batch.column("title").to_pylist(),
                                    batch.column("text").to_pylist()):
            if did in outside_set and did not in got:
                # the same text rule as the prefix rows: this line used to append title + text
                # whatever --text said (no effect on the runs made: they used the default, title_text)
                texts[did] = doc_text(title, text)
                got.add(did)
        if len(got) == len(outside):
            break
    missing = set(outside) - got
    if missing:
        raise SystemExit(f"{len(missing)} judged docs not found in the corpus at all")

with open(os.path.join(a.out, "corpus.tsv"), "w", encoding="utf-8", newline="\n") as f:
    for i, d in enumerate(all_ids):
        f.write(f"{i}\t{d}\t{texts[d].replace(chr(9), ' ').replace(chr(10), ' ').strip()}\n")
with open(os.path.join(a.out, "queries.tsv"), "w", encoding="utf-8", newline="\n") as f:
    for i, q in enumerate(qorder):
        f.write(f"{i}\t{q}\ttest\t{qid2text[q].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
with open(os.path.join(a.out, "qrels.tsv"), "w", encoding="utf-8", newline="\n") as f:
    for qi, q in enumerate(qorder):
        for d, r in sorted(qrels.get(q, {}).items()):
            if d in idx:
                f.write(f"{qi}\t{idx[d]}\t{r}\n")
np.savetxt(os.path.join(a.out, "query_sets", "test.txt"),
           np.arange(len(qorder), dtype=np.int64), fmt="%d")
np.save(os.path.join(a.out, "doc_len.npy"),
        np.array([len(texts[d]) for d in all_ids], dtype=np.int64))
### a context manager, not json.dump(..., open(...)): an exception mid-dump otherwise leaves a
### truncated, unflushed meta.json that the next step reads as a valid artifact
with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8", newline="\n") as _fh:
    json.dump({"source": "BeIR/nq (corpus+queries parquet) + BeIR/nq-qrels test.tsv",
               "prefix": a.prefix, "n_queries": a.n_queries,
               "judged_docs_outside_prefix": len(outside), "outside_policy": a.outside,
               "corpus_order": a.corpus_order, "seed": a.seed, "text_mode": a.text,
               "query_select": a.query_select,
               "queries_with_judgments": len(qrels),
               "corpus_size": len(all_ids), "queries": len(qorder),
               "qrels_rows": sum(len(v) for v in qrels.values()),
               "construction": (f"{'first' if a.corpus_order == 'prefix' else 'random (seeded)'} "
                                f"{a.prefix} corpus rows + judged docs outside them "
                                f"({a.outside}); queries: {a.query_select} {a.n_queries}")},
              _fh, indent=1)
print(f"corpus N={len(all_ids)} (prefix {a.prefix} + {len(outside)} judged), "
      f"Q={len(qorder)}, qrels rows {sum(len(v) for v in qrels.values())}")
print("wrote", a.out)
