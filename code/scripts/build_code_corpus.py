"""Build a CodeSearchNet-Python IR corpus for the code pairs P7/P8 (RUNBOOK 3a).

  python scripts/build_code_corpus.py \
      --parquet csn/python/test-00000-of-00001.parquet csn/python/validation-00000-of-00001.parquet \
      --cosqa cosqa/cosqa-dev.json --out subcsn

Documents are function bodies (`func_code_string`), queries are their docstrings
(`func_documentation_string`); a query's relevant document is the function it was written for.
That is the natural IR reading of CodeSearchNet and it keeps every (comment, code) pair the
dataset ships, so judged coverage is 100% by construction.

CoSQA dev is appended as a second query tag: its `doc` is the query and its `code` the labelled
positive, so those snippets join the corpus with a qrel of 1.

Outputs (same layout as build_subsample.py, so every downstream script works unchanged):
  corpus.tsv   docidx<TAB>pid<TAB>text
  queries.tsv  qidx<TAB>qid<TAB>tag<TAB>text
  qrels.tsv    qidx<TAB>docidx<TAB>rel
  query_sets/<tag>.txt   query indices per tag

Code contains tabs and newlines, either of which would break the TSV readers (they split on
tab for columns and on newline for rows), so newlines are collapsed to single spaces. BM25
tokenizes on whitespace and the BERT-family tokenizers treat the two identically, so nothing
measurable is lost.
"""
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", nargs="+", required=True, help="code_search_net python parquet files")
ap.add_argument("--cosqa", default=None, help="cosqa-dev.json (optional, adds a second query tag)")
ap.add_argument("--out", required=True)
ap.add_argument("--min-query-chars", type=int, default=10)
ap.add_argument("--max-doc-chars", type=int, default=40000)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "query_sets"), exist_ok=True)


def flat(s):
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()




doc_index = {}       # text -> row index in corpus.tsv (exact string, not a hash: the
                     # sha() helper this once used was removed, since an exact key cannot
                     # collide and the corpus is small enough to hold)
doc_texts = []       # row index -> text
doc_pids = []        # row index -> pid
query_id = {}        # query text -> qid
query_text = []      # qid -> text
query_tag = []       # qid -> tag
qrels = {}           # qid -> set(doc row)


def add_doc(text, pid):
    if text in doc_index:
        return doc_index[text]
    doc_index[text] = len(doc_texts)
    doc_texts.append(text)
    doc_pids.append(pid)
    return len(doc_texts) - 1


def add_query(text, tag, doc):
    # A query text is registered once, under the first tag it appears with; its qrels collect
    # every source's documents. The built corpus has no text shared between CSN and CoSQA, and
    # the corpus is fixed (changing the key would renumber every query), so this stays as is.
    if text not in query_id:
        query_id[text] = len(query_text)
        query_text.append(text)
        query_tag.append(tag)
    qi = query_id[text]
    qrels.setdefault(qi, set()).add(doc)
    return qi


n_rows = n_nocode = n_noquery = 0
for path in a.parquet:
    t = pq.read_table(path, columns=["func_code_string", "func_documentation_string",
                                     "func_path_in_repository"])
    for code, doc, pid in zip(t.column("func_code_string").to_pylist(),
                              t.column("func_documentation_string").to_pylist(),
                              t.column("func_path_in_repository").to_pylist()):
        n_rows += 1
        if not code or not code.strip():
            n_nocode += 1
            continue
        di = add_doc(flat(code)[:a.max_doc_chars], pid or f"csn-{len(doc_texts)}")
        if not doc or len(doc.strip()) < a.min_query_chars:
            n_noquery += 1
            continue
        add_query(flat(doc), "csn", di)

if a.cosqa:
    n_cosqa = 0
    for row in json.load(open(a.cosqa, encoding="utf-8")):
        code, query = flat(row.get("code") or ""), flat(row.get("doc") or "")
        if not code or len(query) < a.min_query_chars:
            continue
        di = add_doc(code[:a.max_doc_chars], f"cosqa-{row.get('idx', n_cosqa)}")
        add_query(query, "cosqa", di)
        n_cosqa += 1
    print(f"cosqa rows used: {n_cosqa}")

with open(os.path.join(a.out, "corpus.tsv"), "w", encoding="utf-8", newline="\n") as f:
    for i, (pid, text) in enumerate(zip(doc_pids, doc_texts)):
        f.write(f"{i}\t{pid}\t{text}\n")
with open(os.path.join(a.out, "queries.tsv"), "w", encoding="utf-8", newline="\n") as f:
    for qi, (text, tag) in enumerate(zip(query_text, query_tag)):
        f.write(f"{qi}\t{tag}-{qi}\t{tag}\t{text}\n")
with open(os.path.join(a.out, "qrels.tsv"), "w", encoding="utf-8", newline="\n") as f:
    for qi in sorted(qrels):
        for d in sorted(qrels[qi]):
            f.write(f"{qi}\t{d}\t1\n")
for tag in sorted(set(query_tag)):
    idx = [qi for qi, t in enumerate(query_tag) if t == tag]
    np.savetxt(os.path.join(a.out, "query_sets", f"{tag}.txt"),
               np.array(idx, dtype=np.int64), fmt="%d")
np.save(os.path.join(a.out, "doc_len.npy"),
        np.array([len(t) for t in doc_texts], dtype=np.int64))

print(f"rows read {n_rows} | skipped (no code) {n_nocode} | (no docstring) {n_noquery}")
print(f"corpus N={len(doc_texts)}  queries Q={len(query_text)}  "
      f"tags={ {t: query_tag.count(t) for t in set(query_tag)} }")
print(f"qrels rows {sum(len(v) for v in qrels.values())} over {len(qrels)} judged queries")
print(f"doc chars: median {int(np.median([len(t) for t in doc_texts]))} "
      f"p95 {int(np.percentile([len(t) for t in doc_texts], 95))}")
print("wrote", a.out)
