"""Retrieval quality of one or more models on the same corpus/queries/qrels (exact search).

Used as a pre-check before fixing a pair: the TARGET of a migration pair must be the
better model on that corpus, otherwise the pair is a downgrade. Run it on a small subset
(e.g. one CodeSearchNet language split) before encoding the full corpus.

  python scripts/model_quality.py --qrels sub/qrels.tsv \\
      --model jina=enc/jina-code --model codesage-small=enc/codesage-small --model codesage-base=enc/codesage-base
Each encoding dir holds doc.npy and q.npy (as written by encode.py). The sidecars written
by encode.py (q.npy.json / doc.npy.json) are read so the table shows each model's contract
status, query prefix/prompt and padding check -- the setup a reviewer needs to reproduce it.
--floor NAME flags every model scoring below that model (e.g. MiniLM-L6: an encoder of a
stronger family landing below it is almost always an encoding bug, not a finding).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from reusemap.pair import load_qrels  # noqa: E402
from reusemap.search import normalize_rows, topk_ip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--qrels", required=True)
ap.add_argument("--model", action="append", required=True, help="name=encoding_dir")
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--thr", type=int, default=1)
ap.add_argument("--floor", default=None, help="name of a model that every other model should beat")
ap.add_argument("--allow-partial", action="store_true",
                help="score only the queries that have a vector instead of refusing. Diagnostic "
                     "only: a shortfall means the rows are misaligned, so the score is of the "
                     "wrong queries, not of fewer queries")
a = ap.parse_args()

qrels = load_qrels(a.qrels)
rows = []
for spec in a.model:
    name, d = spec.split("=", 1)
    X = normalize_rows(np.load(os.path.join(d, "doc.npy")))
    Q = normalize_rows(np.load(os.path.join(d, "q.npy")))
    missing = sorted(q for q in qrels if q >= len(Q))
    if missing and not a.allow_partial:
        sys.exit(f"{name}: {len(missing)} of {len(qrels)} judged query ids have no vector in "
                 f"{d}/q.npy ({len(Q)} rows; e.g. {missing[:3]}). Silently scoring the rest is "
                 f"how e5-base reported 0.000 on 6,980 of 7,077 queries - the rows are "
                 f"misaligned, so re-encode it under its contract (or --allow-partial to look).")
    qs = np.array(sorted(q for q in qrels if q < len(Q)))
    if len(qs) == 0:
        sys.exit(f"{name}: not one judged query has a vector in {d}/q.npy ({len(Q)} rows vs "
                 f"judged ids up to {max(qrels)}). There is nothing to score, not even "
                 f"partially; --allow-partial cannot rescue an empty set. The query file is "
                 f"from a different run -- re-encode it under its contract.")
    top, _ = topk_ip(Q[qs], X, a.k)
    r = [recall_at_k(top[i], qrels[q], a.k, a.thr) for i, q in enumerate(qs)]
    n = [ndcg_at_k(top[i], qrels[q], a.k) for i, q in enumerate(qs)]
    r = [v for v in r if v is not None]
    n = [v for v in n if v is not None]
    side = {}
    for kind, fn in (("q", "q.npy.json"), ("d", "doc.npy.json")):
        path = os.path.join(d, fn)
        side[kind] = json.load(open(path)) if os.path.exists(path) else {}
    rows.append((name, np.mean(r), np.mean(n), len(r), side))


def _desc(sd):
    if not sd:
        return "no sidecar", "?", "?"
    pre = sd.get("prompt_name") and f"prompt:{sd['prompt_name']}" or repr(sd.get("text_prefix", ""))[:28]
    pc = (sd.get("padding_check") or {})
    pcs = f"{pc.get('status')}({pc.get('min_cos') and round(pc['min_cos'], 4)})"
    return sd.get("contract_status", "?"), pre, pcs


print(f"{'model':24s} {'Recall@'+str(a.k):>10s} {'nDCG@'+str(a.k):>9s} {'queries':>8s}  "
      f"{'contract':10s} {'query prefix/prompt':30s} {'q padding':14s} {'d padding':14s}")
floor = next((r for r in rows if r[0] == a.floor), None) if a.floor else None
for name, r, n, c, side in sorted(rows, key=lambda x: -x[1]):
    st, pre, qpc = _desc(side["q"])
    _, _, dpc = _desc(side["d"])
    flag = ""
    if floor and name != floor[0] and r < floor[1]:
        flag = f"  <-- below floor model {floor[0]}: check this encoding before using it"
    print(f"{name:24s} {r:10.4f} {n:9.4f} {c:8d}  {st:10s} {pre:30s} {qpc:14s} {dpc:14s}{flag}")
print("\nA migration pair must go from a worse model to a better one on this corpus.")
