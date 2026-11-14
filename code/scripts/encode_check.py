"""Model-free checks on one encoding directory, against the data it was built from.

  python scripts/encode_check.py --enc enc/bge-base --data sub1m
  python scripts/encode_check.py --enc enc/e5-base --data sub1m --json checks/e5-base.json

Exit code 0 = every hard check passed, 1 = at least one failed, 2 = usage error.

Why this exists. Four encodings in a row were wrong in four different ways, and only one of them
was caught by a check:
  * bge-small was mean-pooled because the library ignored the old pooling flags  -> caught by
    the contract check;
  * e5-base scored 0.000 because its query rows were misaligned with the qrels  -> NOT caught:
    model_quality.py silently evaluated only the queries with `qid < len(q.npy)`;
  * codesage-base's download directory was 3.3 MB, so the weights were never there -> NOT
    caught: nothing compared the vectors against anything;
  * bge-m3 was simply never re-encoded                                           -> NOT caught.
Everything here is cheap, needs no model and no GPU, and fails loudly. `encode.py --self-check`
covers the one thing that does need the model (document text through the query path).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap.pair import load_qrels  # noqa: E402
from reusemap import models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--enc", required=True, help="directory holding doc.npy and q.npy")
ap.add_argument("--data", required=True, help="build_subsample.py output (corpus.tsv, queries.tsv, qrels.tsv)")
ap.add_argument("--sample", type=int, default=4096, help="rows sampled for the collapse/duplicate checks")
ap.add_argument("--collapse-above", type=float, default=0.95,
                help="fail if the mean pairwise cosine of the sample exceeds this (collapsed space)")
ap.add_argument("--min-bytes-per-row", type=int, default=64, help="sanity floor on dim * 4")
ap.add_argument("--json", default=None)
a = ap.parse_args()

checks = []


def add(ok, hard, name, detail):
    checks.append({"check": name, "ok": bool(ok), "hard": bool(hard), "detail": detail})


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


for f in ("corpus.tsv", "queries.tsv", "qrels.tsv"):
    if not os.path.exists(os.path.join(a.data, f)):
        sys.exit(f"{a.data}/{f} is missing: --data must be a build_subsample.py output directory")
n_docs_tsv = count_lines(os.path.join(a.data, "corpus.tsv"))
n_q_tsv = count_lines(os.path.join(a.data, "queries.tsv"))
qrels = load_qrels(os.path.join(a.data, "qrels.tsv"))

arrays = {}
for kind, fn, n_tsv in (("doc", "doc.npy", n_docs_tsv), ("query", "q.npy", n_q_tsv)):
    path = os.path.join(a.enc, fn)
    if not os.path.exists(path):
        add(False, True, f"{kind}: file exists", f"{path} is missing")
        continue
    X = np.load(path, mmap_mode="r")
    arrays[kind] = X
    add(X.ndim == 2, True, f"{kind}: 2-D array", f"shape {tuple(X.shape)}")
    if X.ndim != 2:
        continue
    rows, dim = X.shape
    add(rows == n_tsv, True, f"{kind}: row count matches the text file",
        f"{rows} rows vs {n_tsv} lines in {'corpus.tsv' if kind == 'doc' else 'queries.tsv'}")
    add(dim * 4 >= a.min_bytes_per_row, True, f"{kind}: plausible dimension", f"dim {dim}")
    add(X.dtype == np.float32, False, f"{kind}: float32", str(X.dtype))
    rng = np.random.default_rng(0)
    if rows == 0:
        add(False, True, f"{kind}: array is non-empty",
            f"{path} has 0 rows; every later check on it is vacuous. Re-run encode.py.")
        continue
    take = np.sort(rng.choice(rows, min(a.sample, rows), replace=False))
    S = np.asarray(X[take], dtype=np.float32)
    norms = np.linalg.norm(S, axis=1)
    add(np.isfinite(S).all(), True, f"{kind}: all values finite",
        f"{int((~np.isfinite(S)).sum())} non-finite in a sample of {len(take)}")
    add((norms > 1e-6).all(), True, f"{kind}: no zero-norm rows",
        f"{int((norms <= 1e-6).sum())} zero rows in a sample of {len(take)}")
    U = S / np.maximum(norms[:, None], 1e-12)
    m = min(len(U), 1024)
    C = U[:m] @ U[:m].T
    off = C[~np.eye(m, dtype=bool)]
    if off.size:
        mean_cos = float(off.mean())
        add(not (mean_cos > a.collapse_above), True, f"{kind}: space not collapsed",
            f"mean pairwise cosine {mean_cos:.4f} over {m} rows (fail above {a.collapse_above})")
    else:   # one row: nothing to compare, so the check cannot pass
        add(False, True, f"{kind}: space not collapsed", f"only {m} row(s): cannot estimate collapse")
    dup = int(np.unique(np.round(U[:m], 5), axis=0).shape[0])
    add(dup >= 0.9 * m, False, f"{kind}: few duplicate rows",
        f"{m - dup} of {m} sampled rows are duplicates at 5 decimal places")
    side = path + ".json"
    if os.path.exists(side):
        rec = json.load(open(side))
        pad = (rec.get("padding_check") or {}).get("status")
        add(pad != "fail", True, f"{kind}: padding check in the sidecar",
            f"status {pad}, contract {rec.get('contract_status')}")
        add(rec.get("contract_status") == "verified", False, f"{kind}: contract verified",
            f"status {rec.get('contract_status')!r}")
        try:
            cur = models.get_contract(rec["model_id"]) if rec.get("model_id") else None
        except Exception:   # model no longer listed: nothing to compare against
            cur = None
        if cur is not None:
            stale = models.stale_encoding(rec, cur, kind)
            add(not stale, True, f"{kind}: encoded under the current contract",
                "; ".join(stale) or "prefix and prompt name match configs/models.json")
        sr = (rec.get("self_retrieval") or {}).get("status")
        if kind == "doc":
            add(sr != "fail", True, "doc: self-retrieval check in the sidecar",
                f"status {sr}, top1 {(rec.get('self_retrieval') or {}).get('top1_rate')}")
        shp = rec.get("shape")
        add(not shp or list(shp) == list(X.shape), True, f"{kind}: sidecar shape matches the file",
            f"sidecar {shp} vs file {list(X.shape)}")
    else:
        add(False, False, f"{kind}: sidecar present", f"{side} is missing (encoded outside encode.py?)")

if "query" in arrays and arrays["query"].ndim == 2:
    nq = arrays["query"].shape[0]
    bad_q = sorted(q for q in qrels if q >= nq)
    add(not bad_q, True, "qrels: every query id has a vector",
        f"{len(bad_q)} of {len(qrels)} query ids are >= {nq} rows"
        + (f" (e.g. {bad_q[:3]})" if bad_q else "")
        + ". This is the e5-base failure: model_quality.py would silently score only the rest")
if "doc" in arrays and arrays["doc"].ndim == 2:
    nd = arrays["doc"].shape[0]
    bad_d = sorted({d for rels in qrels.values() for d in rels if d >= nd})
    add(not bad_d, True, "qrels: every judged document has a vector",
        f"{len(bad_d)} judged document ids are >= {nd} rows"
        + (f" (e.g. {bad_d[:3]})" if bad_d else ""))

hard_fail = [c for c in checks if c["hard"] and not c["ok"]]
soft_fail = [c for c in checks if not c["hard"] and not c["ok"]]
width = max(len(c["check"]) for c in checks)
for c in checks:
    mark = "ok  " if c["ok"] else ("FAIL" if c["hard"] else "warn")
    print(f"{mark}  {c['check']:{width}s}  {c['detail']}")
print(f"\n{len(checks) - len(hard_fail) - len(soft_fail)} passed, {len(soft_fail)} warning(s), "
      f"{len(hard_fail)} failure(s)")
if a.json:
    os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
    with open(a.json, "w") as f:
        json.dump({"enc": os.path.abspath(a.enc), "data": os.path.abspath(a.data),
                   "checks": checks, "hard_failures": len(hard_fail)}, f, indent=1)
if hard_fail:
    sys.exit(1)
