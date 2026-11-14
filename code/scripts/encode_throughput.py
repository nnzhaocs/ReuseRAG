"""Measured encoder throughput, for two numbers the paper otherwise leaves as limitations
(RUNBOOK 14.6): cov(K') in accelerator-hours, and the inline-repair latency of a cold query.

Loads each model exactly as encode.py does (contract, apply_contract, make_encode_fn), then
  * bulk:   documents per second encoding --n passages at --batch (the background re-encoder);
  * repair: the median and p95 wall time of encoding n passages in ONE call (one batched forward
            pass), for each n in --repair, repeated --reps times. A cold window query encodes the
            unencoded part of its window before answering: the replays average 36-38 documents per
            evaluated query at K'=50 (p95 50, the whole window) and 21 at K'=25 (p95 25). n = 1 is
            the cost of one extra forward pass, which is what S1 adds (the target-model query
            encoding; a passage is an upper bound for a query).
  * the time of one background batch (--batch passages): the most an inline repair waits for the
            accelerator when the background re-encoder holds it and repairs go first at batch
            boundaries; and the adapter's per-query matrix product (S1's other cost).
Nothing is written but the JSON; no vectors are kept.

  python scripts/encode_throughput.py --tsv sub1m/corpus.tsv --max-seq-len 256 \\
      --model BAAI/bge-base-en-v1.5 --model intfloat/e5-base-v2 --out throughput.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap import models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", action="append", required=True, help="model id in configs/models.json (repeatable)")
ap.add_argument("--tsv", required=True, help="corpus.tsv (text is the last column)")
ap.add_argument("--max-seq-len", type=int, required=True)
ap.add_argument("--n", type=int, default=20000, help="passages for the bulk measurement")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--repair", default="1,25,37,50", help="documents per inline repair, comma-separated")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--fp16", action="store_true")
ap.add_argument("--device", default=None)
ap.add_argument("--contracts", default=models.DEFAULT_PATH)
ap.add_argument("--out", required=True)
a = ap.parse_args()
a.repairs = sorted({int(x) for x in str(a.repair).split(",") if x.strip()})
if not a.repairs or min(a.repairs) < 1:
    sys.exit(f"--repair needs positive sizes, got {a.repair!r}")

texts = []
with open(a.tsv, encoding="utf-8") as fh:
    for line in fh:
        texts.append(line.rstrip("\n").split("\t")[-1])
        if len(texts) >= a.n:
            break
if len(texts) <= max(a.batch, max(a.repairs)):
    sys.exit(f"{a.tsv}: only {len(texts)} passages; need more than {max(a.batch, max(a.repairs))}")

try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception:  # torch missing or no device: record that, the timings still run
    torch, gpu = None, None


def machine_state():
    """CPU load and what else is on the GPU: a repair timed beside another job is not a quiet
    measurement, so every run records the conditions it was taken under."""
    import subprocess
    st = {"loadavg": list(os.getloadavg()), "cpus": os.cpu_count()}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        st["gpu"] = (q.stdout or "").strip(); st["gpu_apps"] = [l for l in (apps.stdout or "").strip().splitlines() if l]
    except Exception as e:  # no nvidia-smi: record that
        st["gpu"] = f"unavailable ({type(e).__name__})"
    return st


def sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


out = {"tsv": os.path.abspath(a.tsv), "max_seq_len": a.max_seq_len, "fp16": a.fp16, "gpu": gpu,
       "n": len(texts), "batch": a.batch, "repair_sizes": a.repairs, "reps": a.reps,
       "env": models.env_versions(), "machine_before": machine_state(), "models": {}}
print("machine before:", out["machine_before"])
for mid in a.model:
    c = models.get_contract(mid, a.contracts)
    st = models.load_model(c, a.device, a.fp16)
    findings = models.apply_contract(st, c, a.max_seq_len)
    fn = models.make_encode_fn(st, c, "doc")
    fn(texts[:a.batch], a.batch)                     # warm-up: kernels, caches, lazy init
    sync()
    t = time.perf_counter()
    emb = np.asarray(fn(texts, a.batch))
    sync()
    bulk = time.perf_counter() - t
    rng = np.random.default_rng(0)
    repair = {}
    for n_rep in a.repairs:
        span = len(texts) - n_rep
        lat = []
        for _ in range(a.reps):
            i = int(rng.integers(0, span))
            sync()
            t = time.perf_counter()
            fn(texts[i:i + n_rep], n_rep)
            sync()
            lat.append(time.perf_counter() - t)
        lat = np.asarray(lat) * 1000
        repair[str(n_rep)] = {"ms_median": float(np.median(lat)), "ms_p95": float(np.percentile(lat, 95)),
                              "ms_p99": float(np.percentile(lat, 99)), "ms_max": float(lat.max())}
    # S1's other per-query cost: the adapter, one (d_target x d_source) matrix product on the CPU,
    # timed at the widest source dimension in the study (1024, bge-m3)
    W = rng.standard_normal((emb.shape[1], 1024)).astype(np.float32); qv = emb[:1].astype(np.float32)
    lat_ad = []
    for _ in range(max(a.reps, 200)):
        t = time.perf_counter(); _ = qv @ W; lat_ad.append(time.perf_counter() - t)
    rec = {"contract_status": c["status"], "findings": findings, "dim": int(emb.shape[1]),
           "bulk_seconds": bulk, "docs_per_second": len(texts) / bulk,
           "seconds_per_million_docs": 1e6 * bulk / len(texts), "repair": repair,
           # the longest an inline repair waits for the accelerator if the background re-encoder holds
           # it and repairs take priority at batch boundaries: one background batch
           "background_batch_ms": 1000 * bulk / max(1, -(-len(texts) // a.batch)),
           "adapter_us_median": float(np.median(lat_ad) * 1e6)}
    out["models"][mid] = rec
    print(f"{mid}: {rec['docs_per_second']:.0f} docs/s ({rec['seconds_per_million_docs'] / 3600:.2f} h per 10^6); inline "
          f"repair in one call: " + ", ".join(f"{n} docs {r['ms_median']:.1f} ms (p95 {r['ms_p95']:.1f})"
                                              for n, r in repair.items()))
    del st, fn
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

out["machine_after"] = machine_state()
with open(a.out, "w") as fh:
    json.dump(out, fh, indent=1)
print("wrote", a.out)
