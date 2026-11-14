"""Encode corpus or queries under an explicit per-model contract (configs/models.json).

  python scripts/encode.py --model BAAI/bge-base-en-v1.5 --kind doc   --tsv sub1m/corpus.tsv  --out enc/bge-base/doc.npy --max-seq-len 256
  python scripts/encode.py --model BAAI/bge-base-en-v1.5 --kind query --tsv sub1m/queries.tsv --out enc/bge-base/q.npy  --max-seq-len 256

What it does, in order:
  1. refuses a model with no contract, and a 'blocked' one unless --allow-blocked;
  2. loads the model at the contract's revision, enforces padding side / EOS, checks pooling
     and prompt names against the loaded model (a mismatch it cannot fix is an error);
  3. padding check: the first --padding-n texts encoded batched vs one at a time; aborts if
     they disagree (padding leaking into embeddings), unless --ignore-padding-fail;
  4. encodes, writes <out> (float32, unit norm) and <out>.json with the contract, findings,
     padding check and library versions (plus a short <out>.txt for humans).
Use the SAME --max-seq-len for both models of a pair (min of their caps).
"""
import argparse
import time
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap import models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="exact model id as listed in configs/models.json")
ap.add_argument("--kind", choices=["doc", "query"], required=True)
ap.add_argument("--tsv", required=True, help="corpus.tsv or queries.tsv (text is the last column)")
ap.add_argument("--out", required=True)
ap.add_argument("--max-seq-len", type=int, required=True)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--fp16", action="store_true")
ap.add_argument("--device", default=None)
ap.add_argument("--contracts", default=models.DEFAULT_PATH)
ap.add_argument("--allow-blocked", action="store_true", help="diagnostic runs only")
ap.add_argument("--override", default=None,
                help="JSON object overriding contract fields for an experiment; recorded in the sidecar")
ap.add_argument("--padding-n", type=int, default=64)
ap.add_argument("--ignore-padding-fail", action="store_true")
ap.add_argument("--self-check", type=int, default=256, metavar="N",
                help="--kind doc only: encode N document texts through the QUERY path and check "
                     "each retrieves its own document vector (0 disables). This is the only "
                     "check that sees a model whose weights never downloaded or whose query side "
                     "is pooled wrongly; the contract and padding checks cannot.")
ap.add_argument("--self-check-pool", type=int, default=10000)
ap.add_argument("--ignore-self-check-fail", action="store_true")
ap.add_argument("--chunk", type=int, default=0, metavar="N",
                help="encode N texts at a time straight into the output .npy (memory-mapped), so peak "
                     "RAM is one chunk, not the corpus: 8.8M x 768 float32 is 27 GB, and holding it "
                     "plus the encoder's own copy and the normalised copy needs ~80 GB. 0 (default) "
                     "keeps the one-call path every earlier encoding used. The vectors are the same "
                     "either way (each text is encoded independently; the padding check guards this).")
a = ap.parse_args()

c = models.get_contract(a.model, a.contracts, a.allow_blocked,
                        json.loads(a.override) if a.override else None)
if c["status"] != "verified":
    print(f"WARNING: contract for {a.model} is {c['status']!r} ({c['source']}); the sidecar records this")
texts = [line.rstrip("\n").split("\t")[-1] for line in open(a.tsv, encoding="utf-8")]
if not texts:
    sys.exit(f"{a.tsv}: no lines to encode")

st = models.load_model(c, a.device, a.fp16)
findings = models.apply_contract(st, c, a.max_seq_len)
for f in findings:
    print("contract:", f)
fn = models.make_encode_fn(st, c, a.kind)
# the exact text the model sees for the first input: the model's own named prompt, then the
# contract's literal prefix, then the text. Printed and recorded so an instruction sent twice
# (or not at all) is visible in the sidecar, not only in a quality number.
_pn = models.prompt_name_for(c, a.kind)
effective = ((getattr(st, "prompts", None) or {}).get(_pn, "") if _pn else "") + models.prefix_for(c, a.kind) + (texts[0] if texts else "")
print("model input for the first text:", repr(effective[:300]))
pad = models.padding_check(fn, texts, n=a.padding_n)
print(f"padding check: min cos {pad['min_cos']} over {pad['n']} texts -> {pad['status']}")
if pad["status"] == "fail" and not a.ignore_padding_fail:
    sys.exit("ABORT: batched and one-at-a-time encodings disagree; padding/masking/pooling is "
             "wrong for this model in this environment. Fix the contract or environment first.")

if a.chunk and a.chunk < len(texts):
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".partial.npy"   # renamed into place only once complete, so a crash leaves no .npy
    first = np.asarray(fn(texts[:min(a.chunk, len(texts))], a.batch), dtype=np.float32)
    emb = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(texts), first.shape[1]))
    t0 = time.time()
    for i in range(0, len(texts), a.chunk):
        e = first if i == 0 else np.asarray(fn(texts[i:i + a.chunk], a.batch), dtype=np.float32)
        emb[i:i + len(e)] = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        done = i + len(e)
        print(f"  {done:,}/{len(texts):,} encoded, {done / max(time.time() - t0, 1e-9):.0f} texts/s", flush=True)
    emb.flush(); del first
else:
    emb = np.asarray(fn(texts, a.batch), dtype=np.float32)
    ### not `emb /= ...`: np.asarray returns the SAME object when the input is already a float32
    ### array, and some encoder paths hand back a read-only view, which an in-place divide rejects
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

self_chk = {"status": "skipped", "reason": "queries, or --self-check 0"}
if a.kind == "doc" and a.self_check > 0:
    rng = np.random.default_rng(0)
    n = min(a.self_check, len(texts))
    idx = rng.choice(len(texts), n, replace=False)
    pool = rng.choice(len(texts), min(a.self_check_pool, len(texts)), replace=False)
    self_chk = models.self_retrieval_check(emb, models.make_encode_fn(st, c, "query"), texts,
                                           idx, pool, batch=min(a.batch, 64))
    print(f"self-retrieval check: top1 {self_chk['top1_rate']} over {self_chk['n']} documents in a "
          f"pool of {self_chk['pool']} (chance {self_chk['chance']:.2e}) -> {self_chk['status']}")
    if self_chk["status"] == "fail" and not a.ignore_self_check_fail:
        sys.exit("ABORT: document texts put through the query path do not retrieve their own "
                 "vectors. The weights, the pooling, the prefixes or the environment are wrong "
                 "for this model; a quality number from these vectors would be meaningless.")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
if isinstance(emb, np.memmap):
    del emb
    os.replace(a.out + ".partial.npy", a.out)
    emb = np.load(a.out, mmap_mode="r")
else:
    np.save(a.out, emb)
rec = models.encoding_record(c, a.kind, a.max_seq_len, findings, pad,
                             {"tsv": os.path.abspath(a.tsv), "shape": list(emb.shape),
                              "fp16": a.fp16, "n_texts": len(texts), "self_retrieval": self_chk,
                              "batch": a.batch, "model_input_example": effective[:300]})
with open(a.out + ".json", "w") as f:
    json.dump(rec, f, indent=1)
with open(a.out + ".txt", "w") as f:
    f.write(f"model={a.model}\nkind={a.kind}\nstatus={c['status']}\nprefix={rec['text_prefix']!r}\n"
            f"prompt_name={rec['prompt_name']}\nmax_seq_len={a.max_seq_len}\nfp16={a.fp16}\n"
            f"padding_check={pad['status']} min_cos={pad['min_cos']}\n"
            f"self_retrieval={self_chk['status']} top1={self_chk.get('top1_rate')}\n"
            f"shape={emb.shape}\n")
print("wrote", a.out, emb.shape)
