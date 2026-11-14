"""Golden-example check: reproduce the worked example on a model's card.

Most embedding model cards print the similarity scores their sample queries and documents
should produce. If this stack reproduces them, the prefixes/prompts, pooling, padding and
library versions are right; if not, every number encoded with it is suspect.

  python scripts/golden_check.py configs/golden/Qwen--Qwen3-Embedding-8B.json

Golden file (fill from the model card -- copy the card's texts and printed scores verbatim,
and record the card URL and the date you copied them):
  {"model": "<exact id in configs/models.json>",
   "source": "<model card URL>", "copied": "YYYY-MM-DD",
   "queries": [...], "documents": [...],
   "expected": [[...], ...],          # len(queries) x len(documents), as printed on the card
   "tol": 0.01,                       # absolute tolerance per score
   "max_seq_len": 512}
On a pass, set the contract's status to "verified" and add
"verified": {"date": ..., "golden": <file>, "versions": <printed below>}.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap import models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("golden")
ap.add_argument("--contracts", default=models.DEFAULT_PATH)
ap.add_argument("--device", default=None)
a = ap.parse_args()

g = json.load(open(a.golden))
if not g.get("expected"):
    sys.exit(f"{a.golden}: 'expected' is empty -- copy the scores printed on the model card first")
c = models.get_contract(g["model"], a.contracts, allow_blocked=True)
st = models.load_model(c, a.device)
findings = models.apply_contract(st, c, g.get("max_seq_len", 512))
q = np.asarray(models.make_encode_fn(st, c, "query")(g["queries"], 8), dtype=np.float32)
d = np.asarray(models.make_encode_fn(st, c, "doc")(g["documents"], 8), dtype=np.float32)
got = q @ d.T
try:
    exp = np.asarray(g["expected"], dtype=np.float32)
except ValueError as e:
    sys.exit(f"{a.golden}: 'expected' is not a rectangular numeric matrix ({e}). Every "
             f"row needs the same number of scores.")
if exp.shape != got.shape:
    sys.exit(f"{a.golden}: 'expected' is {exp.shape} but this contract produces {got.shape} "
             f"({len(g['queries'])} queries x {len(g['documents'])} documents). Copy the card's "
             f"matrix verbatim -- a transposed or short matrix would broadcast into a PASS.")
diff = float(np.abs(got - exp).max())
ok = diff <= g.get("tol", 0.01)
print("findings:", findings)
print("expected:\n", np.round(exp, 4), "\ngot:\n", np.round(got, 4))
print(f"max |diff| = {diff:.4f} (tol {g.get('tol', 0.01)}) -> {'PASS' if ok else 'FAIL'}")
print("versions:", json.dumps(models.env_versions()))
sys.exit(0 if ok else 1)
