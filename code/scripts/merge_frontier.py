"""Merge per-pair sweep files into the {pair: {K': G}} frontier plan.py reads.

    python scripts/merge_frontier.py sweeps_fine/sweep_P*.json sweeps_csn/sweep_P*.json --out frontier_all.json

sweeps_fine/frontier.json is rewritten by every sweep run and so holds only the LAST pair swept;
passing it to plan.py silently drops every other pair's 10/25/50 rungs. This script builds the
frontier from the per-pair files instead and refuses a pair that appears twice, or a depth that
appears twice inside one pair's file (which a dict would otherwise collapse to the last entry).
"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("sweeps", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = {}
for f in a.sweeps:
    d = json.load(open(f))
    if d["pair"] in out:
        sys.exit(f"{d['pair']} appears in more than one sweep file ({f}); pick one")
    ks = [int(x["kprime"]) for x in d["depths"]]
    dup = sorted({k for k in ks if ks.count(k) > 1})
    if dup:
        sys.exit(f"{d['pair']}: K'={dup} measured more than once in {f}; pick one")
    out[d["pair"]] = {str(x["kprime"]): x["G"] for x in d["depths"]}
with open(a.out, "w") as fh:   # closed before plan.py can read it; no half-written frontier
    json.dump(out, fh, indent=1)
print(f"{len(out)} pairs: " + ", ".join(f"{p} {sorted(int(k) for k in v)}" for p, v in sorted(out.items())))
