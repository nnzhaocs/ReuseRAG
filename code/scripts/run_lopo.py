"""Leave-one-pair-out winner prediction over a set of pair result JSONs (RQ2).

Excludes pairs marked as controls/sanity checks via --exclude.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reusemap.winner import lopo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--exclude", nargs="*", default=[])
ap.add_argument("--objective", choices=["auto", "v2", "debt", "c_serve"], default="auto",
                help="auto: 'debt' for replay-v3 results, the v2 C_eps rule otherwise. Regret is "
                     "in the objective's units (quality for debt, fractions of N for the others)")
ap.add_argument("--features", nargs="*", default=None,
                help="feature columns to use, e.g. kappa rho coverage. Default: the kappa/rho "
                     "subsets DESIGN.md's Claim A names. Only add coverage once scripts/coverage.py "
                     "has shown it varies across the pairs (see RUNBOOK Phase 5.5)")
ap.add_argument("--rho-sweeps", nargs="*", default=[],
                help="directories of sweep.py outputs (sweep_<pair>.json). A result whose own rho "
                     "ladder lacks its replay window (the K'=50 text results: ladder 500-5000) "
                     "stores rho(first rung) as features.rho; with this flag the run takes "
                     "rho(window) from the sweep instead, and refuses if no sweep has it")
ap.add_argument("--out")
a = ap.parse_args()


def rho_from_sweeps(name, window):
    for d in a.rho_sweeps:
        f = os.path.join(d, f"sweep_{name}.json")
        if os.path.exists(f):
            row = next((x for x in json.load(open(f))["depths"] if int(x["kprime"]) == int(window)), None)
            if row is not None:
                return float(row["rho"]), f
    return None, None

raw, v3 = [], []
for path in a.results:
    r = json.load(open(path))
    if "summaries" not in r or r.get("measure_only") or r["name"] in a.exclude:
        continue
    w = r.get("window")
    if w is not None and str(w) not in r.get("rho", {}):
        # the runner fell back to the first rung of the result's own ladder: the stored feature
        # is NOT rho at the replay depth. Substitute rho(window) from a sweep.
        rho_w, src = rho_from_sweeps(r["name"], w)
        if rho_w is not None:
            r["features"]["rho"] = rho_w
            print(f"{r['name']}: rho at the replay window K'={w} taken from {src}", file=sys.stderr)
        elif a.rho_sweeps:
            sys.exit(f"{r['name']}: no sweep in {a.rho_sweeps} has rho at K'={w}")
        else:
            print(f"WARNING {r['name']}: features.rho is rho({min(r['rho'], key=float)}), not rho at the "
                  f"replay window K'={w}; pass --rho-sweeps sweeps_fine", file=sys.stderr)
    raw.append(r)
    v3.append(r.get("replay_version") == 3)
if not raw:
    sys.exit("no replay results to use")
if any(v3) and not all(v3):
    sys.exit("refusing to mix replay-v2 and replay-v3 results in one LOPO table: their winners "
             "come from different objectives. Re-run the v2 pairs under the v3 config.")

if a.objective == "v2":
    objective = None
elif a.objective == "auto":
    objective = raw[0].get("objective", "debt") if all(v3) else None
else:
    objective = a.objective
if objective is not None and not all(v3):
    sys.exit(f"--objective {a.objective} needs replay-v3 results (these are v2)")

pairs = []
for r in raw:
    if objective is None:
        for s in r["summaries"].values():
            if s.get("C_eps") is None:
                s["C_eps"] = float("inf")
    q = {"name": r["name"], "features": dict(r["features"]), "winner": r["winner"],
         "summaries": r["summaries"], "M_S": r["M_S"], "delta": r["delta"], "N": r["N"]}
    if objective is not None:
        pen = (r.get("lopo_penalty") or {}).get(objective)
        if pen is None:
            # the fallback is 1.0, which in debt's units means losing the entire metric -- a
            # plausible "predicted nothing usable" penalty but nowhere near M_T - M_S, so a
            # silent fallback would make regret incomparable across pairs
            print(f"warning: {r['name']} has no lopo_penalty[{objective}] (pre-v3 result?); "
                  f"falling back to 1.0, which is not this pair's M_T - M_S. Regret for this "
                  f"pair is not comparable with the others -- re-run it, or drop it.")
            pen = 1.0
        q["penalty"] = pen
    pairs.append(q)

feats = {"custom": list(a.features)} if a.features else None
cols = (a.features or ["kappa", "rho"])
head = "  ".join(f"{c:>7s}" for c in cols)
print(f"{'pair':14s} {head}  winner")
for q in pairs:
    vals = "  ".join(f"{q['features'].get(c, float('nan')):7.3f}" for c in cols)
    print(f"{q['name']:14s} {vals}  {q['winner']}")
out = lopo(pairs, objective=objective, features=feats)
unit = "x N" if objective is None or objective == "c_serve" else "quality"
print(f"\nobjective: {objective or 'C_eps (v2)'}   regret unit: {unit}")
print(f"\n{'rule':22s} {'acc':>5s} {'mean regret':>17s}")
for rule, t in sorted(out["summary"].items(), key=lambda kv: kv[1]["mean_regret"]):
    print(f"{rule:22s} {t['acc']:5.2f} {t['mean_regret']:17.4f}")
if a.out:
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
