"""Score the window-mass cost model against the replays, round by round.

    python scripts/coverage_curve_check.py --window-mass window_mass --results results --md curve_check.md

For every replayed pair whose window strategy repairs on demand only (S2/demand: no background
work), the work done after round i is exactly the union of the windows of the arrivals so far.
The model predicts its expectation from the source encoder's multiplicity histogram alone:
N * E[cov(lambda * i)], lambda = the replay's arrivals per round. Nothing is fitted. Reports,
per pair and traffic ratio, the largest absolute and relative error over all rounds, and
whether any realised point exceeds the independent-window bound.
"""
import argparse, glob, json, math, os, sys
import numpy as np


def rounds_candidates(traj_len, rounds_used, keep=400):
    """Every placement of the stored trajectory points in rounds. The runner appends one point per
    round from the first round in which an evaluation query arrives (r0) to the last, then thins
    the list to <= `keep` points with a fixed stride, always keeping the last (runner._thin). r0
    is not stored, and once more than `keep` rounds were used several r0 can give the same thinned
    length. Return the round numbers for EVERY r0 that fits; callers score a series at its worst
    case over them, so an ambiguous placement can never make the model look better than it is."""
    out = []
    for r0 in range(1, rounds_used + 1):
        R = rounds_used - r0 + 1
        if R <= keep:
            idx = list(range(R))
        else:
            st = int(math.ceil(R / keep)); idx = list(range(0, R, st))
            if idx[-1] != R - 1:
                idx.append(R - 1)
        if len(idx) == traj_len:
            out.append((r0, np.array([r0 + i for i in idx])))
    if not out:
        raise ValueError(f"cannot place {traj_len} trajectory points in {rounds_used} rounds")
    return out


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap.coverage_model import expected_coverage_from_hist, independent_bound  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--window-mass", default=None, help="directory of window_mass.py outputs")
ap.add_argument("--bound-only", action="store_true",
                help="check only the independent bound, which holds for ANY workload (Theorem 2); "
                     "use it for skewed-workload replays, where the uniform multiplicity histogram "
                     "does not describe the arrivals")
ap.add_argument("--results", required=True)
ap.add_argument("--strategy", default="S2/demand")
ap.add_argument("--md")
a = ap.parse_args()
if not a.bound_only and not a.window_mass:
    sys.exit("--window-mass is required unless --bound-only")

WM = {}
for f in (glob.glob(os.path.join(a.window_mass, "*.json")) if a.window_mass else []):
    d = json.load(open(f)); WM[d["name"]] = d
rows, worst_rel, n_pts, n_above, n_amb, n_sat, n_skip = [], 0.0, 0, 0, 0, 0, 0
for f in sorted(glob.glob(os.path.join(a.results, "P*.json"))):
    R = json.load(open(f))
    if not R.get("settings"):
        continue
    src = R["meta"].get("source", "synthetic").split("/")[-1]
    wm = WM.get(src)
    if a.bound_only:
        wm, hist = None, None
    elif wm is None:
        rows.append(f"| {R['name']} | {src} | - | no window-mass file for the source | | |"); continue
    if wm is not None and wm["N"] != R["N"]:
        rows.append(f"| {R['name']} | {src} | - | window mass is for N={wm['N']}, replay N={R['N']} | | |"); continue
    if wm is not None:
        row = next((r for r in wm["rows"] if r["kprime"] == R["window"]), None)
        if row is None:
            rows.append(f"| {R['name']} | {src} | - | no K'={R['window']} in the window-mass file | | |"); continue
        hist = {int(k): v for k, v in row["hist"].items()}
    for s in R["settings"]:
        traj = s["trajectories"].get(a.strategy)
        if not traj:
            n_skip += 1; continue
        lam = s["stream_len"]
        cands = rounds_candidates(len(traj), s["summaries"][a.strategy]["rounds_used"])
        real = np.array([c for c, _ in traj], float) / R["N"]
        ab = np.zeros(len(real)); above = np.zeros(len(real), bool); satur = np.zeros(len(real), bool)
        for _, rr in cands:   # worst case over every placement that fits
            ms = lam * rr
            bound = independent_bound(R["window"], R["N"], ms)
            if hist is not None:
                pred = expected_coverage_from_hist(hist, wm["n_queries"], ms, R["N"])
                ab = np.maximum(ab, np.abs(real - pred))
            # the bound is on EXPECTED coverage; a curve that has covered the whole corpus sits on
            # 1.0 while the bound is 1 - O(1e-10). Count those as saturated, not as violations.
            sat = (real >= 1 - 1e-12) & (bound > 1 - 1e-9)
            over = real > bound + 1e-12
            above |= over & ~sat; satur |= over & sat
        rel = ab / np.maximum(real, 1e-12)
        n_pts += len(real); n_above += int(above.sum()); n_sat += int(satur.sum()); worst_rel = max(worst_rel, rel.max())
        amb = "" if len(cands) == 1 else f" (r0 ambiguous: {', '.join(str(r0) for r0, _ in cands)}; worst case)"
        n_amb += len(cands) > 1
        rows.append(f"| {R['name']} | {src} | {s['traffic_ratio']:g} | {len(real)} rounds{amb} | "
                    + ("- | - |" if hist is None else f"{ab.max():.5f} | {100 * rel.max():.2f}% |")
                    + f" {int(above.sum())} |")
hdr = "| pair | source | tr | points | max abs err (frac of N) | max rel err | above bound |\n|---|---|---|---|---|---|---|\n"
summ = (f"{n_pts} realised points; "
        + ("model not compared (--bound-only); " if a.bound_only else f"worst relative error {100 * worst_rel:.2f}%; ")
        + 
        f"{n_above} points above the independent bound ({n_sat} more at full coverage, where the "
        f"bound is within 1e-9 of 1); {n_amb} series with an ambiguous start round "
        f"(scored at their worst case); {n_skip} settings skipped for having no {a.strategy} trajectory")
print(hdr + "\n".join(rows) + "\n\n" + summ)
if a.md:
    open(a.md, "w").write("# Coverage model vs replay\n\n" + hdr + "\n".join(rows) + "\n\n" + summ + "\n")
