"""How much of the map survives a change in the equivalence band tau?

Re-runs the pre-registered winner rule over a grid of tau values, off the summaries already
stored in the result JSONs. No replay, no re-encoding: this reads what the run recorded.

    python scripts/tau_curve.py results/P*.json
    python scripts/tau_curve.py results/P*.json --taus 0,0.002,0.005,0.01 --csv results/tables

Why it exists: tau is a pre-registered judgement (UNVERIFIED.md), and on the 2026-11-04 1M run
the two pairs that make the map look non-trivial are exactly the two that move with it - at
tau = 0.004 P14 picks the window family, at 0.006 P11 does. A single tau column hides that. The
paper should print this curve, or set tau from the bootstrap spread of debt (RUNBOOK 6.3) and
say so.

Exit codes: 0 ok, 2 refused (no usable inputs).
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from reusemap import provenance as prov  # noqa: E402
from reusemap import winner as winner_mod  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+", help="result JSONs (Phase 7 writes results/P*.json)")
ap.add_argument("--taus", default="0,0.002,0.003,0.004,0.005,0.006,0.008,0.01",
                help="comma-separated tau grid, in the objective's units")
ap.add_argument("--csv", default=None, help="directory to write tau_curve.csv into")
ap.add_argument("--include-nonpaper", action="store_true",
                help="also read development runs (--set overrides). Never for paper numbers.")
a = ap.parse_args()

taus = sorted({float(t) for t in a.taus.split(",") if t.strip()})
if not taus:
    sys.exit("--taus must list at least one value")

rows, skipped, accepted = [], [], []
for p in a.results:
    try:
        r = json.load(open(p))
    except (OSError, json.JSONDecodeError) as e:
        skipped.append(f"{os.path.basename(p)}: unreadable ({e})")
        continue
    if r.get("replay_version") != 3:
        skipped.append(f"{r.get('name', p)}: not a v3 replay")
        continue
    if r.get("config", {}).get("_dev_overrides") and not a.include_nonpaper:
        skipped.append(f"{r.get('name', p)}: development overrides")
        continue
    cfg = r.get("config", {})
    objective = cfg.get("objective", "debt")
    tiebreak = cfg.get("band_tiebreak", "simplicity")
    delta, M_S = cfg.get("delta", 0.0), r["reference"]["M_S"]
    cfg_tau = float((cfg.get("tau") or {}).get(winner_mod.OBJECTIVES[objective][1], 0.0))
    accepted.append(r)
    for st in r.get("settings", []):
        sums = st.get("summaries") or {}
        if not sums:
            continue
        picks = {}
        for t in taus:
            w, band = winner_mod.pick_winner_v3(sums, M_S, delta, objective, t,
                                                return_band=True, band_tiebreak=tiebreak)
            picks[t] = (w, len(band))
        rows.append({"pair": r["name"], "traffic_ratio": st["traffic_ratio"],
                     "config_tau": cfg_tau, "picks": picks,
                     "gated": st.get("no_winner_reason") or ""})

if not rows:
    for s in skipped:
        print("  skipped", s)
    print("refused: no usable v3 results", file=sys.stderr)
    sys.exit(2)

ref = min(taus, key=lambda t: abs(t - rows[0]["config_tau"]))
print(f"objective band swept over tau in {taus}; the config's own tau is "
      f"{rows[0]['config_tau']} (nearest grid point {ref})\n")
wid = max(len(str(v[0])) for r_ in rows for v in r_["picks"].values()) + 1
print(f"{'pair':<5} {'tr':>5}  " + " ".join(f"{('tau='+str(t)):<{wid}}" for t in taus))
for r_ in rows:
    marks = " ".join(f"{str(r_['picks'][t][0]):<{wid}}" for t in taus)
    print(f"{r_['pair']:<5} {r_['traffic_ratio']:>5.0f}  {marks}"
          + ("   [gated: no winner reported]" if r_["gated"] else ""))

print(f"\ncells where the winner differs from the one at tau = {ref}"
      f"{'' if ref == rows[0]['config_tau'] else ' (the grid point nearest the config tau ' + str(rows[0]['config_tau']) + ')'}:")
labile = []
for t in taus:
    diff = [f"{x['pair']}@{x['traffic_ratio']:.0f}" for x in rows
            if x["picks"][t][0] != x["picks"][ref][0]]
    print(f"   tau={t:<7} {len(diff):>3} / {len(rows)}" + (f"   {', '.join(diff)}" if diff else ""))
    labile += diff
print(f"\npairs that move anywhere on this grid: "
      f"{sorted({x.split('@')[0] for x in labile}) or 'none'}")
print("A pair that moves is one whose reported strategy is a function of tau, not of the pair.")
for s in skipped:
    print("  skipped", s)

if a.csv:
    os.makedirs(a.csv, exist_ok=True)
    out = os.path.join(a.csv, "tau_curve.csv")
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["pair", "traffic_ratio", "tau", "winner", "n_in_band", "config_tau"])
        for r_ in rows:
            for t in taus:
                w, nb = r_["picks"][t]
                wr.writerow([r_["pair"], r_["traffic_ratio"], t, w, nb, r_["config_tau"]])
    # summarize.py writes the other five tables into this same directory. Whichever of the two
    # is not re-run after a new replay leaves a stale CSV beside current ones, which is how the
    # 2026-11-07 archive shipped a tau_curve.csv disagreeing with its own results on 177 rows.
    prov.write_table_inputs(out, accepted, ROOT)
    print(f"\nwrote {out}")
