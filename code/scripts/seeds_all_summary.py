"""Seed robustness of every windowed pair: what changes when the replay's seed changes.

    python scripts/seeds_all_summary.py --results results --seeds results_seeds \
        --pairs P2 P3 P4 P5 P9 P10 P11 P12 P13 --plan planner/plan.csv --md seeds_all.md

The seed drives the evaluation/traffic split, the arrival stream, the random background order
and the adapter's fit sample. Seed 0 is the paper's replay (results/<P>.json); seeds 1..n are
results_seeds/<P>_seed<s>.json. Per pair and seed, at the primary traffic setting:

  winner       the replay's winner over all families
  admitted     the winner over the families D2 admits (S0-S3), the one the plan is scored on
  S2 debt      S2/demand_then_rand's debt, and its cost at service C_serve / N
  S4 - S2      S2's debt minus S4's (positive: the lexical union serves better)
  tie          whether every completing S2 order lands within tau of S2/demand_then_rand

Every run is deterministic given its seed, so a range over seeds is a range, not an interval.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap import winner as winner_mod  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--seeds", default="results_seeds")
ap.add_argument("--pairs", nargs="+", required=True)
ap.add_argument("--plan", default=None)
ap.add_argument("--md", default=None)
a = ap.parse_args()

plan = {}
if a.plan and os.path.exists(a.plan):
    plan = {r["pair"]: r for r in csv.DictReader(open(a.plan))}


def seed_of(d, fallback):
    cfg = d.get("config") or {}
    s = (cfg.get("_dev_overrides") or {}).get("seed", cfg.get("seed"))
    return fallback if s is None else int(s)


def row(d, s):
    tr = (d.get("primary_setting") or {}).get("traffic_ratio")
    st = next((x for x in d.get("settings", []) if x.get("traffic_ratio") == tr), None)
    if st is None or not st.get("summaries"):
        return {"seed": s, "error": f"no setting at primary traffic ratio {tr}"}
    sm = st["summaries"]; cfg = d["config"]; tau = (cfg.get("tau") or {}).get("debt", 0.0)
    adm = {k: v for k, v in sm.items() if k.split("/")[0] in ("S0", "S1", "S2", "S3")}
    w_adm = winner_mod.pick_winner_v3(adm, (d.get("reference") or {}).get("M_S", d.get("M_S")),
                                      cfg.get("delta", 0.01), band=tau,
                                      band_tiebreak=cfg.get("band_tiebreak", "simplicity"))
    s2 = sm.get("S2/demand_then_rand", {}); s4 = sm.get("S4/demand_then_rand", {})
    s2d = [v["debt"] for k, v in sm.items() if k.startswith("S2/") and v.get("completed")
           and isinstance(v.get("debt"), (int, float))]
    tie = bool(s2d) and max(s2d) - min(s2d) <= tau + 1e-12
    return {"seed": s, "winner": st.get("winner"), "admitted": w_adm if isinstance(w_adm, str) else str(w_adm),
            "s2_debt": s2.get("debt"), "s2_c": s2.get("C_serve_frac"),
            "adv_s4": (s2["debt"] - s4["debt"]) if s2.get("debt") is not None and s4.get("debt") is not None else None,
            "tie": tie, "tau": tau}


out, lines = {}, []
for p in a.pairs:
    files = []
    if os.path.exists(os.path.join(a.results, f"{p}.json")):
        files.append((os.path.join(a.results, f"{p}.json"), 0))
    for f in sorted(glob.glob(os.path.join(a.seeds, f"{p}_seed*.json"))):
        files.append((f, int(re.search(r"_seed(\d+)\.json$", f).group(1))))
    rows = []
    for f, fb in files:
        d = json.load(open(f))
        rows.append(row(d, seed_of(d, fb)))
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        sys.exit(f"{p}: seed(s) repeated across files: {seeds}")
    ok = [r for r in rows if "error" not in r]
    fam = plan.get(p, {}).get("family")
    summ = {"pair": p, "n": len(ok), "seeds": sorted(r["seed"] for r in ok),
            "plan_family": fam,
            "admitted_matches_plan": sum((r["admitted"] or "").split("/")[0] == fam for r in ok) if fam else None,
            "winner_families": sorted({(r["winner"] or "none").split("/")[0] for r in ok}),
            "s2_debt_min": min((r["s2_debt"] for r in ok if r["s2_debt"] is not None), default=None),
            "s2_debt_max": max((r["s2_debt"] for r in ok if r["s2_debt"] is not None), default=None),
            "s2_c_min": min((r["s2_c"] for r in ok if r["s2_c"] is not None), default=None),
            "s2_c_max": max((r["s2_c"] for r in ok if r["s2_c"] is not None), default=None),
            "adv_s4_min": min((r["adv_s4"] for r in ok if r["adv_s4"] is not None), default=None),
            "adv_s4_max": max((r["adv_s4"] for r in ok if r["adv_s4"] is not None), default=None),
            "s4_clears_tau": sum((r["adv_s4"] or 0) > r["tau"] for r in ok),
            "s2_orders_tie": sum(r["tie"] for r in ok), "rows": rows,
            "errors": [r for r in rows if "error" in r]}
    out[p] = summ
    f4 = lambda x: "-" if x is None else f"{x:+.4f}"
    cs = "-" if summ["s2_c_min"] is None else f"{summ['s2_c_min']:.4f} .. {summ['s2_c_max']:.4f}"
    lines.append(f"| {p} | {summ['n']} | {fam or '-'} | "
                 f"{summ['admitted_matches_plan'] if fam else '-'}/{summ['n']} | {', '.join(summ['winner_families'])} | "
                 f"{f4(summ['s2_debt_min'])} .. {f4(summ['s2_debt_max'])} | "
                 f"{cs} | "
                 f"{f4(summ['adv_s4_min'])} .. {f4(summ['adv_s4_max'])} ({summ['s4_clears_tau']}/{summ['n']}) | "
                 f"{summ['s2_orders_tie']}/{summ['n']} |")

hdr = ("| pair | seeds | plan | admitted winner = plan | winner families | S2 debt | S2 C_serve/N | "
       "S2 - S4 debt (> tau) | S2 orders tie |\n|---|---|---|---|---|---|---|---|---|\n")
total_seeds = sum(s["n"] for s in out.values())
match = sum(s["admitted_matches_plan"] or 0 for s in out.values() if s["plan_family"])
scored = sum(s["n"] for s in out.values() if s["plan_family"])
summary = (f"{len(out)} pairs, {total_seeds} replays; the D2-admitted winner equals the plan's family in "
           f"{match} of {scored}; S2's orders tie within tau in {sum(s['s2_orders_tie'] for s in out.values())} of {total_seeds}")
print(hdr + "\n".join(lines) + "\n\n" + summary)
if a.md:
    with open(a.md, "w") as fh:
        fh.write("# Seed robustness, every windowed pair\n\n" + hdr + "\n".join(lines) + "\n\n" + summary +
                 "\n\n```json\n" + json.dumps(out, indent=1, default=str) + "\n```\n")
