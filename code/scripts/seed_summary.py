"""P11 across seeds: is S4's margin over the window strategy noise? (RUNBOOK 14.4)

Each replay is deterministic for its seed; the seed sets the evaluation/traffic split and the
arrival order. Five seeds give five point estimates of the same comparison. This reports, per
seed, the primary-setting winner, the debts of S4 and S2 (demand-then-rand) and S4's advantage
over S2 measured against tau, then the RANGE across seeds. It is a range, not a confidence
interval: the replays record per-strategy means, not per-query quality.

  python scripts/seed_summary.py results/P11.json results_seeds/P11_seed*.json --md seeds_summary.md
"""
import argparse
import json
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+", help="the same pair's replays at different seeds")
ap.add_argument("--a", default="S4/demand_then_rand", help="the strategy whose margin is in question")
ap.add_argument("--b", default="S2/demand_then_rand", help="the strategy it is compared with")
ap.add_argument("--md", default=None)
a = ap.parse_args()

rows = []
for f in a.results:
    with open(f) as fh:
        R = json.load(fh)
    cfg = R.get("config", {})
    seed = cfg.get("_dev_overrides", {}).get("seed", cfg.get("seed"))
    if seed is None:   # no fallback to 0: five files read as seed 0 would claim five seeds for one
        sys.exit(f"{f}: no seed recorded in its config")
    tau = cfg["tau"]["debt"] if isinstance(cfg.get("tau"), dict) else cfg.get("tau", 0.005)
    tr = (R.get("primary_setting") or {}).get("traffic_ratio")
    s = next((x for x in R.get("settings", []) if x.get("traffic_ratio") == tr), None)
    if s is None or a.a not in s["summaries"] or a.b not in s["summaries"]:
        sys.exit(f"{f}: no primary setting, or {a.a}/{a.b} missing from its summaries")
    da, db = s["summaries"][a.a]["debt"], s["summaries"][a.b]["debt"]
    rows.append({"file": f, "seed": seed, "winner": s["winner"], "debt_a": da, "debt_b": db,
                 "advantage": db - da, "over_tau": (db - da) - tau, "tau": tau,
                 "cost_a": s["summaries"][a.a]["C_serve_frac"], "cost_b": s["summaries"][a.b]["C_serve_frac"]})
rows.sort(key=lambda r: r["seed"])
# the point of the summary is variation across seeds, and tau is applied to every row: refuse a
# set that repeats a seed or mixes bands rather than report it as n seeds at one tau
dup = sorted({r["seed"] for r in rows if sum(x["seed"] == r["seed"] for x in rows) > 1})
if dup:
    sys.exit(f"seed(s) {dup} appear more than once: " + ", ".join(r["file"] for r in rows if r["seed"] in dup))
taus = sorted({r["tau"] for r in rows})
if len(taus) > 1:
    sys.exit(f"the files disagree on tau: {taus}")
adv = np.array([r["advantage"] for r in rows])
tau = rows[0]["tau"]
summary = {"n": len(rows), "a": a.a, "b": a.b, "tau": tau,
           "advantage_min": float(adv.min()), "advantage_max": float(adv.max()),
           "advantage_mean": float(adv.mean()), "advantage_sd": float(adv.std(ddof=1)) if len(adv) > 1 else None,
           "seeds_where_a_clears_tau": int((adv > tau).sum()),
           "seeds_where_a_wins": int(sum(bool(r["winner"]) and r["winner"].split("/")[0] == a.a.split("/")[0] for r in rows)),
           "seeds_with_no_winner": int(sum(not r["winner"] for r in rows)),
           "rows": rows}
hdr = "| seed | winner | debt " + a.a + " | debt " + a.b + " | advantage | minus tau |\n|---|---|---|---|---|---|\n"
body = "".join(f"| {r['seed']} | {r['winner']} | {r['debt_a']:+.6f} | {r['debt_b']:+.6f} | {r['advantage']:+.6f} | "
               f"{r['over_tau']:+.6f} |\n" for r in rows)
line = (f"{len(rows)} seeds: advantage of {a.a} over {a.b} ranges {adv.min():+.6f} to {adv.max():+.6f} "
        f"(tau {tau}); it clears tau on {summary['seeds_where_a_clears_tau']} of {len(rows)}, "
        f"and {a.a.split('/')[0]} wins on {summary['seeds_where_a_wins']} of {len(rows)}"
        + (f" ({summary['seeds_with_no_winner']} with no winner)" if summary["seeds_with_no_winner"] else "")
        + ". A range over "
        f"deterministic runs, not a confidence interval.")
print(hdr + body + "\n" + line)
if a.md:
    with open(a.md, "w") as fh:
        fh.write("# Seed robustness\n\n" + hdr + body + "\n" + line + "\n\n```json\n"
                 + json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1) + "\n```\n")
