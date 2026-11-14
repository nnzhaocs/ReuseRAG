"""Zipf replays at further skews: write the predictions before the replays, score them after.

    # before any replay (the driver commits the file it writes; the commit is the timestamp)
    python scripts/zipf2.py prereg --scan zipf_scan --price union_price_zipf \
        --plan "0.3,0.6,1.2=P2,P3,P4,P5,P9,P10,P11,P12,P13,P14" --plan "0.9=P3,P4,P9,P10,P12,P14" \
        --window 50 --md PREREG_ZIPF2.md

    # after
    python scripts/zipf2.py score --prereg PREREG_ZIPF2.md --results-pattern "results_zipf_a{A}" \
        --md ZIPF2_SCORE.md

Each prediction is for one pair replayed at one alpha, popularity order 0 (the config seed, as in
the Zipf(0.9) replays), window K' = --window. From the static scan (zipf_scan/, D1 and D3 on the
weighted quality inputs) and the static prices (union_price/ run with the same workloads):

  identity   the rebuild's replay debt equals the weighted gap (|diff| <= 1e-9); a check, not a test
  D1/D2      the replay's winner among the families D2 admits (S0-S3) has the plan's family:
             S0 if D1 says rebuild or D3 finds no depth, S2 otherwise
  priced     the replay's winner over every family has the family the winner rule picks from the
             static prices at K' = --window (union_price.static_winner), with and without the margin
  depth gap  |S2 debt - G_S2(K')|, reported
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap import winner as winner_mod  # noqa: E402


def load_block(path):
    return json.loads(re.search(r"```json\n(.*?)```", open(path).read(), re.S).group(1))


def prereg(a):
    cells = []
    for spec in a.plan:
        als, prs = spec.split("=")
        for al in als.split(","):
            for p in prs.split(","):
                cells.append((float(al), p))
    preds = []
    for al, p in cells:
        sc = json.load(open(os.path.join(a.scan, f"{p}.json")))
        row = next((r for r in sc["rows"] if abs(r["alpha"] - al) < 1e-9 and r["pop_seed"] == 0), None)
        if row is None:
            sys.exit(f"{p}: the scan has no alpha {al}, popularity order 0")
        pr = json.load(open(os.path.join(a.price, f"{p}.json")))
        wl = f"zipf:{al:g}"
        pw = pr["workloads"].get(wl)
        if pw is None:
            sys.exit(f"{p}: union_price has no workload {wl}")
        d = pw["depths"].get(str(a.window))
        if d is None:
            sys.exit(f"{p}: union_price has no depth {a.window}")
        if abs(pw["gap"] - row["gap"]) > 1e-9:
            sys.exit(f"{p} {wl}: the scan's weighted gap {row['gap']} and union_price's {pw['gap']} differ; "
                     f"the two scripts weight differently")
        fam = row["family"]
        preds.append({"pair": p, "alpha": al, "workload": wl, "window": a.window,
                      "weighted_gap": row["gap"], "n_eff": row["n_eff"],
                      "plan_family": fam, "plan_kprime": row["kprime"],
                      "static_winner": d["winner"], "static_winner_margin": d["winner_margin"],
                      "static_G_S2": d["prices"].get("S2", {}).get("G"),
                      "static_prices": d["prices"]})
    md = ["# Pre-registered: Zipf replays at further skews (written before any of them ran)", "",
          f"{len(preds)} replays, window K' = {a.window}, popularity order 0 (the config seed). Scored by "
          "`scripts/zipf2.py score`; see its docstring for the criteria.", "",
          "| pair | alpha | n_eff | weighted gap | plan (D1-D3) | priced winner | priced, margin |",
          "|---|---|---|---|---|---|---|"]
    for r in preds:
        md.append(f"| {r['pair']} | {r['alpha']:g} | {r['n_eff']:,.0f} | {r['weighted_gap']:+.4f} | "
                  f"{r['plan_family']}@{r['plan_kprime'] or '-'} | {r['static_winner']} | {r['static_winner_margin']} |")
    md += ["", "```json", json.dumps({"window": a.window, "predictions": preds}, indent=1), "```", ""]
    open(a.md, "w").write("\n".join(md))
    print(f"{len(preds)} predictions written to {a.md}")


def score(a):
    pre = load_block(a.prereg)
    rows = []
    for r in pre["predictions"]:
        f = os.path.join(a.results_pattern.replace("{A}", f"{r['alpha']:g}"), f"{r['pair']}.json")
        if not os.path.exists(f):
            rows.append({**r, "missing": f}); continue
        d = json.load(open(f))
        if d["config"].get("workload") != r["workload"]:
            sys.exit(f"{f} ran under {d['config'].get('workload')}, predicted for {r['workload']}")
        tr = d["primary_setting"]["traffic_ratio"]
        st = next(s for s in d["settings"] if s["traffic_ratio"] == tr)
        sm, cfg = st["summaries"], d["config"]
        tau = (cfg.get("tau") or {}).get("debt", 0.0)
        adm = {k: v for k, v in sm.items() if k.split("/")[0] in ("S0", "S1", "S2", "S3")}
        w_adm = winner_mod.pick_winner_v3(adm, (d.get("reference") or {}).get("M_S", d.get("M_S")),
                                          cfg.get("delta", 0.01), band=tau,
                                          band_tiebreak=cfg.get("band_tiebreak", "simplicity"))
        w_adm = w_adm if isinstance(w_adm, str) else None
        win = st.get("winner")
        s0 = (sm.get("S0/rand") or {}).get("debt")
        s2 = (sm.get("S2/demand_then_rand") or {}).get("debt")
        fam = lambda w: w.split("/")[0] if w else None
        rows.append({**r, "replay_winner": win, "replay_admitted": w_adm,
                     "identity_diff": (s0 - r["weighted_gap"]) if s0 is not None else None,
                     "d1d2_ok": fam(w_adm) == r["plan_family"],
                     "priced_ok": fam(win) == r["static_winner"],
                     "priced_margin_ok": fam(win) == r["static_winner_margin"],
                     "s2_debt": s2, "s2_debt_minus_G": (s2 - r["static_G_S2"]) if s2 is not None and r["static_G_S2"] is not None else None})
    done = [x for x in rows if "missing" not in x]
    bad_id = [x for x in done if x["identity_diff"] is None or abs(x["identity_diff"]) > 1e-9]
    summ = {"n": len(rows), "done": len(done), "identity_bad": len(bad_id),
            "d1d2_ok": sum(x["d1d2_ok"] for x in done), "priced_ok": sum(x["priced_ok"] for x in done),
            "priced_margin_ok": sum(x["priced_margin_ok"] for x in done)}
    md = ["# Zipf replays at further skews, scored against PREREG_ZIPF2.md", "",
          f"{summ['done']} of {summ['n']} replays; identity holds on {summ['done'] - summ['identity_bad']}; "
          f"D1-D3 family right on {summ['d1d2_ok']}; priced winner right on {summ['priced_ok']} "
          f"({summ['priced_margin_ok']} with the margin)", "",
          "| pair | alpha | plan | admitted winner | priced | replay winner | S2 debt - G |", "|---|---|---|---|---|---|---|"]
    for x in rows:
        if "missing" in x:
            md.append(f"| {x['pair']} | {x['alpha']:g} | missing: {x['missing']} | | | | |"); continue
        dg = f"{x['s2_debt_minus_G']:+.4f}" if x["s2_debt_minus_G"] is not None else "-"
        md.append(f"| {x['pair']} | {x['alpha']:g} | {x['plan_family']} | {x['replay_admitted']} "
                  f"{'ok' if x['d1d2_ok'] else 'MISS'} | {x['static_winner']} | {x['replay_winner']} "
                  f"{'ok' if x['priced_ok'] else 'MISS'} | {dg} |")
    md += ["", "```json", json.dumps({"summary": summ, "rows": rows}, indent=1), "```", ""]
    open(a.md, "w").write("\n".join(md))
    print(md[2])
    if bad_id:
        sys.exit(f"IDENTITY FAILED on {[(x['pair'], x['alpha']) for x in bad_id]}: the replay's weights "
                 f"are not the scan's")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("prereg")
    a1.add_argument("--scan", required=True); a1.add_argument("--price", required=True)
    a1.add_argument("--plan", action="append", required=True); a1.add_argument("--window", type=int, default=50)
    a1.add_argument("--md", required=True)
    a2 = sub.add_parser("score")
    a2.add_argument("--prereg", required=True); a2.add_argument("--results-pattern", required=True)
    a2.add_argument("--md", required=True)
    a = ap.parse_args()
    prereg(a) if a.cmd == "prereg" else score(a)


if __name__ == "__main__":
    main()
