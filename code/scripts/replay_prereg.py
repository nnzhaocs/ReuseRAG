"""Predictions for a replay from its pair's static sweep, written before the replay; scored after.

    python scripts/replay_prereg.py prereg --sweeps sweeps_full/sweep_P9f.json ... --window 50 \
        --label full --md PREREG_REPLAY_full.md
    python scripts/replay_prereg.py score --prereg PREREG_REPLAY_full.md --results results_full \
        --md REPLAY_full_SCORE.md

From the sweep at depth K' = --window (the replay's window), per pair:
  cost     the on-demand window's cost-at-service equals cov(K') to within 0.1% of N
           (Lemma 1: it encodes exactly the union of the served windows)
  quality  the window's debt lies in [G(K') - tau, G(K') + margin]: G is the static price and the
           1M replays put debt above it by at most `margin`
  family   the winner among the families D2 admits (S0-S3) is S2 if D1 migrates (gap > tau) and
           K' is admissible for D3 (G <= eps, gap - G > tau), S0 otherwise
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap import winner as winner_mod  # noqa: E402

TAU, EPS = 0.005, 0.01


def prereg(a):
    preds = []
    for f in a.sweeps:
        s = json.load(open(f))
        row = next((r for r in s["depths"] if r["kprime"] == a.window), None)
        if row is None:
            sys.exit(f"{f}: no row at K' = {a.window}")
        gap = s["M_T"] - s["M_S"]
        fam = "S2" if gap > TAU and row["G"] <= EPS and gap - row["G"] > TAU else "S0"
        preds.append({"pair": s["pair"], "N": s["N"], "window": a.window, "gap": gap, "G": row["G"],
                      "cov": row["coverage"], "family": fam, "margin": a.margin})
    md = [f"# Pre-registered: replays at {a.label} (written from the static sweeps, before the replays)", "",
          "| pair | N | gap | G(K') | cov(K') | predicted family | predicted debt |", "|---|---|---|---|---|---|---|"]
    for p in preds:
        md.append(f"| {p['pair']} | {p['N']:,} | {p['gap']:+.4f} | {p['G']:+.4f} | {p['cov']:.4f} | {p['family']} | "
                  f"[{p['G'] - TAU:+.4f}, {p['G'] + p['margin']:+.4f}] |")
    md += ["", "```json", json.dumps({"label": a.label, "window": a.window, "predictions": preds}, indent=1), "```", ""]
    open(a.md, "w").write("\n".join(md))
    print(f"{len(preds)} predictions written to {a.md}")


def score(a):
    pre = json.loads(re.search(r"```json\n(.*?)```", open(a.prereg).read(), re.S).group(1))
    rows = []
    for p in pre["predictions"]:
        f = os.path.join(a.results, f"{p['pair']}.json")
        if not os.path.exists(f):
            rows.append({**p, "missing": f}); continue
        d = json.load(open(f))
        if int(d.get("N", -1)) != int(p["N"]):
            sys.exit(f"{f}: replayed at N={d.get('N')}, predicted from a sweep at N={p['N']}")
        if int(d["window"]) != p["window"]:
            sys.exit(f"{f}: replayed at window {d['window']}, predicted for {p['window']}")
        tr = d["primary_setting"]["traffic_ratio"]
        st = next(s for s in d["settings"] if s["traffic_ratio"] == tr)
        sm, cfg = st["summaries"], d["config"]
        tau = (cfg.get("tau") or {}).get("debt", 0.0)
        adm = {k: v for k, v in sm.items() if k.split("/")[0] in ("S0", "S1", "S2", "S3")}
        w = winner_mod.pick_winner_v3(adm, (d.get("reference") or {}).get("M_S", d.get("M_S")),
                                      cfg.get("delta", 0.01), band=tau,
                                      band_tiebreak=cfg.get("band_tiebreak", "simplicity"))
        w = w if isinstance(w, str) else None
        s2 = sm.get("S2/demand_then_rand") or {}
        c = s2.get("C_serve_frac"); debt = s2.get("debt")
        # inline encodings of the whole run: the on-demand order encodes exactly the served windows
        # until the last arrival, and this count exists even when the window never reached target
        # quality (then C_serve is N by convention and says nothing about the identity)
        rep = (s2.get("repairs_all_total") or 0) / d["N"] if s2 else None
        cost = c if s2.get("C_serve_basis") == "at_last_arrival" else rep
        rows.append({**p, "replay_admitted": w, "replay_winner": st.get("winner"),
                     "s2_C_serve_frac": c, "s2_basis": s2.get("C_serve_basis"), "s2_debt": debt,
                     "s2_inline_frac": rep, "s2_cost_used": cost,
                     "cost_ok": cost is not None and abs(cost - p["cov"]) <= 0.001,
                     "quality_ok": debt is not None and p["G"] - TAU <= debt <= p["G"] + p["margin"],
                     "family_ok": (w or "").split("/")[0] == p["family"]})
    done = [r for r in rows if "missing" not in r]
    summ = {k: sum(r[k] for r in done) for k in ("cost_ok", "quality_ok", "family_ok")}
    summ.update(n=len(rows), done=len(done))
    md = [f"# Replays at {pre['label']}, scored against {os.path.basename(a.prereg)}", "",
          f"{summ['done']} of {summ['n']} replays; cost {summ['cost_ok']}, quality {summ['quality_ok']}, "
          f"family {summ['family_ok']}", "",
          "| pair | cov | C_serve/N | G | debt | family | admitted winner | replay winner |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "missing" in r:
            md.append(f"| {r['pair']} | missing | | | | | | |"); continue
        cs = f"{r['s2_cost_used']:.4f} ({r['s2_basis']})" if r["s2_cost_used"] is not None else "-"
        db = f"{r['s2_debt']:+.4f}" if r["s2_debt"] is not None else "-"
        md.append(f"| {r['pair']} | {r['cov']:.4f} | {cs} {'ok' if r['cost_ok'] else 'MISS'} | {r['G']:+.4f} | "
                  f"{db} {'ok' if r['quality_ok'] else 'MISS'} | {r['family']} | {r['replay_admitted']} "
                  f"{'ok' if r['family_ok'] else 'MISS'} | {r['replay_winner']} |")
    md += ["", "```json", json.dumps({"summary": summ, "rows": rows}, indent=1), "```", ""]
    open(a.md, "w").write("\n".join(md))
    print(md[2])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("prereg")
    a1.add_argument("--sweeps", nargs="+", required=True); a1.add_argument("--window", type=int, default=50)
    a1.add_argument("--margin", type=float, default=0.006); a1.add_argument("--label", required=True)
    a1.add_argument("--md", required=True)
    a2 = sub.add_parser("score")
    a2.add_argument("--prereg", required=True); a2.add_argument("--results", required=True)
    a2.add_argument("--md", required=True)
    a = ap.parse_args()
    prereg(a) if a.cmd == "prereg" else score(a)


if __name__ == "__main__":
    main()
