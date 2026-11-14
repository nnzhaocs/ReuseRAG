"""How many judged queries do D1 and D3 need? (RUNBOOK 14.5)

D1 decides from the gap M_T - M_S, D3 from the window gap G(K'). Both are means over judged
queries. G(K') also costs target encodings: the documents in the judged queries' source windows,
at most q*K' (a document sample cannot stand in for a K'-deep window over N). This script
recomputes both decisions from q randomly drawn evaluation queries, several seeds per q, on the
vectors that already exist (no new encoding), and reports the smallest q at which every seed
decides exactly as the full evaluation set does, plus the window encodings that q costs.

  python scripts/query_sweep.py pairs/P2 pairs/P3 --config configs/msmarco_1m_v3.json \\
      --q 50,100,250,500,1000,2000 --seeds 5 --out-dir query_sweep

The rules are the paper's: D1 rebuilds if gap <= tau; D3 picks the smallest K' > K with
G(K') <= eps and gap - G(K') > tau, else rebuild. D2 is NOT part of this sweep: it reads cov(K')
on the source index and no judged query, so the number of queries cannot change it. The sweep
passes no coverage to plan.decide, which then answers "window at K'" wherever D1 and D3 do; on a
pair where D2 would turn that window into a rebuild (ceiling <= 1%; no pair in the committed
plan: every rebuild there is D1's), the sweep's S2 means "D1 and D3 chose a window". The full-set
decision is checked against plan.csv when one is given (--plan); a difference there is printed
as a warning, with the reason.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap import runner  # noqa: E402
from reusemap.measure import window_gap  # noqa: E402
from reusemap.metrics import weighted_quality  # noqa: E402
from reusemap.pair import Pair  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan as _plan  # noqa: E402


def decide(M_S, M_T, G, K, tau, eps):
    """(family, K') under D1 and D3; G maps int K' -> window gap.

    Delegates to plan.decide for D1 and D3 (D2 is not a function of the queries; see the module
    docstring), so those two rules cannot drift apart. Depths at or
    below K are dropped first, as the paper's D3 says (K' > K); plan.py reaches the same answer
    on the registered ladders because at K' = K the window is the source top-K and G(K) is the
    whole gap, which its second condition rejects."""
    d = _plan.decide("q", M_S, M_T, {k: v for k, v in G.items() if k > K}, None, tau, eps)
    return d["family"], d["kprime"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="pair directories")
    ap.add_argument("--config", required=True, help="the registered config (K, eps, tau, metric)")
    ap.add_argument("--kprime", default="25,50,100,200,500", help="depths D3 may choose from")
    ap.add_argument("--q", default="50,100,250,500,1000,2000", help="judged-query counts to try")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--plan", default=None, help="planner_*/plan.csv to cross-check the full-set rule")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    ks = sorted({int(k) for k in a.kprime.split(",")})
    qgrid = sorted({int(q) for q in a.q.split(",")})
    if not ks or min(ks) <= 0 or not qgrid or min(qgrid) <= 0:
        sys.exit("--kprime and --q must list positive integers")
    with open(a.config) as fh:
        cfg_file = json.load(fh)
    plan = {}
    if a.plan:
        with open(a.plan) as fh:
            plan = {r["pair"]: r for r in csv.DictReader(fh)}
    os.makedirs(a.out_dir, exist_ok=True)

    for pair_dir in a.pairs:
        name = os.path.basename(pair_dir.rstrip("/"))
        cfg = dict(runner.DEFAULT_CFG)
        cfg.update(cfg_file)
        cfg["kprime"] = ks
        cfg.pop("kprime_frac", None)
        cfg["replay"] = False
        p = Pair(pair_dir, cfg)
        if "window" not in cfg_file:
            cfg["window"] = max(cfg["K"], int(round(cfg.get("window_frac", 0.0005) * p.N)))
            p.cfg = cfg
        p.precompute()
        K, prim = cfg["K"], cfg["primary"]
        tau = float(cfg["tau"]["debt"]) if isinstance(cfg.get("tau"), dict) else float(cfg.get("tau", 0.005))
        eps = float(cfg["eps"])
        ev = np.asarray(p.eval_q, np.int64)

        def at(qs):
            ms = weighted_quality(lambda qi: p.src_top[qi, :K], qs, p.qrels, p.weights, K, prim["metric"], prim["thr"])
            mt = weighted_quality(lambda qi: p.tgt_top[qi, :K], qs, p.qrels, p.weights, K, prim["metric"], prim["thr"])
            g = window_gap(p, ks, mt, prim["metric"], prim["thr"], "src", queries=qs)
            g = {int(k): (None if v is None else float(v)) for k, v in g.items()}
            return float(ms), float(mt), g

        ms0, mt0, g0 = at(ev)
        full = decide(ms0, mt0, g0, K, tau, eps)
        check = None
        if name in plan:
            r = plan[name]
            pk = int(r["kprime"]) if r.get("kprime") else None
            check = {"plan_family": r["family"], "plan_kprime": pk,
                     "agrees": (r["family"] == full[0]) and (pk == full[1] or full[0] != "S2")}
            if not check["agrees"]:
                print(f"WARNING {name}: full-set rule gives {full}, plan.csv says {r['family']} {pk}; "
                      f"the plan used other inputs (e.g. the frontier at other depths)", file=sys.stderr)
        rows = []
        for q in qgrid:
            if q > len(ev):
                continue
            for s in range(a.seeds):
                rng = np.random.default_rng(1000 * q + s)
                qs = np.sort(rng.choice(ev, q, replace=False))
                ms, mt, g = at(qs)
                d = decide(ms, mt, g, K, tau, eps)
                kd = d[1] or full[1]
                enc = (int(np.unique(p.src_top[qs, :kd]).size) if kd else 0)
                rows.append({"q": q, "seed": s, "gap": mt - ms, "G": g, "family": d[0], "kprime": d[1],
                             "same_as_full": d == full, "window_docs_encoded": enc,
                             "window_frac_of_N": enc / p.N})
        agree = {}
        for q in sorted({r["q"] for r in rows}):
            rs = [r for r in rows if r["q"] == q]
            agree[q] = {"all_seeds_agree": all(r["same_as_full"] for r in rs),
                        "share_agree": sum(r["same_as_full"] for r in rs) / len(rs),
                        "gap_sd": float(np.std([r["gap"] for r in rs])),
                        "max_window_frac_of_N": max(r["window_frac_of_N"] for r in rs)}
        ok = [q for q in sorted(agree) if all(agree[x]["all_seeds_agree"] for x in agree if x >= q)]
        out = {"pair": name, "N": p.N, "n_eval": int(len(ev)), "K": K, "tau": tau, "eps": eps,
               "kprime": ks, "full": {"M_S": ms0, "M_T": mt0, "gap": mt0 - ms0, "G": g0,
                                      "family": full[0], "kprime": full[1]},
               "plan_check": check, "by_q": agree, "rows": rows,
               "smallest_q_all_seeds_agree": ok[0] if ok else None}
        with open(os.path.join(a.out_dir, f"{name}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        sq = out["smallest_q_all_seeds_agree"]
        print(f"{name}: full-set decision {full[0]}{'@' + str(full[1]) if full[1] else ''}; "
              f"smallest q at which all {a.seeds} seeds agree: {sq}"
              + (f" (window encodings <= {agree[sq]['max_window_frac_of_N']:.2%} of N)" if sq else ""))


if __name__ == "__main__":
    main()
