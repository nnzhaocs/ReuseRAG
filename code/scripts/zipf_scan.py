"""How workload skew moves the planner's quality inputs: D1 and D3 under Zipf(alpha), statically.

    python scripts/zipf_scan.py pairs/P2 pairs/P3 ... --config configs/msmarco_1m_v3.json \
        --alpha 0,0.3,0.6,0.9,1.2 --pop-seeds 5 --zipf-results results_zipf --out-dir zipf_scan

Quality is workload-weighted: M_S, M_T and G(K') are means over the evaluation queries weighted
by how often each is served. Under Zipf(alpha) the weights follow a random popularity order over
the queries (pair.make_weights, seeded), so which queries are heavy depends on that order. This
script computes each evaluation query's recall once (source top-K, target top-K, the target's
re-ranking of each source window) and then, for every alpha and popularity order, the weighted
gap, the weighted G ladder, D1/D3's decision (plan.decide, depths <= K dropped) and the
effective number of evaluation queries, n_eff = (sum w)^2 / sum w^2.

Validation: the pre-registered Zipf(0.9) replays (results_zipf/) used popularity order 0, and a
rebuild serves the old model until it switches, so S0's replay debt there is the weighted gap.
With --zipf-results the script prints both; they must agree.

No replay, no new vectors; minutes per pair on the CPU.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap import runner  # noqa: E402
from reusemap.measure import _union_rank  # noqa: E402
from reusemap.metrics import METRICS  # noqa: E402
from reusemap.pair import Pair, make_weights  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan as _plan  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--kprime", default="25,50,100,200,500")
    ap.add_argument("--alpha", default="0,0.3,0.6,0.9,1.2", help="0 means uniform")
    ap.add_argument("--pop-seeds", type=int, default=5, help="popularity orders per alpha (order 0 = the replays')")
    ap.add_argument("--zipf-results", default=None, help="results_zipf/ to validate against (alpha 0.9, order 0)")
    ap.add_argument("--out-dir", required=True)
    return ap


def scan(a):
    """Run the scan for parsed arguments `a`; returns {pair: output dict}. Exits non-zero
    if a validation against a Zipf replay disagrees (|diff| > 1e-9)."""
    ks = sorted({int(k) for k in a.kprime.split(",") if k.strip()})
    alphas = sorted({float(x) for x in a.alpha.split(",") if x.strip()})
    if not ks or min(ks) <= 0 or not alphas or min(alphas) < 0 or a.pop_seeds < 1:
        sys.exit("--kprime must list positive depths, --alpha non-negative values, --pop-seeds >= 1")
    cfg_file = json.load(open(a.config))
    K0 = int(cfg_file.get("K", runner.DEFAULT_CFG.get("K", 10)))
    if min(ks) <= K0:   # D3 drops depths <= K, so they would leave an empty ladder and None decisions
        sys.exit(f"--kprime must list depths > K = {K0}; got {ks}")
    os.makedirs(a.out_dir, exist_ok=True)
    summary, outs, bad = [], {}, []

    for pair_dir in a.pairs:
        name = os.path.basename(pair_dir.rstrip("/"))
        cfg = dict(runner.DEFAULT_CFG); cfg.update(cfg_file)
        cfg["kprime"] = ks; cfg.pop("kprime_frac", None); cfg["replay"] = False; cfg["workload"] = "uniform"
        p = Pair(pair_dir, cfg)
        if "window" not in cfg_file:
            cfg["window"] = max(cfg["K"], int(round(cfg.get("window_frac", 0.0005) * p.N))); p.cfg = cfg
        p.precompute()
        K, prim = cfg["K"], cfg["primary"]
        f = METRICS[prim["metric"]]
        tau = float(cfg["tau"]["debt"]) if isinstance(cfg.get("tau"), dict) else float(cfg.get("tau", 0.005))
        eps = float(cfg["eps"]); seed0 = int(cfg.get("seed", 0))
        ev = np.asarray(p.eval_q, np.int64)

        def per_query(rank):
            out = np.full(len(ev), np.nan)
            for i, qi in enumerate(ev):
                rels = p.qrels.get(int(qi))
                if rels:
                    v = f(rank(qi), rels, K, prim["thr"])
                    if v is not None:
                        out[i] = v
            return out
        vS = per_query(lambda qi: p.src_top[qi, :K])
        vT = per_query(lambda qi: p.tgt_top[qi, :K])
        vW = {k: per_query(lambda qi, k=k: _union_rank(p.src_top[qi, :k], p.tscore(qi, p.src_top[qi, :k]), K))
              for k in ks if k <= p.src_top.shape[1]}
        ok = ~np.isnan(vS) & ~np.isnan(vT)

        def wmean(v, w):
            m = ok & ~np.isnan(v)
            return float((w[m] * v[m]).sum() / w[m].sum())

        rows = []
        for al in alphas:
            for s in range(a.pop_seeds if al > 0 else 1):
                wl = "uniform" if al == 0 else f"zipf:{al}"
                w = make_weights(p.Q, wl, seed0 + s)[ev]
                ms, mt = wmean(vS, w), wmean(vT, w)
                G = {k: mt - wmean(v, w) for k, v in vW.items()}
                d = _plan.decide(name, ms, mt, {k: g for k, g in G.items() if k > K}, None, tau, eps)
                wm = w[ok]
                rows.append({"alpha": al, "pop_seed": s, "M_S": ms, "M_T": mt, "gap": mt - ms, "G": G,
                             "family": d["family"], "kprime": d["kprime"],
                             "n_eff": float(wm.sum() ** 2 / (wm ** 2).sum()), "top1_share": float(wm.max() / wm.sum())})
        uni = rows[0]
        val = None
        if a.zipf_results and os.path.exists(os.path.join(a.zipf_results, f"{name}.json")):
            z = json.load(open(os.path.join(a.zipf_results, f"{name}.json")))
            st = next((x for x in z.get("settings", []) if x.get("traffic_ratio") == (z.get("primary_setting") or {}).get("traffic_ratio")), None)
            al_z = float(z["config"]["workload"].split(":")[1])
            mine = next((r for r in rows if abs(r["alpha"] - al_z) < 1e-12 and r["pop_seed"] == 0), None)
            s0 = None if st is None else (st["summaries"].get("S0/rand") or {}).get("debt")
            if st is None or s0 is None:
                # a Zipf replay we cannot check against is a failed self-check, not a skipped one
                print(f"{name}: VALIDATION MISSING: the replay has no primary setting or no S0/rand")
                bad.append(name)
            elif mine is None:
                print(f"{name}: no validation (alpha {al_z} was not scanned; add it to --alpha)")
            else:
                val = {"alpha": al_z, "replay_S0_debt": s0, "static_gap": mine["gap"], "diff": s0 - mine["gap"]}
                ok_v = abs(val["diff"]) <= 1e-9
                print(f"{name}: Zipf({al_z}) replay S0 debt {s0:+.6f} vs static weighted gap "
                      f"{val['static_gap']:+.6f} (diff {val['diff']:+.2e}){'' if ok_v else '  VALIDATION FAILED'}")
                if not ok_v:
                    bad.append(name)
        flips = [r for r in rows if (r["family"], r["kprime"]) != (uni["family"], uni["kprime"])]
        out = {"pair": name, "N": p.N, "n_eval": int(len(ev)), "K": K, "tau": tau, "eps": eps, "kprime": ks,
               "uniform": {"family": uni["family"], "kprime": uni["kprime"], "gap": uni["gap"]},
               "rows": rows, "validation": val,
               "decision_changes": [{"alpha": r["alpha"], "pop_seed": r["pop_seed"], "family": r["family"],
                                     "kprime": r["kprime"], "gap": r["gap"]} for r in flips]}
        json.dump(out, open(os.path.join(a.out_dir, f"{name}.json"), "w"), indent=1)
        outs[name] = out
        for al in alphas:
            rs = [r for r in rows if r["alpha"] == al]
            same = sum((r["family"], r["kprime"]) == (uni["family"], uni["kprime"]) for r in rs)
            decs = ", ".join(sorted({"%s@%s" % (r["family"], r["kprime"] or "-") for r in rs}))
            summary.append(f"| {name} | {al:g} | {np.mean([r['n_eff'] for r in rs]):,.0f} | "
                           f"{min(r['gap'] for r in rs):+.4f} .. {max(r['gap'] for r in rs):+.4f} | "
                           f"{same}/{len(rs)} | {decs} |")
        print(f"{name}: uniform {uni['family']}@{uni['kprime']}; {len(flips)} of {len(rows) - 1} skewed settings decide otherwise")

    hdr = ("| pair | alpha | n_eff | weighted gap | same decision as uniform | decisions |\n"
           "|---|---|---|---|---|---|\n")
    with open(os.path.join(a.out_dir, "ZIPF_SCAN.md"), "w") as fh:
        fh.write("# D1/D3 under Zipf(alpha), static\n\n" + hdr + "\n".join(summary) + "\n")
    print(hdr + "\n".join(summary))
    if bad:
        sys.exit(f"VALIDATION FAILED on {bad}: the static weighted gap does not equal the Zipf replay's "
                 f"S0 debt, so the weights or the evaluation set differ from the replay's")
    return outs


def main():
    return scan(build_parser().parse_args())


if __name__ == "__main__":
    main()
