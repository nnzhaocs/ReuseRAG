"""Compute static candidate-union coverage/quality for window families S2--S5.

    python scripts/union_price.py pairs/P2 pairs/P11 ... --config configs/msmarco_1m_v3.json \
        --kprime 25,50,100 --results results --out-dir union_price

For a family F and depth K', this script unions the candidate sets of *all servable queries*:

    candidate_union_cov_F(K') = |U_q C_F(q, K')| / N,
    C_S2 = W_q, C_S3 = A_q, C_S4 = W_q u B_q, C_S5 = W_q u A_q u B_q.

This is a serving-side static quantity: it requires no target-document encoding. It is the endpoint
of inline candidate materialization if every servable query is eventually seen, and therefore a
useful conservative price for comparing families. It is NOT, in general, an identity for the
first-arrival C_serve used by the replay: the last evaluation query may arrive before every traffic
query, and completing background orders can encode documents outside the candidate union. Replay
C_serve is shown only as a diagnostic comparison. For the delayed demand-then-rand reference order,
inline repairs up to the first-arrival endpoint are a subset of this static candidate union.

The quality quantity is

    G_F(K') = M_T - quality of the target model re-ranking C_F(q, K'),

measured on evaluation queries under the workload. The static winner applies the registered
admissibility/equivalence-band rule using candidate-union coverage as its serving-side cost proxy.
`--margin m` additionally applies eps-m.

With --results, the matching replay is read at its primary traffic setting and its own window depth.
The script reports static G and candidate-union coverage beside replay debt, C_serve and inline
repairs. No replay or new vectors are created here.
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

FAMILIES = ("S0", "S1", "S2", "S3", "S4", "S5")
SIMPLE = {f: i for i, f in enumerate(FAMILIES)}
REFERENCE_ORDER = "demand_then_rand"


def servable(p):
    """Same set as sweep.servable: traffic pool + primary evaluation queries with weight > 0."""
    w = np.asarray(p.all_query_weights(), float)
    return np.flatnonzero(w > 0)


def candidates(p, fam, qi, k, kb):
    src = p.src_top[qi, :k]
    if fam == "S2":
        return src
    if fam == "S3":
        return p.map_top[qi, :k]
    if fam == "S4":
        return np.unique(np.concatenate([src, p.bm25[qi, :kb]]))
    if fam == "S5":
        return np.unique(np.concatenate([src, p.map_top[qi, :k], p.bm25[qi, :kb]]))
    raise ValueError(fam)


def static_winner(prices, gap, ma_minus_ms, tau, eps):
    """prices: {fam: {"G": .., "cov": ..}} for window families at one depth; S0/S1 are added here.
    Returns (winner family or None, admissible families)."""
    cand = {"S0": {"G": gap, "cov": 0.0}, "S1": {"G": gap - ma_minus_ms, "cov": 0.0}}
    cand.update(prices)
    adm = {}
    for f, v in cand.items():
        if v["G"] is None or not np.isfinite(v["G"]):
            continue
        if v["G"] > eps:
            continue
        if f == "S1" and ma_minus_ms < 0:
            continue
        adm[f] = v
    if not adm:
        return None, []
    best = min(v["G"] for v in adm.values())
    band = [f for f, v in adm.items() if v["G"] <= best + tau + 1e-12]
    win = min(band, key=lambda f: (adm[f]["cov"], SIMPLE[f]))
    return win, sorted(adm, key=SIMPLE.get)


def replay_cell(res):
    """Replay at the primary traffic setting. For each family, read the demand-then-rand
    reference order's debt, first-arrival C_serve and inline repairs. These are diagnostics beside
    the static candidate-union coverage; they are not assumed equal."""
    prim = res["primary_setting"]["traffic_ratio"]
    st = next((s for s in res["settings"] if s["traffic_ratio"] == prim), None)
    if st is None:
        return None
    fams = {}
    for k, v in st["summaries"].items():
        f, o = k.split("/")
        if f in ("S0", "S1") or o == REFERENCE_ORDER:
            fams[f] = {"debt": v.get("debt"), "C_serve_frac": v.get("C_serve_frac"),
                       "basis": v.get("C_serve_basis"), "reaches_target": v.get("reaches_target"),
                       # every inline encoding of the run, a count that exists whether or not the
                       # strategy reached target quality (C_serve is N by convention when it did not)
                       "repairs_frac": (v.get("repairs_all_total") or 0) / res["N"]}
    w = st.get("winner")
    return {"window": int(res.get("window") or res["config"].get("window")), "winner": w,
            "winner_family": w.split("/")[0] if w else None, "families": fams}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--kprime", default="25,50,100")
    ap.add_argument("--workloads", default="uniform",
                    help="comma list, e.g. uniform,zipf:0.3,zipf:0.6 (popularity order = the config seed, "
                         "as in the replays)")
    ap.add_argument("--results", action="append", default=[], metavar="WORKLOAD=DIR",
                    help="replay results to score against, per workload (e.g. uniform=results)")
    ap.add_argument("--margin", type=float, default=0.006)
    ap.add_argument("--families", default="S2,S3,S4,S5")
    ap.add_argument("--out-dir", required=True)
    return ap


def compare(depth, res, results_dir):
    rc = replay_cell(res)
    if not rc:
        return None
    if rc["window"] not in depth:
        return {"results": results_dir, "window": rc["window"], "skipped": "window not on --kprime"}
    d = depth[rc["window"]]
    cmp_ = {f: {"static_G": d["prices"][f]["G"],
                "static_cov": d["prices"][f]["cov"],  # legacy key: candidate-union coverage
                "static_candidate_union_cov": d["prices"][f]["cov"],
                "replay_debt": rc["families"].get(f, {}).get("debt"),
                "replay_C_serve_frac": rc["families"].get(f, {}).get("C_serve_frac"),
                "replay_basis": rc["families"].get(f, {}).get("basis"),
                "replay_repairs_frac": rc["families"].get(f, {}).get("repairs_frac")}
            for f in d["prices"]}
    return {"results": results_dir, "window": rc["window"], "winner": rc["winner"],
            "winner_family": rc["winner_family"], "static_winner": d["winner"],
            "static_winner_margin": d["winner_margin"], "families": cmp_,
            "replay_S0_debt": rc["families"].get("S0", {}).get("debt"),
            "agree": d["winner"] == rc["winner_family"],
            "agree_margin": d["winner_margin"] == rc["winner_family"]}


def run(a):
    ks = sorted({int(k) for k in a.kprime.split(",") if k.strip()})
    fams = [f for f in a.families.split(",") if f]
    wls = [w for w in a.workloads.split(",") if w]
    resmap = dict(r.split("=", 1) for r in a.results)
    cfg_file = json.load(open(a.config))
    K0 = int(cfg_file.get("K", runner.DEFAULT_CFG["K"]))
    if not ks or min(ks) <= K0:
        sys.exit(f"--kprime must list depths > K = {K0}; got {ks}")
    if any(f not in ("S2", "S3", "S4", "S5") for f in fams):
        sys.exit(f"--families must be window families S2..S5; got {fams}")
    if any(w != "uniform" and not w.startswith("zipf:") for w in wls):
        sys.exit(f"--workloads: uniform or zipf:<alpha>; got {wls}")
    os.makedirs(a.out_dir, exist_ok=True)
    outs = {}
    for pair_dir in a.pairs:
        name = os.path.basename(pair_dir.rstrip("/"))
        cfg = dict(runner.DEFAULT_CFG); cfg.update(cfg_file)
        cfg["kprime"] = ks; cfg.pop("kprime_frac", None); cfg["replay"] = False; cfg["workload"] = "uniform"
        p = Pair(pair_dir, cfg)
        if "window" not in cfg_file:
            cfg["window"] = max(cfg["K"], int(round(cfg.get("window_frac", 0.0005) * p.N)))
        cfg["window"] = max(ks + [int(cfg["window"])])   # src/map retrieval deep enough for every depth
        p.cfg = cfg
        p.precompute()
        pf = [f for f in fams if f not in ("S4", "S5") or p.bm25 is not None]
        K, prim = cfg["K"], cfg["primary"]
        f_m = METRICS[prim["metric"]]
        tau = float(cfg["tau"]["debt"]) if isinstance(cfg.get("tau"), dict) else float(cfg.get("tau", 0.005))
        eps = float(cfg["eps"])
        seed0 = int(cfg.get("seed", 0))
        kb = int(cfg.get("bm25_depth", cfg["window"]))
        if p.bm25 is not None and kb > p.bm25.shape[1]:
            # the replay slices bm25[qi, :bm25_depth], which numpy truncates to the stored depth;
            # price what the replay actually fetched, and say so
            print(f"{name}: WARNING bm25_depth {kb} exceeds bm25.npy depth {p.bm25.shape[1]}; "
                  f"using {p.bm25.shape[1]}, as the replay does")
            kb = int(p.bm25.shape[1])
        ev = np.asarray(p.eval_q, np.int64)

        def per_query(rank):            # each query's metric once; weights applied per workload
            v = np.full(len(ev), np.nan)
            for i, qi in enumerate(ev):
                rels = p.qrels.get(int(qi))
                if rels:
                    x = f_m(rank(qi), rels, K, prim["thr"])
                    if x is not None:
                        v[i] = x
            return v
        vS = per_query(lambda qi: p.src_top[qi, :K]); vT = per_query(lambda qi: p.tgt_top[qi, :K])
        vA = per_query(lambda qi: p.map_top[qi, :K])
        rows = servable(p)
        vF, cov = {}, {}
        for k in ks:
            for fam in pf:
                seen = np.zeros(p.N, bool)
                for qi in rows:
                    seen[candidates(p, fam, qi, k, kb)] = True
                cov[(k, fam)] = float(seen.mean())

                def rank(qi, fam=fam, k=k):
                    c = candidates(p, fam, qi, k, kb)
                    return _union_rank(c, p.tscore(qi, c), K)
                vF[(k, fam)] = per_query(rank)
        out = {"pair": name, "N": int(p.N), "n_eval": int(len(ev)), "n_servable": int(len(rows)),
               "bm25_depth": kb if p.bm25 is not None else None, "tau": tau, "eps": eps,
               "margin": a.margin, "families": pf,
               "cost_semantics": "static candidate-union coverage over all servable queries; not exact first-arrival C_serve",
               "workloads": {}}
        for wl in wls:
            wfull = make_weights(p.Q, wl, seed0)

            def wmean(v):              # same skip rule as metrics.weighted_quality: undefined queries out
                m = ~np.isnan(v)
                ww = wfull[ev][m]
                return float((ww * v[m]).sum() / ww.sum())
            M_S, M_T, M_A = wmean(vS), wmean(vT), wmean(vA)
            gap = M_T - M_S
            depth = {}
            for k in ks:
                prices = {fam: {"G": M_T - wmean(vF[(k, fam)]), "cov": cov[(k, fam)]} for fam in pf}
                win, adm = static_winner(prices, gap, M_A - M_S, tau, eps)
                win_m, adm_m = static_winner(prices, gap, M_A - M_S, tau, eps - a.margin)
                depth[k] = {"prices": prices, "winner": win, "admissible": adm,
                            "winner_margin": win_m, "admissible_margin": adm_m}
            o = {"M_S": M_S, "M_T": M_T, "M_A": M_A, "gap": gap,
                 "n_eff": float(wfull[ev].sum() ** 2 / (wfull[ev] ** 2).sum()),
                 "depths": {str(k): v for k, v in depth.items()}}
            rd = resmap.get(wl)
            if rd and os.path.exists(os.path.join(rd, f"{name}.json")):
                res = json.load(open(os.path.join(rd, f"{name}.json")))
                if res["config"].get("workload", "uniform") != wl:
                    sys.exit(f"{rd}/{name}.json was replayed under {res['config'].get('workload')}, not {wl}")
                o["replay"] = compare(depth, res, rd)
            out["workloads"][wl] = o
            line = "  ".join(f"K'={k}: {d['winner']} " + " ".join(f"{f} G{v['G']:+.4f} cov{v['cov']:.3f}"
                                                             for f, v in d["prices"].items())
                             for k, d in depth.items())
            print(f"{name} [{wl}]: gap {gap:+.4f}  {line}")
            r = o.get("replay") or {}
            if "agree" in r:
                print(f"   replay at {r['window']}: {r['winner']}  static {r['static_winner']} "
                      f"(margin {r['static_winner_margin']})  agree {r['agree']}")
                for f, c in r["families"].items():
                    if c["replay_C_serve_frac"] is not None:
                        print(f"     {f}: candidate-union cov {c['static_candidate_union_cov']:.4f}; "
                              f"replay C_serve {c['replay_C_serve_frac']:.4f} ({c['replay_basis']}), "
                              f"inline repairs {c['replay_repairs_frac']:.4f}; "
                              f"G {c['static_G']:+.4f} vs debt {c['replay_debt']:+.4f}")
        json.dump(out, open(os.path.join(a.out_dir, f"{name}.json"), "w"), indent=1)
        outs[name] = out
    md = ["# Static candidate-union coverage and quality for every window family", "",
          "`cov` below is the union of candidates over all servable queries. It is a serving-side static "
          "quantity / conservative inline-work price, **not an identity for first-arrival C_serve**. "
          "Replay C_serve and repairs are diagnostic comparisons.", "",
          "| pair | workload | gap | K' | " + " | ".join(f"G {f} / cov {f}" for f in fams)
          + " | static winner | replay winner |",
          "|---|---|---|---|" + "---|" * len(fams) + "---|---|"]
    for name, o in outs.items():
        for wl, ow in o["workloads"].items():
            r = ow.get("replay") or {}
            for k, d in ow["depths"].items():
                cells = [f"{d['prices'][f]['G']:+.4f} / {d['prices'][f]['cov']:.3f}" if f in d["prices"] else "-"
                         for f in fams]
                rw = r.get("winner") if r.get("window") == int(k) else ""
                md.append(f"| {name} | {wl} | {ow['gap']:+.4f} | {k} | " + " | ".join(cells)
                          + f" | {d['winner']} | {rw or ''} |")
    open(os.path.join(a.out_dir, "UNION_PRICE.md"), "w").write("\n".join(md) + "\n")
    return outs


def main():
    return run(build_parser().parse_args())


if __name__ == "__main__":
    main()
