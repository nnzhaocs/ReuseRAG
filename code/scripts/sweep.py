"""The pre-migration sweep: rho(K'), G(K') and coverage(K') on one K' ladder, per pair.

  # pilot first - one pair, subsampled evaluation set, reports timing and extrapolates
  python scripts/sweep.py pairs/P2 --config configs/msmarco_1m_v3.json \
      --kprime 25,50,100,200,350,500,1000 --pilot 200 --out-dir sweeps

  # then the full set
  python scripts/sweep.py pairs/P* --config configs/msmarco_1m_v3.json \
      --kprime 25,50,100,200,350,500,1000 --out-dir sweeps

Everything here is measure-only: existing vectors, no replay, no re-encoding.

`Pair.precompute` does **three** exact retrievals, not one: the source index to `max(ladder,
probe_c * window)`, the target index to `tgt_depth`, and the adapter-mapped queries against the
source index. All three are unconditional, so `--mapped` costs no extra retrieval - it only adds
the rho computation over `map_top`, which is already there. Within each of the three, every depth
on the ladder is a nested prefix, so lengthening the ladder is nearly free; the cost is the three
retrievals, which is what the pilot measures rather than the arithmetic.

WHY THIS RUNS BEFORE ANYTHING ELSE. Four questions, one measurement:

 1. **How large is the saving, really?** The ceiling on what any window strategy can save
    against a full rebuild is `1 - coverage(K')`. On the 1M run at the pre-registered K' = 500
    that was 12-16%. coverage falls steeply with depth (0.879 at 500, 0.645 at 200, 0.437 at
    100, 0.265 at 50 for MiniLM), so the ceiling at K' = 100 would be 56%. Nobody has measured
    whether quality survives down there, because G was only ever computed at 500 and above.

 2. **Does quality survive?** A window strategy is admissible only where `G(K') <= eps`. G was
    <= 0 at K' = 500 on every real pair - the window slightly BEAT the full index, because it
    also drops distractors - but that is a statement about one depth and says nothing about 100.
    G, not rho, is the quality price: the code pair had rho = 0.372 at K' = 50 (37% of the
    target's top-10 outside the window) for G = 0.011, so displacement and damage decouple.

 3. **Does the map's own feature have any variance?** `features.rho` is read at the window
    depth. At K' = 500 the five real pairs spanned rho = 0.030-0.085 while the synthetic sweep
    discriminates over 0.50-0.96, so every real pair sat in one cell and LOPO had nothing to
    predict (majority scored accuracy 1.00, regret 0.0000). If rho spreads out at smaller K',
    the pre-registered window was flattening the feature. This script reports the cross-pair
    spread at every depth - as a RESULT, never as a criterion for choosing the window. Picking
    the depth that maximises feature variance would be feature-hacking.

 4. **Which pairs to run?** Among the encoded models the valid upgrade pairs are the population,
    not a sample, so the answer is "all of them" and no selection is needed - which is also what
    keeps LOPO free of a selection effect.

The window is then chosen from the cost/quality frontier alone - the smallest K' whose G stays
within eps on every pair - and that choice is a pre-registration decision: it changes
`features.rho`, G, coverage, the ceiling, the per-query repair costs and the meaning of tau, so
it implies re-running every pair. Record it in DECISIONS.md before the re-runs.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap import provenance, runner  # noqa: E402
from reusemap.measure import reference_qualities, residual_miss, window_gap  # noqa: E402
from reusemap.pair import Pair  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pairs", nargs="+", help="pair directories (assemble_pair.py output)")
ap.add_argument("--config", default=None, help="the config the replay will use, for K/eps/metric")
ap.add_argument("--kprime", default="25,50,100,200,350,500,1000", help="comma-separated ladder")
ap.add_argument("--pilot", type=int, default=0, metavar="N",
                help="subsample N evaluation queries for a fast timing estimate. Marks the output "
                     "pilot=true; never a result")
ap.add_argument("--mapped", action="store_true",
                help="also compute rho over the adapter-mapped window (S3's nominator)")
ap.add_argument("--out-dir", default=None)
a = ap.parse_args()

ks = sorted({int(k) for k in a.kprime.split(",") if k.strip()})
if not ks:
    sys.exit("--kprime must list at least one positive depth")
cfg_file = json.load(open(a.config)) if a.config else {}
eps_list = [float(e) for e in (cfg_file.get("eps_sensitivity") or [cfg_file.get("eps", 0.01)])]


def servable(p):
    """The queries the v3 stream can serve: traffic pool + primary evaluation, minus the
    evaluation-only secondary set, minus anything the workload gives probability 0. Static coverage
    is measured over exactly these. It is the full servable-query window union; first-arrival C_serve
    can be slightly smaller because the last evaluation query may arrive before every traffic query.
    On the 1M run the two happened to match to four decimals.

    NOTE for cross-checking against coverage.py: that script takes an explicit `--exclude-qids`
    (and now `--weights`), so the two agree only when they are given the same query set. Under a
    weighted workload, zero-weight queries are dropped here and are dropped there only if
    --weights is passed.
    """
    try:
        w = np.asarray(p.all_query_weights(), float)
    except (ValueError, AttributeError):
        # every servable query has weight 0; fall back to "everything except the
        # evaluation-only secondary set" rather than to all Q, which would include it
        keep = np.ones(p.Q, bool)
        if len(getattr(p, "secondary_q", [])):
            keep[np.asarray(p.secondary_q, np.int64)] = False
        # still honour the workload's zeros where they can be read, so this path and coverage.py
        # --weights cannot drift apart silently
        pw = getattr(p, "weights", None)
        if pw is not None and np.asarray(pw).shape == (p.Q,) and (keep & (np.asarray(pw, float) > 0)).any():
            keep &= np.asarray(pw, float) > 0
        return np.flatnonzero(keep)
    return np.flatnonzero(w > 0)


def coverage_ladder(src_top, rows, ks, N):
    """|union of src_top[rows, :K']| / N for every K', computed incrementally: the depths are
    nested prefixes, so one boolean mask is filled in ascending order instead of a fresh
    np.unique per depth."""
    mask = np.zeros(N, bool)
    out, lo = {}, 0
    for k in ks:
        hi = min(k, src_top.shape[1])
        if hi > lo:
            idx = np.unique(src_top[np.ix_(rows, np.arange(lo, hi))])
            # exact search at depth <= N never pads; a -1 pad would silently mark the last document
            assert (idx >= 0).all(), "padded (-1) ids in the source top-k: depth exceeds N"
            mask[idx] = True
            lo = hi
        out[k] = int(mask.sum())
    return out


results, timings = [], {}
for pair_dir in a.pairs:
    name = os.path.basename(pair_dir.rstrip("/"))
    t0 = time.time()
    cfg = dict(runner.DEFAULT_CFG)
    cfg.update(cfg_file)
    cfg["kprime"] = ks                       # the ladder, not the config's own
    cfg.pop("kprime_frac", None)
    # The sweep never replays, so the replay-only guard in Pair.__init__ ("no traffic queries
    # left after holding out the eval set") must not block it: that lets the sweep run against a
    # measure-only config such as nq_100k_v2.json, which holds out every judged query.
    cfg["replay"] = False
    full_n_eval = full_n_servable = None
    if a.pilot:
        # the untruncated sizes, for an honest extrapolation. Pair.__init__ does no retrieval,
        # so this costs nothing, and it avoids hard-coding 3490 (which is only MS MARCO 1M at
        # eval_frac 0.5 and is wrong for the code corpus).
        full = Pair(pair_dir, dict(cfg, n_eval_queries=None))
        full_n_eval = int(len(full.eval_q))
        try:
            full_n_servable = int((np.asarray(full.all_query_weights(), float) > 0).sum())
        except (ValueError, AttributeError):
            full_n_servable = int(full.Q)
        cfg["n_eval_queries"] = int(a.pilot)
    p = Pair(pair_dir, cfg)
    if "window" not in cfg_file:
        cfg["window"] = max(cfg["K"], int(round(cfg.get("window_frac", 0.0005) * p.N)))
        p.cfg = cfg
    t_load = time.time() - t0

    t = time.time()
    p.precompute()
    t_pre = time.time() - t

    prim = cfg["primary"]
    t = time.time()
    ref = reference_qualities(p, prim["metric"], prim["thr"])
    t_ref = time.time() - t

    t = time.time()
    rho = residual_miss(p, ks, "src")
    rho_map = residual_miss(p, ks, "map") if a.mapped else {}
    t_rho = time.time() - t

    t = time.time()
    G = window_gap(p, ks, ref["M_T"], prim["metric"], prim["thr"], "src")
    t_G = time.time() - t

    t = time.time()
    rows = servable(p)
    cov_n = coverage_ladder(p.src_top, rows, ks, p.N)
    t_cov = time.time() - t

    # Three stages, three different counts, so three scalings:
    #   precompute  - retrievals over ALL Q queries, unaffected by n_eval_queries -> fixed
    #   ref/rho/G   - per evaluation query                  -> scales with n_eval
    #   coverage    - per SERVABLE query, {q : weight > 0}, which does NOT depend on
    #                 n_eval_queries, so this scale is 1.0 today. It is computed rather than
    #                 assumed so the estimate stays right if that definition ever changes - and
    #                 it is why --pilot leaves coverage and the ceiling exact rather than biased.
    res_est = None
    if a.pilot:
        q_scale = (full_n_eval / len(p.eval_q)) if len(p.eval_q) else 1.0
        c_scale = (full_n_servable / len(rows)) if len(rows) else 1.0
        res_est = {"per_pair_seconds": t_pre + (t_ref + t_rho + t_G) * q_scale + t_cov * c_scale,
                   "q_scale": q_scale, "cov_scale": c_scale, "precompute_seconds": t_pre,
                   "full_n_eval": full_n_eval, "full_n_servable": full_n_servable}
    depths = []
    for k in ks:
        if k not in G:
            continue  # deeper than the retrieval
        c = cov_n[k] / p.N
        depths.append({"kprime": k, "rho": rho.get(k), "rho_mapped": rho_map.get(k),
                       "G": G[k], "coverage": c, "ceiling_frac": 1.0 - c,
                       "covered": cov_n[k],
                       "admissible": {str(e): bool(G[k] <= e) for e in eps_list}})
    secs = {"load": t_load, "precompute": t_pre, "reference": t_ref, "rho": t_rho,
            "G": t_G, "coverage": t_cov, "total": time.time() - t0}
    timings[name] = secs
    # the input fingerprint ties this sweep to the exact vectors it read (make_numbers checks the
    # P5p/P5q sweeps against the fixed Qwen3 query files with it)
    res = {"pair": name, "pair_dir": os.path.abspath(pair_dir), "N": p.N, "Q": p.Q,
           "data_fingerprint": provenance.data_fingerprint(pair_dir),
           "pilot_extrapolation": (res_est if a.pilot else None),
           "n_eval": int(len(p.eval_q)), "n_servable": int(len(rows)),
           "window_in_config": p.Kw, "retrieval_depth": int(p.src_top.shape[1]),
           "kappa": p.compat["kappa"], "cka": p.compat["cka"],
           "M_S": ref["M_S"], "M_T": ref["M_T"], "M_A": ref["M_A"],
           "eps_list": eps_list, "pilot": bool(a.pilot), "meta": p.meta,
           "depths": depths, "seconds": secs}
    results.append(res)

    print(f"\n=== {name}  N={p.N}  n_eval={len(p.eval_q)}  n_servable={len(rows)}  "
          f"kappa={p.compat['kappa']:.4f}  M_S={ref['M_S']:.4f}  M_T={ref['M_T']:.4f}"
          + ("   [PILOT - not a result]" if a.pilot else ""))
    print(f"{'K-prime':>8s} {'rho':>8s} {'G':>9s} {'coverage':>9s} {'ceiling':>8s}  "
          + "  ".join(f"G<={e:g}" for e in eps_list))
    for d in depths:
        flags = "  ".join(f"{str(d['admissible'][str(e)]):>6s}" for e in eps_list)
        print(f"{d['kprime']:8d} {d['rho']:8.4f} {d['G']:+9.4f} {d['coverage']:9.4f} "
              f"{d['ceiling_frac']:8.4f}  {flags}")
    print(f"  timing: precompute {t_pre:.1f}s  rho {t_rho:.1f}s  G {t_G:.1f}s  "
          f"coverage {t_cov:.1f}s  total {secs['total']:.1f}s")
    if res_est:
        est = res_est["per_pair_seconds"]
        print(f"  PILOT extrapolation: ~{est/60:.1f} min per pair at {full_n_eval} evaluation "
              f"queries and {full_n_servable} servable (precompute {t_pre:.1f}s fixed; per-query "
              f"work x{res_est['q_scale']:.1f}; coverage x{res_est['cov_scale']:.2f}); "
              f"~{est*10/3600:.1f} h for ten pairs.")

# ---------------------------------------------------------------- cross-pair frontier
if results:
    print("\n\n=== frontier: smallest admissible depth per pair, and what it would save ===")
    print(f"{'pair':10s} {'eps':>7s} {'K* (min K: G<=eps)':>20s} {'ceiling at K*':>14s} "
          f"{'G at K*':>9s} {'rho at K*':>10s}")
    frontier = []
    for r in results:
        for e in r["eps_list"]:
            ok = [d for d in r["depths"] if d["admissible"][str(e)]]
            d = min(ok, key=lambda x: x["kprime"]) if ok else None
            frontier.append({"pair": r["pair"], "eps": e,
                             "kstar": d["kprime"] if d else None,
                             "ceiling_frac": d["ceiling_frac"] if d else None,
                             "G": d["G"] if d else None, "rho": d["rho"] if d else None})
            print(f"{r['pair']:10s} {e:7g} {str(d['kprime']) if d else 'none':>20s} "
                  f"{(format(d['ceiling_frac'],'.4f') if d else '-'):>14s} "
                  f"{(format(d['G'],'+.4f') if d else '-'):>9s} "
                  f"{(format(d['rho'],'.4f') if d else '-'):>10s}")

    # The map-feature question, REPORTED not used: how much does rho vary across pairs at each
    # depth? At K'=500 on the 1M run this spread was 0.055 and LOPO had nothing to predict.
    print("\n=== cross-pair rho spread per depth (a result, NOT a criterion for picking K') ===")
    print(f"{'K-prime':>8s} {'n_pairs':>8s} {'rho min':>9s} {'rho max':>9s} {'spread':>8s} "
          f"{'G max':>9s} {'all admissible at eps=' + format(eps_list[0], 'g'):>28s}")
    spread = []
    for k in ks:
        vals = [d for r in results for d in r["depths"] if d["kprime"] == k]
        if not vals:
            continue
        rs = [v["rho"] for v in vals]
        gs = [v["G"] for v in vals]
        row = {"kprime": k, "n_pairs": len(vals), "rho_min": min(rs), "rho_max": max(rs),
               "rho_spread": max(rs) - min(rs), "G_max": max(gs),
               "all_admissible": {str(e): all(v["admissible"][str(e)] for v in vals)
                                  for e in eps_list}}
        spread.append(row)
        print(f"{k:8d} {len(vals):8d} {min(rs):9.4f} {max(rs):9.4f} {max(rs)-min(rs):8.4f} "
              f"{max(gs):+9.4f} {str(row['all_admissible'][str(eps_list[0])]):>28s}")

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        for r in results:
            with open(os.path.join(a.out_dir, f"sweep_{r['pair']}.json"), "w") as f:
                json.dump(r, f, indent=1)
        with open(os.path.join(a.out_dir, "frontier.json"), "w") as f:
            json.dump({"kprime_ladder": ks, "eps_list": eps_list, "pilot": bool(a.pilot),
                       "per_pair": results, "frontier": frontier, "rho_spread": spread,
                       "timings": timings}, f, indent=1)
        print(f"\nwrote {len(results)} sweep file(s) and frontier.json to {a.out_dir}")
    if a.pilot:
        print("\nPILOT RUN - TIMING ONLY. rho and G above come from a subsampled evaluation set "
              "and MUST NOT be used to choose the window, the pair set or anything else.\n"
              "  They are noisy, not biased, and the noise is enough to cross the admissibility "
              "threshold: on the synthetic check pair, pilot G(100) = +0.033 (inadmissible) "
              "against the full run's -0.007 (admissible).\n"
              "  coverage and the ceiling are EXACT here, not estimates: the servable set is "
              "{q : weight > 0}, which does not depend on n_eval_queries, so subsampling leaves "
              "them bit-identical. Re-run without --pilot for the rho and G results.")
