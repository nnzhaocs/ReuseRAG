"""Mechanism predictions for one pair, from STATIC measurements only.

  python scripts/predict.py --result results/P2_measure.json --coverage coverage/MiniLM.json \
      --config configs/msmarco_1m_v3.json --out predictions/P2.json

Commit the output BEFORE running the revised replay, then check the replay against it. That is
the protection that is missing while nothing is formally pre-registered and the development
results have already been seen: a protocol change justified by a mechanism survives review, one
justified by "the winner labels looked meaningless" does not - and the way to tell them apart is
whether the mechanism predicted the numbers in advance.

Agreement validates the implementation. Disagreement INSIDE the stated band is the arrival
schedule's randomness; outside it, it is a bug, not a finding.

What is predicted, and from what:
  reach         a window strategy is admissible iff G(K') <= eps          [G from measure.py]
  optimal K'    the smallest depth whose G(K') <= eps                     [G]
  ceiling       the largest possible saving against a rebuild, 1 - coverage(K')  [coverage.py]
  C_serve       1 - (1 - coverage)(1 - f_bg), f_bg = background share done by the last arrival
  debt(window)  ~ G(K'): a repaired window serves target quality minus the window gap
  debt(S0)      ~ (M_T - M_S) x (share of arrivals that happen before completion)
  per-query     a cold query at migration fraction f needs ~ (1 - f) K' encodings

The band is NOT one blanket tolerance. The union term, |union of windows| / N, is a fixed set
given the source index and K': it has no Monte Carlo component at all. All the variance is in
f_bg - which documents background happens to have reached by each arrival - so the band is put
on f_bg alone (+/- one round of background work, plus the arrival-schedule spread).

Note on the arrival model: draws are with replacement from the workload, so the rounds needed to
serve the evaluation population is a weighted coupon-collector quantity, NOT n_eval/stream_len.
The per-evaluation-query draw probabilities come from the result's `arrival` block, which
`run_pair.py` records exactly. They cannot be reconstructed from the counts — the held-out set is
a seeded permutation of the judged queries, not the first n_eval rows — so for a non-uniform
workload (ORCAS) without that block this script REFUSES rather than guess.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from reusemap.simulate import expected_arrival_rounds  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--result", required=True, help="pair result JSON (measure-only is enough: needs G, rho, reference)")
ap.add_argument("--coverage", required=True, help="coverage JSON for the pair's SOURCE model (scripts/coverage.py)")
ap.add_argument("--config", default=None, help="the v3 config that will be run (for eps, rounds, traffic_ratio)")
ap.add_argument("--weights", default=None,
                help="weights.npy, for a result that predates the recorded 'arrival' block. Needs "
                     "--eval-qids too, or the script refuses rather than guess which queries are "
                     "the held-out evaluation set")
ap.add_argument("--eval-qids", default=None,
                help="file of the held-out evaluation query indices, one per line (same style as "
                     "coverage.py's --qids). Only needed with --weights")
ap.add_argument("--out", default=None)
a = ap.parse_args()

res = json.load(open(a.result))
cov = json.load(open(a.coverage))
cfg = json.load(open(a.config)) if a.config else {}
cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}

N = int(res["N"])
if int(cov["N"]) != N:
    sys.exit(f"coverage was measured on N={cov['N']} but the pair has N={N}")
G = {int(k): v for k, v in res["G"].items()}
rho = {int(k): v for k, v in res["rho"].items()}
covmap = {int(r["kprime"]): float(r["coverage"]) for r in cov["rows"]}
M_S, M_T = float(res["reference"]["M_S"]), float(res["reference"]["M_T"])
### before min(G): an empty G (a malformed or hand-written measure-only result) would
### otherwise raise "min() arg is an empty sequence" here, 50 lines before the readable
### no-common-depth error below ever runs
if not G or not covmap:
    sys.exit(f"no window depth measured: G has {sorted(G)}, coverage has {sorted(covmap)}. "
             f"Re-run the pair with --measure-only under its own config.")
window = int(res.get("window") or cfg.get("window") or min(G))
rounds = int(cfg.get("rounds", 100))
eps_list = [float(e) for e in (cfg.get("eps_sensitivity") or [cfg.get("eps", 0.01)])]
ratios = cfg.get("traffic_ratio", [1.0])
ratios = [float(t) for t in (ratios if isinstance(ratios, list) else [ratios])]
max_rounds = int(cfg.get("max_rounds_factor", 2) * rounds)
target_frac = float(cfg.get("arrival_target_frac", 0.99))

n_eval = int(res.get("n_eval", 0))
n_pool = int(res.get("n_traffic_pool", 0))
n_servable = n_eval + n_pool

# The per-evaluation-query draw probabilities. These CANNOT be reconstructed from the result's
# counts: the held-out set is a seeded permutation of the judged queries, not the first n_eval
# rows, and the stream's distribution zeroes the secondary set and renormalises. Under a uniform
# workload every choice of n_eval rows gives the same answer, so guessing was invisible there and
# wrong everywhere else. run_pair.py now records the exact vector as result["arrival"].
arrival = res.get("arrival") or {}
if arrival.get("eval_weights"):
    w = np.asarray(arrival["eval_weights"], float)
    eval_idx = np.arange(len(w))
    n_servable = int(arrival.get("n_servable", n_servable))
    note_w = (f"exact per-evaluation-query draw probabilities from the result "
              f"(workload={arrival.get('workload', '?')!r}, {len(w)} servable evaluation queries"
              + (f", {arrival['n_eval_zero_weight']} dropped at weight 0"
                 if arrival.get("n_eval_zero_weight") else "") + ")")
elif a.eval_qids and a.weights:
    full = np.asarray(np.load(a.weights), float)
    ev = np.loadtxt(a.eval_qids, dtype=np.int64, ndmin=1)
    bad = ev[(ev < 0) | (ev >= len(full))]
    if len(bad):
        sys.exit(f"--eval-qids has {len(bad)} index/indices outside [0, {len(full)})")
    tot = float(full.sum())
    if tot <= 0:
        sys.exit(f"--weights {a.weights} sums to {tot}: the workload has no arrivals to predict")
    w = full[np.unique(ev)] / tot
    eval_idx = np.arange(len(w))
    note_w = f"weights from {a.weights} restricted to the queries in {a.eval_qids}"
else:
    workload = cfg.get("workload", res.get("config", {}).get("workload", "uniform"))
    if workload != "uniform" or a.weights:
        sys.exit(
            "refusing to guess the arrival distribution for a non-uniform workload.\n"
            f"  workload={workload!r}, result has no 'arrival' block"
            f"{', --weights given' if a.weights else ''}.\n"
            "  Re-run the measure-only pass with this version of run_pair.py (it records the "
            "exact per-query probabilities), or pass --eval-qids together with --weights.\n"
            "  Guessing would take the FIRST n_eval rows of weights.npy, which are not the "
            "held-out evaluation queries, and every arrival prediction would be wrong.")
    if not n_eval:
        sys.exit("result has n_eval = 0: there is no evaluation population to predict arrivals for")
    w = np.full(n_eval, 1.0 / max(n_servable, 1))
    eval_idx = np.arange(len(w))
    note_w = ("uniform weights assumed: workload='uniform', so each of the "
              f"{n_servable} servable queries is drawn with probability 1/{n_servable} "
              "(the result predates the recorded 'arrival' block)")

b = math.ceil(N / rounds)
depths = sorted(set(G) & set(covmap))
if not depths:
    sys.exit(f"no window depth measured in both files: G has {sorted(G)}, coverage has {sorted(covmap)}")

reach = []
for e in eps_list:
    ok = [k for k in depths if G[k] <= e]
    kstar = min(ok) if ok else None
    reach.append({"eps": e, "reaches_at": ok, "optimal_kprime": kstar,
                  "ceiling_frac": (1.0 - covmap[kstar]) if kstar else None,
                  "predicted_debt_window": (G[kstar] if kstar else None)})

per_ratio = []
for tr in ratios:
    stream_len = int(cfg.get("stream_len") or max(1, int(round(tr * b / window))))
    # weighted, because every quality number is workload-weighted and the runner gates on the
    # weight. Under skew the count-based figure is orders of magnitude larger (the tail is
    # almost never drawn), so it is reported next to it rather than used.
    r99 = expected_arrival_rounds(w, eval_idx, stream_len, target_frac, weighted=True)
    r99_count = expected_arrival_rounds(w, eval_idx, stream_len, target_frac)
    wt = w[eval_idx]
    frac_at = (lambda T: float((wt * (1.0 - np.power(1.0 - wt, max(T, 0) * stream_len))).sum()
                               / wt.sum()) if wt.sum() > 0 else float("nan"))
    # share of arrivals that land before background completes (= before round `rounds`)
    f_end = frac_at(min(r99, max_rounds) if math.isfinite(r99) else max_rounds)
    # share of the arrivals that come before the background finishes; if every arrival is in by
    # then (r99 < rounds) the ratio exceeds 1, and debt(S0) would exceed the quality gap itself
    pre = min(1.0, frac_at(rounds) / f_end) if f_end > 0 else float("nan")
    f_bg = min(1.0, (min(r99, max_rounds) if math.isfinite(r99) else max_rounds) / rounds)
    f_bg_band = [min(1.0, max(0.0, (min(r99, max_rounds) + d) / rounds)) for d in (-1, 1)] \
        if math.isfinite(r99) else [1.0, 1.0]
    row = {"traffic_ratio": tr, "stream_len": stream_len, "background_per_round": b,
           "arrival_rounds_for_target_frac": (r99 if math.isfinite(r99) else None),
           "arrival_rounds_for_target_frac_by_count": (r99_count if math.isfinite(r99_count) else None),
           "arrival_rounds_exceeds_max_rounds": not (math.isfinite(r99) and r99 <= max_rounds),
           "served_frac_at_max_rounds": frac_at(max_rounds),
           "share_of_arrivals_before_completion": pre,
           "f_bg_at_last_arrival": f_bg, "f_bg_band": sorted(f_bg_band),
           "predicted_debt_S0": (M_T - M_S) * pre, "by_depth": []}
    for k in depths:
        c = covmap[k]
        pred = 1.0 - (1.0 - c) * (1.0 - f_bg)
        band = sorted(1.0 - (1.0 - c) * (1.0 - fb) for fb in f_bg_band)
        row["by_depth"].append({
            "kprime": k, "coverage": c, "G": G[k], "rho": rho.get(k),
            "predicted_C_serve_frac": pred, "predicted_C_serve_frac_band": band,
            "predicted_saving_vs_rebuild": 1.0 - pred,
            "predicted_per_query_cost_at_f": {str(f): (1.0 - f) * k for f in (0.0, 0.25, 0.5, 0.75)},
        })
    per_ratio.append(row)

out = {"pair": res.get("name"), "N": N, "window": window, "M_S": M_S, "M_T": M_T,
       "coverage_source": cov.get("name"), "coverage_file": os.path.abspath(a.coverage),
       "result_file": os.path.abspath(a.result), "config_file": os.path.abspath(a.config) if a.config else None,
       "rounds": rounds, "max_rounds": max_rounds, "n_eval": n_eval, "n_servable": n_servable,
       "weights_note": note_w, "warm_threshold_v2_f": {str(k): 0.5 ** (1.0 / k) for k in depths},
       "reach": reach, "per_ratio": per_ratio,
       "how_to_check": [
           "C_serve_frac of the window strategies must fall inside predicted_C_serve_frac_band",
           "debt of a window strategy that reaches the target must be ~ G(K') of its depth",
           "debt of S0/rand must be ~ predicted_debt_S0",
           "per_query_cost curve must follow (1 - f) * K' while the strategy is window-serving",
           "a window strategy is admissible only where G(K') <= eps (see reach)",
       ]}

print(f"pair {out['pair']}  N={N}  window={window}  M_S={M_S:.4f}  M_T={M_T:.4f}")
print(f"\nreach condition (a window strategy is admissible iff G(K') <= eps):")
for r in reach:
    print(f"  eps={r['eps']:<6g} optimal K'={r['optimal_kprime']}  ceiling="
          f"{'' if r['ceiling_frac'] is None else format(r['ceiling_frac'], '.4f')}  "
          f"predicted window debt={'' if r['predicted_debt_window'] is None else format(r['predicted_debt_window'], '.4f')}")
for row in per_ratio:
    print(f"\ntraffic_ratio={row['traffic_ratio']:g}  stream_len={row['stream_len']}  "
          f"arrival rounds for {target_frac:.0%} of the workload weight="
          f"{row['arrival_rounds_for_target_frac']}  "
          f"(by query count: {row['arrival_rounds_for_target_frac_by_count']})  "
          f"(max_rounds={max_rounds})")
    if row["arrival_rounds_exceeds_max_rounds"]:
        print(f"  !! only {row['served_frac_at_max_rounds']:.1%} of the evaluation population can be "
              f"served inside max_rounds: this setting reports no winner")
    print(f"  f_bg at last arrival={row['f_bg_at_last_arrival']:.3f} band={row['f_bg_band']}  "
          f"predicted debt(S0)={row['predicted_debt_S0']:.4f}")
    print(f"  {'K-prime':>8s} {'coverage':>9s} {'G':>8s} {'pred C_serve/N':>15s} {'band':>21s} {'saving':>8s}")
    for d in row["by_depth"]:
        print(f"  {d['kprime']:8d} {d['coverage']:9.4f} {d['G']:8.4f} {d['predicted_C_serve_frac']:15.4f} "
              f"[{d['predicted_C_serve_frac_band'][0]:.4f},{d['predicted_C_serve_frac_band'][1]:.4f}]  "
              f"{d['predicted_saving_vs_rebuild']:8.4f}")
print(f"\n{note_w}")
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out, "- commit this before running the replay")
