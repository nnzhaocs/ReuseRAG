"""Run every measurement for one pair and write a JSON result."""
import json
import math
import time

import numpy as np

from . import heat, probe
from . import winner as winner_mod
from .measure import (reference_qualities, residual_miss, secondary_frontier,
                      window_gap)
from .pair import Pair
from .simulate import (WINDOW_STRATEGIES, Replay, auc_over, expected_arrival_rounds,
                       summarize, summarize_arrivals)
from .winner import pick_winner

DEFAULT_CFG = {
    "K": 10,
    "kprime_frac": [0.0005, 0.001, 0.002, 0.005],
    "window_frac": 0.0005,          # window depth used by the strategies
    "primary": {"metric": "recall", "thr": 1},
    "secondary": [{"metric": "ndcg", "thr": 1}],
    "secondary_eval_qids": None,    # path to a file of query indices (e.g. TREC-DL) or None
    "eps": 0.01,
    "delta": 0.01,                  # PRE-REGISTER before running any pair
    "workload": "uniform",
    # --- query roles (no evaluation leak) ---
    "eval_frac": 0.5,               # share of primary-judged queries held out for evaluation
    "repeat_rate": [0.0],           # P(traffic draw is a re-asked eval query); list = grid
    # --- traffic vs encoding budget ---
    "traffic_ratio": [1.0],         # (stream_len * window) / background budget per round; list = grid
    "primary_setting": {"traffic_ratio": 1.0, "repeat_rate": 0.0},  # PRE-REGISTER
    "rounds": 100,                  # background budget per round = N / rounds
    "max_rounds_factor": 2,         # stop non-completing runs (pure demand) after factor*rounds
    "stream_len": None,             # explicit override of the traffic_ratio rule (legacy)
    "adapter_fit": "docs",          # S1/S3 adapter training pairs: docs | queries | both
    "mapped_query_margin": 0.05,
    "replay": True,                 # False = measure only (M_S, M_T, M_A, kappa, rho, G): calibration    # screening flag if M_A > M_S + margin (see warnings)
    "orders": ["rand", "size", "demand", "signal"],
    "strategies": ["S0", "S1", "S2", "S3", "S4", "S5"],
    "mode": "sync",
    "adapter": "procrustes",
    "probe_budgets_frac": [0.001, 0.005, 0.01],
    "probe_queries": 200,
    "strat_per_query": None,        # stratum sample per probe query (None = K'; see probe.py)
    "seed": 0,
    # --- replay v3 (2026-10-29): first-arrival measurement ---
    # "none" = v2: state-based checkpoints every round, evaluation queries served only via
    #          repeat_rate. Kept so the v2 configs reproduce exactly.
    # "first_service" = v3: every evaluation query is measured once, when it is first served;
    #          window repairs are charged to C; completion is an admissibility constraint and
    #          the objective is the quality lost during the migration ('debt').
    "eval_arrival": "none",
    "objective": "debt",            # 'debt' (pre-registered primary) | 'c_serve' (diagnostic)
    "tau": {"debt": 0.0, "c_serve_frac": 0.0},  # equivalence bands, in each objective's units
    "eps_sensitivity": None,        # list of eps values for the c_serve winner (debt is eps-free)
    "band_tiebreak": "simplicity",  # how an equivalence-band tie is broken: simplicity | cost
    "min_eval_served": 0.9,         # a setting reports no winner below this served fraction -- the
                                    # share of the EVALUATION set's weight that arrived, not of all traffic
    "arrival_target_frac": 0.99,    # the fraction used by the pre-run arrival-rounds estimate
    "per_query_sample": 200,        # read-only per-query cost curve sample size
    "per_query_every": None,        # sample it every N rounds (default: ~200 points per run)
}


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _thin(seq, n):
    """At most n points of seq, evenly strided, always keeping the last one."""
    seq = list(seq)
    if len(seq) <= n:
        return seq
    step = int(math.ceil(len(seq) / n))
    out = seq[::step]
    if out[-1] != seq[-1]:
        out.append(seq[-1])
    return out


def _arrival_info(p, warnings=None):
    """The exact per-evaluation-query draw probabilities the v3 stream uses, recorded in every
    result JSON.

    `scripts/predict.py` needs these to estimate the arrival rounds, and it cannot reconstruct
    them: the held-out evaluation set is a seeded permutation of the judged queries, not the
    first `n_eval` rows, and the stream's distribution additionally zeroes the secondary set and
    renormalises. Under `workload: "uniform"` any `n_eval` rows happen to give the same answer,
    which is why the error was invisible there; under a weighted workload (ORCAS) every arrival
    prediction would be computed from the wrong queries' weights.
    """
    try:
        w = p.all_query_weights()
    except ValueError:
        return None
    ev = np.array(sorted(int(q) for q in p.eval_q if w[int(q)] > 0), np.int64)
    dropped = int(len(p.eval_q) - len(ev))
    if dropped and warnings is not None:
        warnings.append(
            f"{dropped} of {len(p.eval_q)} evaluation queries have draw probability 0 under this "
            f"workload, so they can never arrive and are excluded from the measurement "
            f"(n_expected = {len(ev)}). The served fraction is reported against that number, not "
            f"against n_eval.")
    return {"eval_q": ev.tolist(), "eval_weights": [float(w[q]) for q in ev],
            "n_servable": int((w > 0).sum()), "n_eval_zero_weight": dropped,
            "workload": p.cfg.get("workload", "uniform")}


def _replay_v3(p, rp, cfg, ref, hg, signal, name, log, warnings):
    """Replay v3: first-arrival measurement, 'debt' objective, completion as a constraint.

    Returns (settings, primary_setting_row, final_state). See simulate.run_arrivals for the
    semantics and DESIGN.md "Replay protocol v3" for why the objective is quality lost during
    the migration rather than encodings to reach the target.
    """
    N = p.N
    b = int(math.ceil(N / cfg["rounds"]))
    eps, delta = cfg["eps"], cfg["delta"]
    objective = cfg.get("objective", "debt")
    if objective not in winner_mod.OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(winner_mod.OBJECTIVES)}, got {objective!r}")
    if cfg.get("mode", "sync") != "sync":
        raise ValueError("eval_arrival='first_service' requires mode='sync': an arriving query "
                         "must have its window repaired inline to be served at target quality")
    bands = cfg.get("tau") or {}
    band = float(bands.get(winner_mod.OBJECTIVES[objective][1], 0.0))
    tiebreak = cfg.get("band_tiebreak", "simplicity")
    if tiebreak not in winner_mod.BAND_TIEBREAKS:
        raise ValueError(f"band_tiebreak must be one of {winner_mod.BAND_TIEBREAKS}, "
                         f"got {tiebreak!r}")
    max_rounds = int(cfg.get("max_rounds_factor", 2) * cfg["rounds"])
    arrival_w = p.all_query_weights()
    can_arrive = np.array(sorted(int(q) for q in rp.eval_q if arrival_w[int(q)] > 0), np.int64)
    if len(can_arrive) == 0:
        raise ValueError("no evaluation query can be served under this workload, so nothing "
                         "can be measured at first arrival")
    ratios = [float(t) for t in _as_list(cfg["traffic_ratio"])]
    prim_ratio = float(cfg["primary_setting"]["traffic_ratio"])
    if prim_ratio not in ratios:
        raise ValueError(f"primary_setting traffic_ratio {prim_ratio} is not in the grid {ratios}")
    target_frac = float(cfg.get("arrival_target_frac", 0.99))
    min_served = float(cfg.get("min_eval_served", 0.9))
    settings = []
    for tr_ratio in ratios:
        stream_len = int(cfg["stream_len"] or max(1, int(round(tr_ratio * b / p.Kw))))
        est = expected_arrival_rounds(arrival_w, can_arrive, stream_len, target_frac, weighted=True)
        est_count = expected_arrival_rounds(arrival_w, can_arrive, stream_len, target_frac)
        note = (f"ratio={tr_ratio:g} stream_len={stream_len}: ~{est} rounds to serve "
                f"{target_frac:.0%} of the workload weight over {len(can_arrive)} evaluation "
                f"queries (~{est_count} rounds for that share of the query COUNT); max_rounds="
                f"{max_rounds}, background completes at round {cfg['rounds']}")
        log(f"[{name}] {note}")
        if not math.isfinite(est) or est > max_rounds:
            warnings.append(
                "ARRIVAL SHORTFALL: " + note + ". Draws are with replacement, so this is a "
                "weighted coupon-collector quantity, not n_eval/stream_len: raise traffic_ratio "
                "or max_rounds_factor, or this setting cannot serve its evaluation population.")
        raws, summaries = {}, {}
        for st in cfg["strategies"]:
            if st in ("S4", "S5") and p.bm25 is None:
                continue
            orders = ["rand"] if st in ("S0", "S1") else cfg["orders"]
            for od in orders:
                if od == "signal" and not hg.get("gate_open"):
                    continue  # pre-registered gate: O-signal runs only if the AUC CI excludes 0.5
                raw = rp.run_arrivals(st, od, cfg["rounds"], stream_len, cfg["seed"], signal,
                                      max_rounds=max_rounds, arrival_w=arrival_w,
                                      per_query_sample=cfg.get("per_query_sample", 200),
                                      per_query_every=cfg.get("per_query_every"))
                if raw is None:
                    continue
                combo = f"{st}/{od}"
                raws[combo] = raw
                summaries[combo] = summarize_arrivals(raw, "primary", ref["M_S"], ref["M_T"], eps, N)
        if not summaries:
            raise ValueError("no strategy/order combination could be run")
        # ONE common range for every area, so a run that stops earlier is not credited with a
        # shorter integration range (this is what made 'demand' incomparable in v2, while the
        # area was still used to break ties).
        lo = [s["auc_range_frac"][0] for s in summaries.values() if math.isfinite(s["auc_range_frac"][0])]
        hi = [s["auc_range_frac"][1] for s in summaries.values() if math.isfinite(s["auc_range_frac"][1])]
        c_lo, c_hi = (max(lo) if lo else float("nan")), (min(hi) if hi else float("nan"))
        common = math.isfinite(c_lo) and math.isfinite(c_hi) and c_hi > c_lo
        for s in summaries.values():
            pts = s.pop("auc_points", [])
            s["auc"] = auc_over(pts, N, c_hi * N, c_lo * N) if common else float("nan")
            s["auc_common_range_frac"] = [c_lo, c_hi] if common else [float("nan")] * 2
        seqs = {c: [a[0] for a in r["arrivals"]] for c, r in raws.items()}
        n_min = min((len(v) for v in seqs.values()), default=0)
        ref_seq = next(iter(seqs.values()))[:n_min]
        identical = all(v[:n_min] == ref_seq for v in seqs.values())
        if not identical:
            warnings.append(f"ARRIVAL SEQUENCES DIFFER at ratio={tr_ratio:g}: combinations were "
                            f"not compared on the same queries (this is a pipeline bug)")
        served_frac = float(np.nanmax([s["eval_served_frac"] for s in summaries.values()]))
        served_wfrac = float(np.nanmax([s["eval_served_weight_frac"] for s in summaries.values()]))
        reason = None
        winner, bandset = winner_mod.pick_winner_v3(summaries, ref["M_S"], delta, objective,
                                                    band, return_band=True,
                                                    band_tiebreak=tiebreak)
        # gate on the WEIGHT served, which is what the quality numbers are averaged over. Gating
        # on the query count would reject every skewed workload (ORCAS, the Zipf code sweep): at
        # a 35,000x weight spread, 99% of the weight arrives by round ~100 while 99% of the count
        # takes ~30,000 rounds, because the tail is almost never drawn.
        if served_wfrac < min_served:
            reason = (f"only {served_wfrac:.1%} of the evaluation workload weight was served "
                      f"inside max_rounds ({served_frac:.1%} of the queries by count; "
                      f"pre-registered minimum {min_served:.0%} by weight)")
            warnings.append(f"NO WINNER at ratio={tr_ratio:g}: {reason}")
            winner, bandset = None, []
        eps_rows = []
        for e in _as_list(cfg.get("eps_sensitivity") or [eps]):
            sums_e = {c: summarize_arrivals(r, "primary", ref["M_S"], ref["M_T"], float(e), N)
                      for c, r in raws.items()}
            for s in sums_e.values():
                s.pop("auc_points", None)
            w_e = winner_mod.pick_winner_v3(sums_e, ref["M_S"], delta, "c_serve",
                                            float(bands.get("c_serve_frac", 0.0)),
                                            band_tiebreak=tiebreak)
            eps_rows.append({"eps": float(e), "winner_c_serve": w_e,
                             "C_serve_frac": sums_e[w_e]["C_serve_frac"] if w_e else None,
                             # admissible combinations only: 'demand' can reach the target while
                             # never completing, and counting it here would overstate the row
                             "n_reaching_target": sum(1 for s in sums_e.values()
                                                      if s["reaches_target"] and s["completed"])})
        for combo, s in sorted(summaries.items()):
            log(f"[{name}] ratio={tr_ratio:g} {combo:18s} debt={s['debt']:.4f} "
                f"served={s['served_mean']:.4f} C_serve/N={s['C_serve_frac']:.3f} "
                f"completed={s['completed']} repairs/q={s['repairs_per_query_mean']:.1f}")
        tied = winner_mod.tied_with(summaries, winner, objective)
        clash = winner_mod.cost_basis_conflict(summaries, bandset) if tiebreak == "cost" else []
        if clash:
            warnings.append(
                f"COST TIE-BREAK ACROSS BASES at ratio={tr_ratio:g}: the band contains rows whose "
                f"C_serve means different things ({', '.join(clash)}), and band_tiebreak='cost' "
                f"ordered them against each other anyway. A 'no_repair_needed' row reports how far "
                f"its background got, not work it did, so its small C_serve is a quality "
                f"concession inside tau rather than a cheaper migration. The ordering is "
                f"pre-registered and unchanged; read the winner with C_serve_basis beside it.")
        log(f"[{name}] ratio={tr_ratio:g} winner={winner} band={len(bandset)} tied={len(tied)} "
            f"non_completing={sorted(c for c, s in summaries.items() if not s['completed'])}")
        settings.append({
            "traffic_ratio": tr_ratio, "repeat_rate": None, "stream_len": stream_len,
            "background_per_round": b, "est_arrival_rounds": est, "max_rounds": max_rounds,
            "summaries": summaries, "winner": winner, "winner_band": bandset,
            "winner_ties": bandset,  # name kept so summarize.py/smoke.py read both v2 and v3
            # within tau (winner_ties/winner_band) vs identical to the winner (tied_exact):
            # the second is what says how much the reported combination was actually determined
            "tied_exact": tied, "tied_exact_families": sorted({c.split("/")[0] for c in tied}),
            "cost_basis_conflict": clash,
            "non_completing": sorted(c for c, s in summaries.items() if not s["completed"]),
            "band_tiebreak": tiebreak,
            "arrivals_identical": bool(identical), "n_arrival_prefix_checked": int(n_min),
            "eval_served_frac": served_frac, "eval_served_weight_frac": served_wfrac,
            "est_arrival_rounds_count": est_count, "no_winner_reason": reason,
            "eps_sensitivity": eps_rows,
            "trajectories": {c: _thin(r["traj"], 400) for c, r in raws.items()},
            "per_query_cost": {c: _thin(r["per_query_cost"], 400) for c, r in raws.items()},
            "arrival_cost": ({c: [[a[1], a[2]] for a in _thin(r["arrivals"], 1000)]
                              for c, r in raws.items()} if tr_ratio == prim_ratio else {}),
        })
    prim = next(x for x in settings if x["traffic_ratio"] == prim_ratio)
    ones = np.ones(N, bool)
    final_state = {st: rp.checkpoint(st, ones, "sync", "state") for st in cfg["strategies"]
                   if not (st in ("S4", "S5") and p.bm25 is None)}
    return settings, prim, final_state


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def run_pair(pair_dir, cfg_overrides=None, name=None, log=print):
    cfg = dict(DEFAULT_CFG)
    cfg.update(cfg_overrides or {})
    t0 = time.time()
    p = Pair(pair_dir, cfg)
    if "window" not in cfg:
        cfg["window"] = max(cfg["K"], int(round(cfg["window_frac"] * p.N)))
        p.cfg = cfg
    p.precompute()
    log(f"[{name}] N={p.N} Q={p.Q} eval={len(p.eval_q)} traffic_pool={len(p.traffic_pool)} window={p.Kw} "
        f"kappa={p.compat['kappa']:.3f} cka={p.compat['cka']:.3f} ({time.time()-t0:.1f}s)")

    prim = cfg["primary"]
    metrics = [("primary", prim["metric"], prim["thr"], p.eval_q)]
    sec_q = p.secondary_q if len(p.secondary_q) else p.eval_q
    for i, s in enumerate(cfg.get("secondary", [])):
        metrics.append((f"secondary{i}", s["metric"], s["thr"], sec_q))

    ref = reference_qualities(p, prim["metric"], prim["thr"])
    ks = p.kprime_list()
    rho = residual_miss(p, ks, "src")
    rho_map = residual_miss(p, ks, "map")
    G = window_gap(p, ks, ref["M_T"], prim["metric"], prim["thr"], "src")
    # the same frontier on the deeply-judged secondary set, to separate a real cascade effect
    # from judgment sparsity as the explanation for G < 0 (see measure.secondary_frontier)
    sec_front = {}
    # the FIRST secondary metric only: the frontier exists to answer one question (is G < 0
    # judgment sparsity?) and that needs the deeply-judged metric, which is secondary[0] in
    # every shipped config. Loop here if a config ever needs the frontier on several.
    sec_cfg = (cfg.get("secondary") or [{}])[0]
    if sec_cfg.get("metric"):
        sec_front = secondary_frontier(p, ks, sec_cfg["metric"], sec_cfg.get("thr", 1))
        if sec_front:
            log(f"[{name}] secondary ({sec_front['queries']} deeply-judged queries, "
                f"{sec_front['metric']}@{p.cfg['K']}): M_T={sec_front['M_T']:.4f} "
                f"G={ {k: round(v, 4) for k, v in sec_front['G'].items()} }")
            neg_p = sum(1 for v in G.values() if v < 0)
            neg_s = sum(1 for v in sec_front["G"].values() if v < 0)
            if neg_p and not neg_s:
                log(f"[{name}] NOTE G < 0 at {neg_p} depth(s) on the sparse primary set but at "
                    f"none on the deeply-judged set -- consistent with judgment sparsity rather "
                    f"than a cascade effect. This is the comparison the paper turns on.")
    log(f"[{name}] M_S={ref['M_S']:.4f} M_T={ref['M_T']:.4f} M_A={ref['M_A']:.4f} rho={rho}")

    warnings = []
    if p.leaky:
        warnings.append("LEAKY: eval_frac=0, evaluation queries are also the traffic stream")
    if ref["M_A"] > ref["M_S"] + cfg.get("mapped_query_margin", 0.02):
        warnings.append(
            f"SOURCE QUERY ENCODING SUSPECT (screening flag): target queries mapped into the "
            f"source space (M_A={ref['M_A']:.4f}) beat the source model's own queries "
            f"(M_S={ref['M_S']:.4f}) on the source document index. A much stronger target can "
            f"also cause this; confirm with encode.py's padding check and the golden check "
            f"before treating the source encoding as broken")
    arrival = _arrival_info(p, warnings)  # before the log loop, so its warning is printed too
    for w_ in warnings:
        log(f"[{name}] WARNING {w_}")

    if not cfg.get("replay", True):
        res = {"name": name or pair_dir, "config": cfg, "meta": p.meta, "measure_only": True,
               "N": p.N, "Q": p.Q, "n_eval": int(len(p.eval_q)), "window": p.Kw,
               "n_traffic_pool": int(len(p.traffic_pool)), "leaky": bool(p.leaky),
               "compat": p.compat, "adapter_info": p.adapter_info, "reference": ref,
               "rho": rho, "rho_mapped": rho_map, "G": G, "secondary_frontier": sec_front,
               "features": {"kappa": p.compat["kappa"], "rho": rho.get(p.Kw, rho[min(rho)])},
               "arrival": arrival,
               "warnings": warnings, "settings": [], "summaries": {}, "winner": None,
               "winner_ties": [], "trajectories": {}, "M_S": ref["M_S"], "delta": cfg["delta"],
               "seconds": time.time() - t0}
        log(f"[{name}] measure-only (replay=false): G={G}")
        return _jsonable(res)

    prim_set = cfg["primary_setting"]
    if "traffic_ratio" not in prim_set:
        raise ValueError(f"primary_setting is missing 'traffic_ratio' (got {sorted(prim_set)}): "
                         f"it names one cell of the grid. DEFAULT_CFG's is "
                         f"{{'traffic_ratio': 1.0, 'repeat_rate': 0.0}}.")
    if "replay_version" in cfg:
        # it is an OUTPUT field, and setting it looks like it should select v3 while doing
        # nothing -- you would silently get v2, whose objective is C_eps rather than debt
        raise ValueError("replay_version is reported in the result, not set in the config. "
                         "Select the v3 replay with eval_arrival='first_service'.")
    arrival_mode = cfg.get("eval_arrival", "none")
    if arrival_mode not in ("none", "first_service"):
        raise ValueError(f"eval_arrival must be 'none' or 'first_service', got {arrival_mode!r}")
    hg_w = (p.all_query_weights() if arrival_mode == "first_service"
            else p.traffic_weights(prim_set["repeat_rate"]))
    hg = heat.heat_gate(p, seed=cfg["seed"], weights=hg_w)
    signal = hg.pop("signal_scores", None)

    rp = Replay(p, metrics)
    b = int(math.ceil(p.N / cfg["rounds"]))

    if arrival_mode == "first_service":
        settings, prim, final_state = _replay_v3(p, rp, cfg, ref, hg, signal, name, log, warnings)
        kp_probe = p.Kw
        pr = probe.evaluate_probe(p, kp_probe, cfg["probe_budgets_frac"],
                                  n_queries=cfg["probe_queries"], seed=cfg["seed"],
                                  strat_per_query=cfg.get("strat_per_query"))
        res = {
            "name": name or pair_dir, "config": cfg, "meta": p.meta,
            "N": p.N, "Q": p.Q, "n_eval": int(len(p.eval_q)), "window": p.Kw,
            "n_traffic_pool": int(len(p.traffic_pool)), "leaky": bool(p.leaky),
            "compat": p.compat, "adapter_info": p.adapter_info, "reference": ref,
            "rho": rho, "rho_mapped": rho_map, "G": G, "secondary_frontier": sec_front,
            "features": {"kappa": p.compat["kappa"], "rho": rho.get(p.Kw, rho[min(rho)])},
            "heat_gate": hg, "warnings": warnings, "primary_setting": prim_set,
            "arrival": arrival,
            "replay_version": 3, "objective": cfg.get("objective", "debt"),
            "settings": settings, "final_state": final_state,
            "summaries": prim["summaries"], "winner": prim["winner"],
            "winner_ties": prim["winner_ties"], "trajectories": prim["trajectories"],
            "lopo_penalty": {"debt": ref["M_T"] - ref["M_S"], "c_serve": 1.0},
            "M_S": ref["M_S"], "delta": cfg["delta"], "probe": pr,
            "seconds": time.time() - t0,
        }
        log(f"[{name}] winner={prim['winner']} band={len(prim['winner_ties'])}  "
            f"({res['seconds']:.1f}s)")
        return _jsonable(res)

    grid = [(float(t), float(r)) for t in _as_list(cfg["traffic_ratio"])
            for r in _as_list(cfg["repeat_rate"])]
    # only the v2 path indexes the repeat_rate axis; v3 weights arrivals by the whole query
    # set and never reads it, which is why the check for it lives here and not above
    if "repeat_rate" not in prim_set:
        raise ValueError(f"primary_setting needs 'repeat_rate' for the v2 replay "
                         f"(got {sorted(prim_set)}); it is a grid axis here. The v3 replay "
                         f"(eval_arrival='first_service') does not use it.")
    if (float(prim_set["traffic_ratio"]), float(prim_set["repeat_rate"])) not in grid:
        raise ValueError(f"primary_setting {prim_set} is not in the traffic_ratio x repeat_rate grid")
    settings = []
    for tr_ratio, rr in grid:
        stream_len = cfg["stream_len"] or max(1, int(round(tr_ratio * b / p.Kw)))
        tw = p.traffic_weights(rr)
        runs, summaries = {}, {}
        for st in cfg["strategies"]:
            if st in ("S4", "S5") and p.bm25 is None:
                continue
            orders = ["rand"] if st in ("S0", "S1") else cfg["orders"]
            for od in orders:
                if od == "signal" and not hg.get("gate_open"):
                    continue  # pre-registered gate: O-signal runs only if the AUC CI excludes 0.5
                tr = rp.run(st, od, cfg["mode"], cfg["rounds"], stream_len, cfg["seed"], signal,
                            max_rounds=int(cfg.get("max_rounds_factor", 2) * cfg["rounds"]),
                            how=cfg.get("checkpoint", "state"), traffic_w=tw)
                if tr is None:
                    continue
                combo = f"{st}/{od}"
                runs[combo] = tr
                summaries[combo] = summarize(tr, "primary", ref["M_S"], ref["M_T"], cfg["eps"], p.N)
                se = getattr(rp, "sync_encodings", [])
                if st in WINDOW_STRATEGIES and cfg["mode"] == "sync" and se:
                    summaries[combo]["sync_enc_per_query_mean"] = float(np.mean(se))
                s = summaries[combo]
                log(f"[{name}] ratio={tr_ratio:g} repeat={rr:g} {combo:10s} "
                    f"C_eps/N={s['C_eps_frac']:.3f} floor={s['floor']:.4f} final={s['final']:.4f}")
        winner, ties = pick_winner(summaries, ref["M_S"], cfg["delta"], p.N, return_ties=True)
        log(f"[{name}] ratio={tr_ratio:g} repeat={rr:g} winner={winner} ties={len(ties)}")
        settings.append({"traffic_ratio": tr_ratio, "repeat_rate": rr, "stream_len": stream_len,
                         "background_per_round": b, "summaries": summaries, "winner": winner,
                         "winner_ties": ties,
                         "trajectories": {k: [(c, m) for c, m in v] for k, v in runs.items()}})
    prim = next(x for x in settings if (x["traffic_ratio"], x["repeat_rate"]) ==
                (float(prim_set["traffic_ratio"]), float(prim_set["repeat_rate"])))
    winner, ties = prim["winner"], prim["winner_ties"]

    kp_probe = p.Kw
    pr = probe.evaluate_probe(p, kp_probe, cfg["probe_budgets_frac"],
                              n_queries=cfg["probe_queries"], seed=cfg["seed"],
                              strat_per_query=cfg.get("strat_per_query"))
    feat_k = p.Kw
    res = {
        "name": name or pair_dir,
        "config": cfg,
        "meta": p.meta,
        "N": p.N, "Q": p.Q, "n_eval": int(len(p.eval_q)), "window": p.Kw,
        "n_traffic_pool": int(len(p.traffic_pool)), "leaky": bool(p.leaky),
        "compat": p.compat,            # Procrustes on documents: the map coordinate
        "adapter_info": p.adapter_info,  # what S1/S3 used
        "reference": ref,
        "rho": rho, "rho_mapped": rho_map, "G": G, "secondary_frontier": sec_front,
        "features": {"kappa": p.compat["kappa"], "rho": rho.get(feat_k, rho[min(rho)])},
        "heat_gate": hg,
        "warnings": warnings,
        "arrival": arrival,
        "primary_setting": prim_set,
        "settings": settings,
        # primary-setting copies, so run_lopo.py and smoke.py keep working unchanged
        "summaries": prim["summaries"],
        "winner": winner,
        "winner_ties": ties,
        "trajectories": prim["trajectories"],
        "M_S": ref["M_S"], "delta": cfg["delta"],
        "probe": pr,
        "seconds": time.time() - t0,
    }
    log(f"[{name}] winner={winner} ties={len(ties)}  ({res['seconds']:.1f}s)")
    return _jsonable(res)


def save(res, path):
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
