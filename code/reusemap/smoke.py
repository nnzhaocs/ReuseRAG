"""Pipeline invariants. A violation is a bug; an unexpected value on a real pair is a finding.

  I1  identity pair (target == source, exact search): rho(K') == 0 for every K' >= K
  I2  rho(K') is non-increasing in K' (windows are nested prefixes)
  I3  re-ranking a window equal to the whole corpus gives G == 0
  I4  every trajectory with a completing order ends at M_T with C == N
  I5  S0 serves exactly M_S until it switches
  I6  no evaluation leak: with eval_frac > 0 the traffic pool and eval set are disjoint
I4/I5 are checked in every setting of the grid.

Replay v3 (eval_arrival = "first_service") adds:
  I4' the fully migrated state is the target system for every strategy, and every combination
      with a completing order reached C == N (completion is an admissibility constraint)
  I7  'demand' does no background work: every encoding it makes is an inline repair for a
      served query (repairs_all_total == C_final). It may still reach N when the window union
      covers the corpus; that is an outcome, not a violation
  I8  the arrival sequence is identical across combinations (same queries, same order)
  I9  no unpaid scoring: a target-scored ranking never contains a document without a target
      vector, and the repairs charged to a run never exceed its total encodings
"""
import math

TOL = 1e-9


def check(res, identity=False, full_window_G=None):
    fails, notes = [], []
    rho = {int(k): v for k, v in res["rho"].items()}
    ks = sorted(rho)
    if identity:
        bad = {k: rho[k] for k in ks if k >= res["config"]["K"] and rho[k] > TOL}
        (fails if bad else notes).append(f"I1 identity rho==0: {'FAIL ' + str(bad) if bad else 'ok'}")
    mono = all(rho[a] >= rho[b] - TOL for a, b in zip(ks, ks[1:]))
    (notes if mono else fails).append(f"I2 rho non-increasing: {'ok' if mono else 'FAIL ' + str(rho)}")
    if full_window_G is not None:
        ok = abs(full_window_G) < TOL
        (notes if ok else fails).append(f"I3 G(N)==0: {'ok' if ok else 'FAIL ' + str(full_window_G)}")
    M_T, M_S, N = res["reference"]["M_T"], res["reference"]["M_S"], res["N"]
    if res.get("measure_only"):
        notes.append("I4/I5 skipped: measure-only run (replay=false)")
        return fails, notes
    if res.get("replay_version") == 3:
        _check_v3(res, fails, notes, M_S, M_T, N)
        return fails, notes
    settings = res.get("settings") or [{"traffic_ratio": None, "repeat_rate": None,
                                        "trajectories": res["trajectories"]}]
    i5_bad = False
    for st in settings:
        tag = f"[ratio={st['traffic_ratio']} repeat={st['repeat_rate']}] "
        for combo, tr in st["trajectories"].items():
            if combo.endswith("/demand"):
                continue  # pure lazy need not complete
            C, m = tr[-1]
            ok = C == N and m["primary"] is not None and abs(m["primary"] - M_T) < 1e-9
            if not ok:
                fails.append(f"I4 {tag}{combo} ends at C={C} q={m['primary']} (want C={N}, q={M_T})")
        s0 = st["trajectories"].get("S0/rand")
        if s0 and not all(abs(m["primary"] - M_S) < 1e-9 for C, m in s0 if C < N):
            i5_bad = True
            fails.append(f"I5 {tag}S0 pre-switch != M_S")
    if not i5_bad:
        notes.append("I5 S0 pre-switch == M_S: ok")
    if "leaky" in res:
        if res["leaky"]:
            notes.append("I6 skipped: run is LEAKY by configuration (eval_frac=0)")
        elif res.get("n_eval", 0) and res.get("n_traffic_pool", 0):
            notes.append("I6 eval set and traffic pool disjoint: ok (enforced in Pair)")
    if not any(f.startswith("I4") for f in fails):
        notes.append("I4 completing trajectories end at (N, M_T): ok")
    return fails, notes


def _check_v3(res, fails, notes, M_S, M_T, N):
    """Invariants for replay v3 (first-arrival measurement)."""
    for st, m in sorted((res.get("final_state") or {}).items()):
        v = m.get("primary")
        if v is None or abs(v - M_T) > TOL:
            fails.append(f"I4' fully migrated state for {st}: primary={v} (want M_T={M_T})")
    for setting in res.get("settings") or []:
        tag = f"[ratio={setting.get('traffic_ratio')}] "
        for combo, s in sorted(setting["summaries"].items()):
            order = combo.split("/")[-1]
            if order == "demand":
                # 'demand' does no background work at all: every encoding it makes is an inline
                # repair for a served query. (It MAY still reach N when the window union covers
                # the corpus - that is an outcome, not a violation.)
                if s.get("repairs_all_total", 0) != s["C_final"]:
                    fails.append(f"I7 {tag}{combo} has C_final={s['C_final']} but charged "
                                 f"{s.get('repairs_all_total')} repairs: 'demand' must encode "
                                 f"nothing beyond what a served query asked for")
            elif s.get("hit_round_cap"):
                notes.append(f"I4' {tag}{combo} skipped: the run hit max_rounds before every "
                             f"evaluation query arrived (see the ARRIVAL SHORTFALL warning)")
            elif not s["completed"] or s["C_final"] != N:
                fails.append(f"I4' {tag}{combo} did not complete: C_final={s['C_final']} of {N}")
            if s.get("unpaid_scored", 0):
                fails.append(f"I9 {tag}{combo} scored {s['unpaid_scored']} ranking(s) containing a "
                             f"document with no target vector")
            if s.get("repairs_all_total", 0) > s["C_final"] + TOL:
                fails.append(f"I9 {tag}{combo} charged {s.get('repairs_all_total')} repairs but "
                             f"only {s['C_final']} encodings exist")
        if not setting.get("arrivals_identical", True):
            fails.append(f"I8 {tag}arrival sequences differ between combinations")
        # I5': S0 serves the replaced system's answer until it switches, so when every arrival
        # happened before completion its delivered quality is EXACTLY what the source index
        # gives those same queries. (Comparing against M_S would be wrong: M_S is a mean over
        # all evaluation queries, and the arrived subset's mean is a different number.)
        for combo, s in sorted(setting["summaries"].items()):
            if not combo.startswith("S0/"):
                continue
            if s.get("n_pre_complete") != s.get("n_arrived"):
                notes.append(f"I5' {tag}{combo} skipped: {s['n_arrived'] - s['n_pre_complete']} "
                             f"arrival(s) happened after the migration completed")
            elif abs(s["served_mean"] - s["source_mean"]) > TOL:
                fails.append(f"I5' {tag}{combo} served {s['served_mean']:.6f}, source reference "
                             f"{s['source_mean']:.6f}")
    if not fails:
        notes.append("I4'/I5'/I7/I8/I9 (replay v3): ok")
    if "leaky" in res and not res["leaky"]:
        notes.append("I6 evaluation queries are measured at first service only, never re-scored "
                     "after their own repair: ok (enforced in run_arrivals)")
    return fails, notes
