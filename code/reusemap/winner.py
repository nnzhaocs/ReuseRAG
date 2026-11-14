"""Pre-registered winner rule and leave-one-pair-out (LOPO) winner prediction.

Winner rule (fixed before any pair is run):
  admissible  : floor >= M_S - delta  (never serve worse than the system being replaced)
  winner      : admissible combination with the smallest C_eps; ties -> larger AUC.
LOPO regret for a held-out pair = (C_eps of the predicted combination, or N if it is
inadmissible / never reaches the target / not run) - C_eps of the true winner, divided by N.
"""
import math

import numpy as np


# Pre-declared simplicity order used ONLY to break exact ties (simpler operation first).
STRATEGY_RANK = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}
ORDER_RANK = {"rand": 0, "size": 1, "signal": 2, "demand": 3, "demand_then_rand": 4}
#: An order missing from ORDER_RANK sorts last (rank 9), which is the intended default for
#: anything new; the entries above fix the order among the ones that exist.


def _simplicity(combo):
    st, od = combo.split("/")
    return (STRATEGY_RANK.get(st, 9), ORDER_RANK.get(od, 9))


def pick_winner(summaries, M_S, delta, N, return_ties=False):
    """summaries: {combo: summary dict for the primary metric}."""
    cand = []
    for combo, s in summaries.items():
        if (not math.isfinite(s["floor"]) or s["floor"] < M_S - delta
                or not math.isfinite(s["C_eps"])):
            continue
        cand.append(((s["C_eps"], round(-s["auc"], 12)), combo))
    if not cand:
        return (None, []) if return_ties else None
    best_key = min(k for k, _ in cand)
    ties = sorted((c for k, c in cand if k == best_key), key=_simplicity)
    return (ties[0], ties) if return_ties else ties[0]


#: v3 objectives. Each entry: (summary key, equivalence-band config key).
#: Both are "smaller is better". 'debt' is the pre-registered primary objective: with
#: completion required, every strategy encodes exactly N documents, so the cost axis cannot
#: separate them and the quality lost during the migration is what does.
OBJECTIVES = {"debt": ("debt", "debt"), "c_serve": ("C_serve_frac", "c_serve_frac")}


def admissible_v3(s, M_S, delta, objective="debt"):
    """Admissible under replay v3: the strategy completed the migration, and the quality it
    served during the migration was not worse than the system being replaced.

    Completion is a CONSTRAINT, not part of the cost. Without it the cost-minimal policy is
    simply 'never migrate anything the served queries did not ask for', which wins by
    construction and turns the paper into Thaw's C2 (economics of abandoning completion,
    out of scope per DESIGN.md).
    """
    key = OBJECTIVES[objective][0]
    v = s.get(key)
    # The floor is the quality the replaced system would have served THE SAME arrived queries
    # (summary['source_mean']); M_S is a mean over all evaluation queries and so is the wrong
    # constant to compare a subset against. M_S is the fallback for older result files.
    base = s.get("source_mean")
    base = float(base) if _finite(base) else float(M_S)
    return bool(s.get("completed")) and _finite(s.get("served_mean")) and _finite(v) \
        and s["served_mean"] >= base - delta


def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


#: How a tie inside the equivalence band is broken. Pre-registered, recorded in every result.
#:   "simplicity" - the simplest combination wins (the original v3 rule).
#:   "cost"       - the cheapest by C_serve wins, simplicity only among equal costs.
#: Why "cost" matters: on the 2026-11-01 1M run every window strategy had debt within 0.0006
#: of zero, so twelve to sixteen combinations sat inside tau = 0.005 and simplicity picked
#: S2/rand on 20 of 21 settings. Inside that same band, S2/demand_then_rand reached full
#: population service at 0.835-0.879 of N against S2/rand's 1.000 - i.e. the band contained
#: the entire measured 12-16% saving and the rule discarded it. C_serve is already a
#: pre-registered secondary objective, so ordering the band by it is a stated-ordering change,
#: not a new metric.
BAND_TIEBREAKS = ("simplicity", "cost")


def cost_basis_conflict(summaries, members):
    """The distinct `C_serve_basis` values among `members`, when there is more than one.

    The "cost" tie-break orders the band by C_serve_frac, which is only a comparable cost when
    every member reached the target the same way. It is not: a row with basis
    "no_repair_needed" reports how far its background got (small, and smaller still at higher
    traffic_ratio) while a row with basis "at_last_arrival" reports work it actually did. On the
    2026-11-04 1M run that put P14's S0/rand at 0.0300 against S2/demand_then_rand's 0.2642 at
    ratio 64 and handed S0 the band, though both encode N in the end.

    This reports the clash instead of silently ranking across it. The ordering is pre-registered
    and deliberately NOT changed here (DECISIONS.md 2026-11-04); the caller warns.
    """
    bases = sorted({str(summaries[c].get("C_serve_basis") or "unknown") for c in members})
    return bases if len(bases) > 1 else []


def tied_with(summaries, winner, objective="debt", tol=1e-9):
    """Combinations whose objective equals the winner's to within `tol`.

    Distinct from the equivalence band: these are not merely within tau, they are the same
    number. On the 1M run at traffic_ratio 64 and 256 the background order does not move `debt`
    (it does at 4 and 16, where the background has migrated a large share of the corpus before
    the workload is served), so at the primary setting the order half of
    a reported winner is one of several identically-scoring choices made by the simplicity rule,
    and a
    table that prints "S2/demand_then_rand" without this count overstates how determined it was.
    """
    key = OBJECTIVES[objective][0]
    if not winner or winner not in summaries:
        return []
    wv = summaries[winner].get(key)
    if not _finite(wv):
        return []
    return sorted((c for c, s in summaries.items()
                   if s.get("completed") and _finite(s.get(key))
                   and abs(float(s[key]) - float(wv)) <= tol), key=_simplicity)


def primary_setting(result):
    """The setting a result's headline row should be read from: the one whose traffic_ratio
    equals `primary_setting.traffic_ratio`. Returns {} if there is none.

    This exists because selecting it by winner equality is wrong and was wrong in shipped
    tables. Window strategies win at every traffic ratio, so the same winner string appears
    in several settings, and "the first setting whose winner matches the headline winner"
    resolved to ratio 4 while the primary setting is ratio 64 -- on 10 of 13 pairs in the
    2026-11-07 run. The `tied_exact` column then came from the wrong setting and was wrong on
    8 of them (P11 read 4 ties instead of 6), and that column is printed in the paper.
    """
    want = (result.get("primary_setting") or {}).get("traffic_ratio")
    if want is None:
        return {}
    for st in result.get("settings") or []:
        if st.get("traffic_ratio") == want:
            return st
    return {}


def pick_winner_v3(summaries, M_S, delta, objective="debt", band=0.0, return_band=False,
                   band_tiebreak="simplicity"):
    """Smallest objective among admissible combinations; everything within `band` of the best
    is equivalent, and `band_tiebreak` decides which of those wins.

    The band exists because without it a difference far below the noise floor decides a
    winner: in the v2 development runs a 0.13% difference in C_eps flipped a pair between S0
    and S5. `band` is in the objective's own units (quality for 'debt', fraction of N for
    'c_serve') and is pre-registered, as is `band_tiebreak` (see BAND_TIEBREAKS).

    The default is "simplicity" so that an existing result file re-runs to the same winner from
    its own stored config; the v3 configs opt into "cost".
    """
    if band_tiebreak not in BAND_TIEBREAKS:
        raise ValueError(f"band_tiebreak must be one of {BAND_TIEBREAKS}, got {band_tiebreak!r}")
    if not band >= 0:   # also catches NaN; a negative band would leave nobody in it
        raise ValueError(f"band must be >= 0 (the config's tau), got {band}")
    key = OBJECTIVES[objective][0]
    cand = [(float(s[key]), combo) for combo, s in summaries.items()
            if admissible_v3(s, M_S, delta, objective)]
    if not cand:
        return (None, []) if return_band else None
    best = min(v for v, _ in cand)
    members = [c for v, c in cand if v <= best + band + 1e-12]
    if band_tiebreak == "cost":
        def order(c):
            v = summaries[c].get("C_serve_frac")
            v = float(v) if _finite(v) else float("inf")
            return (round(v, 9),) + _simplicity(c)
    else:
        order = _simplicity
    inband = sorted(members, key=order)
    return (inband[0], inband) if return_band else inband[0]


def objective_of(summary, M_S, delta, penalty, objective="debt"):
    """The held-out pair's objective value for a predicted combination. A prediction that is
    inadmissible, was not run, or never reached the target costs `penalty` - for 'debt' the
    debt of serving the whole migration at the old system's quality, i.e. of not migrating at
    all during the window; for 'c_serve' a full rebuild (N)."""
    if summary is None or not admissible_v3(summary, M_S, delta, objective):
        return float(penalty)
    return float(summary[OBJECTIVES[objective][0]])


def cost_of(summary, M_S, delta, N):
    if (summary is None or not math.isfinite(summary["floor"]) or summary["floor"] < M_S - delta
            or not math.isfinite(summary["C_eps"])):
        return float(N)
    return summary["C_eps"]


def _knn_predict(Xtr, ytr, x):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    d = np.linalg.norm((Xtr - mu) / sd - (x - mu) / sd, axis=1)
    return ytr[int(np.argmin(d))]


def _tree_predict(Xtr, ytr, x):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xtr, ytr)
    return clf.predict(x.reshape(1, -1))[0]


def lopo(pairs, objective=None, features=None):
    """Leave-one-pair-out winner prediction.

    pairs: dicts with name, features{...}, winner, summaries, M_S, delta, N and - for v3 -
    `penalty`. objective=None uses the v2 rule (cost C_eps, penalty N); "debt" or "c_serve"
    uses the v3 rule. `features` overrides the feature subsets tried (default: kappa and rho,
    which is what DESIGN.md's Claim A names; pass e.g. {"all": ["kappa", "rho", "coverage"]}
    once coverage is measured and shown to vary across pairs).
    """
    pairs = [q for q in pairs if q["winner"] is not None]
    feats = features or {"both": ["kappa", "rho"], "kappa_only": ["kappa"], "rho_only": ["rho"]}
    feats = {k: v for k, v in feats.items()
             if all(c in (pairs[0]["features"] if pairs else {}) for c in v)}
    rules = {}
    for fname, cols in feats.items():
        rules[f"1nn[{fname}]"] = (cols, _knn_predict)
        rules[f"tree2[{fname}]"] = (cols, _tree_predict)
    rows = []
    for i, held in enumerate(pairs):
        train = [q for j, q in enumerate(pairs) if j != i]
        if not train:
            continue
        ytr = np.array([q["winner"] for q in train], dtype=object)
        vals, cnt = np.unique(ytr, return_counts=True)
        preds = {"majority": vals[np.argmax(cnt)]}
        for rname, (cols, fn) in rules.items():
            Xtr = np.array([[q["features"][c] for c in cols] for q in train], float)
            x = np.array([held["features"][c] for c in cols], float)
            preds[rname] = fn(Xtr, ytr, x)
        if objective is None:
            val = lambda s: cost_of(s, held["M_S"], held["delta"], held["N"])
            scale = held["N"]
        else:
            val = lambda s: objective_of(s, held["M_S"], held["delta"], held["penalty"], objective)
            scale = 1.0
        # Regret is measured against the BEST achievable value, not against the labelled
        # winner's. With an equivalence band the label is the simplest combination in the band,
        # not the band's minimum, so a predictor that lands on a cheaper member of the same band
        # would otherwise score a negative regret.
        true_cost = min([val(s) for s in held["summaries"].values()]
                        + [val(held["summaries"].get(held["winner"]))])
        for rname, pred in preds.items():
            c = val(held["summaries"].get(pred))
            rows.append({"held_out": held["name"], "rule": rname, "pred": str(pred),
                         "true": held["winner"], "correct": pred == held["winner"],
                         "regret": (c - true_cost) / scale})
    table = {}
    for r in rows:
        t = table.setdefault(r["rule"], {"n": 0, "acc": 0.0, "mean_regret": 0.0})
        t["n"] += 1
        t["acc"] += r["correct"]
        t["mean_regret"] += r["regret"]
    for t in table.values():
        t["acc"] /= t["n"]
        t["mean_regret"] /= t["n"]
    return {"per_pair": rows, "summary": table}
