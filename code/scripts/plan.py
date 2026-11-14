"""The planner: decide D1/D2/D3 from STATIC inputs, then score against the replay.

    python scripts/plan.py results/P*.json --coverage-dir coverage --config configs/msmarco_1m_v3.json
    python scripts/plan.py results/P*.json --coverage-dir coverage --config ... --csv results/tables

Three decisions, each from quantities computable without encoding the corpus with the target
model (OUTLINE.md "The planner"):

  D1  migrate at all?   M_T - M_S on a query sample; below tau, serve the old index (S0)
  D2  which family?     never S1/S3 (mapping: 0 wins in 52 configurations at every tau) and
                        never S4/S5 (cost unbounded by anything static; quality inside the
                        band). So: the window, iff 1 - cov(K') leaves a saving to take
  D3  what depth?       smallest K' on the measured ladder with G(K') <= eps; cost is cov(K')

`decide()` is handed only static fields and CANNOT see `settings`, so a decision cannot leak
the replayed winner. `score()` is the only place the replay is read. That separation is the
whole point: it is what lets the table be read as a test of the planner rather than a fit.

What this is NOT. The replays were run and inspected before this script existed, so agreement
is a CONSISTENCY result, not a pre-registered prediction -- the same caveat predict.py carries.
Say so wherever the number appears. The honest claim is "the planner's family decision is retrospectively consistent with the
replayed winner on N of M pairs, with the exceptions named", not "the planner predicts winners".

Exit codes: 0 scored, 1 at least one unexplained miss, 2 refused (no usable inputs).
"""
import argparse
import csv
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

MAPPING = ("S1", "S3")       # 0 wins in 52 configurations, at every tau on the grid
UNPRICED = ("S4", "S5")      # cov(K') counts windows; these also fetch BM25 nominations


def decide(name, M_S, M_T, g_ladder, coverage_rows, tau, eps, ceiling_floor=0.01):
    """The planner's three decisions, from static inputs only.

    `g_ladder`   {K' (int): G(K')}  from measure.py on a judged sample
    `coverage_rows` [{kprime, coverage, ceiling_frac}, ...] from coverage.py on the SOURCE index
    Returns a dict; `family` is what D2 names, `kprime`/`cost` what D3 adds.
    """
    out = {"pair": name, "quality_gap": None, "family": None, "kprime": None,
           "cost": None, "ceiling": None, "why": None, "excluded": ",".join(MAPPING + UNPRICED)}

    # D1 -- the only input is the quality gap on a query sample.
    if M_S is None or M_T is None:
        out["why"] = "D1 undecidable: no M_S/M_T"
        return out
    gap = float(M_T) - float(M_S)
    out["quality_gap"] = gap
    if gap <= tau:
        out["family"], out["kprime"], out["cost"] = "S0", None, None
        out["why"] = (f"D1: gap {gap:+.5f} <= tau {tau} -- the upgrade is inside the band the "
                      f"study calls equivalent, so serve the old index and rebuild in the "
                      f"background")
        return out

    # D3 -- the depth decides whether there is a window to choose at all.
    #
    # TWO conditions, not one. "Smallest K' with G <= eps" alone minimises cost subject to an
    # admissibility constraint and never looks at debt, but a shallower window RAISES debt: the
    # window serves target quality minus the window gap, so debt(window) ~ G(K'). Where the
    # quality gap is small the two pull apart and the one-condition rule picks a depth at which
    # migrating is worse than not migrating. P5 is the case: gap +0.00657, and G(10) = +0.00657,
    # so at K' = 10 the window gives back exactly what the upgrade was worth. The rule as first
    # written picked 10 there.
    #
    # Both conditions are static -- G from the ladder, the gap from D1 -- so this costs nothing.
    if not any(v is not None for v in (g_ladder or {}).values()):
        out["why"] = "D3 undecidable: the result has no G(K') ladder"
        return out
    usable, rejected = [], []
    for k in sorted(g_ladder):
        g = g_ladder[k]
        if g is None:
            continue
        g = float(g)
        if g > eps:
            rejected.append((k, f"K'={k} inadmissible (G={g:+.5f} > eps {eps})"))
        elif gap - g <= tau:
            rejected.append((k, f"K'={k} wins nothing (G={g:+.5f} leaves only {gap - g:+.5f} of the "
                                f"{gap:+.5f} gap, inside tau {tau})"))
        else:
            usable.append(k)
    if not usable:
        out["family"] = "S0"
        out["why"] = ("D3: no measured depth both clears eps and keeps the window more than tau "
                      "better than not migrating: " + "; ".join(m for _, m in rejected) + ". Rebuild")
        return out
    kp = usable[0]
    out["kprime"] = kp
    # the depths D3 passed over, and on which condition. A chosen depth with no trail reads as
    # the only option, when the interesting fact is usually that a cheaper one existed and was
    # rejected. Built once here so every window branch below carries it.
    skipped = [m for k, m in rejected if k < kp]   # (depth, reason) pairs: no parsing of the text
    trail = ("; passed over " + "; ".join(skipped)) if skipped else ""

    # D2 -- is there a saving to take? cov(K') on the source index alone.
    cov = {int(r["kprime"]): r for r in coverage_rows} if coverage_rows else {}
    row = cov.get(kp)
    if row is None:
        out["family"] = "S2"
        out["why"] = (f"D2: window at K'={kp} (G={float(g_ladder[kp]):+.5f} <= eps); cost "
                      f"unknown -- coverage has no K'={kp} row" + trail)
        return out
    ceiling = float(row["ceiling_frac"]) if "ceiling_frac" in row else 1.0 - float(row["coverage"])
    out["ceiling"], out["cost"] = ceiling, float(row["coverage"])
    if ceiling <= ceiling_floor:
        out["family"], out["kprime"], out["cost"] = "S0", None, None
        out["why"] = (f"D2: cov(K'={kp}) = {float(row['coverage']):.4f} leaves a ceiling of "
                      f"{ceiling:.4f} -- the window would touch essentially the whole corpus, "
                      f"so there is no saving to take; rebuild" + trail)
        return out
    out["family"] = "S2"
    out["why"] = (f"D2/D3: window at K'={kp} (G={float(g_ladder[kp]):+.5f} <= eps), cost "
                  f"{float(row['coverage']):.4f} of N, ceiling {ceiling:.4f}")
    # Record the depths D3 passed over. A chosen depth with no trail reads as the only option,
    # when the interesting fact is often that a cheaper one existed and was rejected -- and on
    # which of the two conditions. That trail is the paper's justification for the depth.
    out["why"] += trail
    return out


def score(decision, replay_family, replay_combo, replay_cost):
    """Compare one decision to the replay. The ONLY function that reads replay output."""
    if replay_family is None:
        return {"verdict": "no replay", "note": "measure-only pair"}
    if decision["family"] is None:   # D1 could not decide (M_S or M_T missing): not a wrong call
        return {"verdict": "undecidable", "note": f"planner could not decide; replay said {replay_combo}"}
    if decision["family"] == replay_family:
        # A family match whose cost the planner could not compute is NOT the same result as a
        # match that also priced the migration. Reported as "match" it reads as though the cost
        # column had been validated, when D3 produced no number at all -- and the cost estimate
        # is the half of the planner a database referee cares about.
        if decision["family"] != "S0" and decision.get("cost") is None:
            return {"verdict": "match (unpriced)", "note":
                    f"family matches ({replay_family}) but D3 produced no cost, so the replay's "
                    f"{replay_cost:.4f} of N is unverified against the planner. Count these "
                    f"separately from priced matches"}
        return {"verdict": "match", "note": ""}
    if replay_family in UNPRICED:
        return {"verdict": "known miss", "note":
                f"replay picked {replay_combo}, which D2 excludes by rule: its cost "
                f"({replay_cost:.4f} of N) is not bounded by cov(K') and the quality it buys "
                f"sits at or inside the band. Report it, do not fit to it"}
    if replay_family in MAPPING:
        return {"verdict": "CONTRADICTION", "note":
                f"replay picked {replay_combo}, a mapping family that D2 says never wins -- "
                f"this would falsify the paper's central negative result"}
    return {"verdict": "MISS", "note": f"planner said {decision['family']}, replay said "
                                       f"{replay_combo}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="result JSONs")
    ap.add_argument("--coverage-dir", required=True,
                    help="directory of coverage.py outputs (cov_<model>.json)")
    ap.add_argument("--config", default=None,
                    help="config to read tau/eps from; otherwise each result's own config")
    ap.add_argument("--eps", type=float, default=None, help="override eps")
    ap.add_argument("--tau", type=float, default=None, help="override tau (D1's threshold)")
    ap.add_argument("--frontier", default=None,
                    help="JSON {pair: {K': G}} of G measured outside the replay (Phase 11.1's "
                         "K' = 10/25/50 sweep). Merged into each pair's ladder without "
                         "overwriting it; needed because the replay ladder starts at 500 on the "
                         "text pairs, which is not the depth those runs used")
    ap.add_argument("--csv", default=None, help="directory to write plan.csv into")
    a = ap.parse_args()

    cfg_over = json.load(open(a.config, encoding="utf-8")) if a.config else {}
    frontier = json.load(open(a.frontier, encoding="utf-8")) if a.frontier else {}
    covs = []
    for p in sorted(glob.glob(os.path.join(a.coverage_dir, "**", "cov_*.json"), recursive=True)):
        try:
            c = json.load(open(p, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        covs.append({"file": os.path.basename(p), "name": str(c.get("name", "")).lower(),
                     "N": c.get("N"), "rows": c.get("rows") or []})

    def coverage_for(source, N):
        """A pair's coverage must come from ITS source model on ITS corpus. Returns (rows, why).

        Contract: the basename of the pair's meta.source (e.g. 'all-MiniLM-L6-v2' from
        'sentence-transformers/all-MiniLM-L6-v2') must equal the --name coverage.py was given;
        the drivers pass the model's directory name for both. Otherwise D2 reports "unpriced".

        This REFUSES rather than guessing. A substring fallback is what the first version did
        and it silently crossed corpora: P7's source is all-MiniLM-L6-v2 but its corpus is
        CodeSearchNet (N = 45,773), so the loose match handed it cov_MiniLM.json (MS MARCO,
        N = 1,000,000) and priced a 45k-document corpus off a million-document index. The cost
        column read 0.4366 and nothing flagged it. A missing cost is recoverable; a confident
        wrong one is not.
        """
        s = str(source or "").split("/")[-1].lower()
        if not s:
            return [], "no source model recorded in the result"
        same_model = [c for c in covs if c["name"] == s]
        if not same_model:
            return [], (f"no coverage file for source {s!r} (have: "
                        f"{', '.join(sorted({c['name'] for c in covs})) or 'none'})")
        exact = [c for c in same_model if N is not None and c["N"] == N]
        if exact:
            return exact[0]["rows"], ""
        got = ", ".join(f"{c['file']} (N={c['N']:,})" for c in same_model)
        return [], (f"coverage for {s!r} exists but on a different corpus: result N={N:,}, "
                    f"{got}. Coverage is a count over a specific index, so it does not carry "
                    f"across; re-run coverage.py on this pair's corpus")

    rows, problems = [], []
    for path in sorted(a.results):
        try:
            r = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            problems.append(f"{os.path.basename(path)}: unreadable ({e})")
            continue
        cfg = {**(r.get("config") or {}), **cfg_over}
        tau = a.tau if a.tau is not None else float((cfg.get("tau") or {}).get("debt", 0.005))
        eps = a.eps if a.eps is not None else float(cfg.get("eps", 0.01))
        ref = r.get("reference") or {}
        g_ladder = {int(k): v for k, v in (r.get("G") or {}).items() if v is not None}
        source = (r.get("meta") or {}).get("source") or (r.get("config") or {}).get("source")
        # extra G values measured outside the replay (Phase 11.1's K' = 10/25/50 frontier).
        # The replay's own ladder starts at 500 on the text pairs, so without these D3 cannot
        # see the depth the run actually used and prices the window ~3x too high.
        for k, v in (frontier.get(r.get("name")) or {}).items():
            g_ladder.setdefault(int(k), v)
        cov_rows, cov_why = coverage_for(source, r.get("N"))

        d = decide(r.get("name"), r.get("M_S"), ref.get("M_T"), g_ladder, cov_rows, tau, eps)
        if not cov_rows and cov_why and d["family"] == "S2":
            d["why"] += f"; COST UNKNOWN: {cov_why}"

        fam = combo = None
        cost = float("nan")
        if "primary_setting" in r and r.get("settings"):
            prim = (r.get("primary_setting") or {}).get("traffic_ratio")
            st = next((s for s in r["settings"] if s["traffic_ratio"] == prim), None)
            if st and st.get("winner"):
                combo = st["winner"]
                fam = combo.split("/")[0]
                raw_cost = st.get("summaries", {}).get(combo, {}).get("C_serve_frac")
                cost = float(raw_cost) if raw_cost is not None else float("nan")   # unpriced, not a crash
        s = score(d, fam, combo, cost)
        rows.append({**d, "replay": combo or "-", "replay_cost": cost,
                     "verdict": s["verdict"], "note": s["note"], "tau": tau, "eps": eps})

    if not rows:
        for p in problems:
            print("  ", p, file=sys.stderr)
        print("refused: no usable results", file=sys.stderr)
        sys.exit(2)

    print(f"{'pair':<6}{'gap':>9}{'plan':>6}{'Kp':>6}{'cost':>8}  {'replay':<22}"
          f"{'replay cost':>12}  verdict")
    for x in rows:
        # built as plain strings, not nested f-strings: an inner f-string reusing the outer
        # quote is PEP 701 and a SyntaxError on Python <= 3.11, which this script hit.
        gap = x["quality_gap"]
        gap_s = "       -" if gap is None else f"{gap:>+9.5f}"
        cost_s = "-" if x["cost"] is None else f"{x['cost']:.4f}"
        kp_s = "-" if x["kprime"] is None else str(x["kprime"])
        print(f"{x['pair']:<6}{gap_s:>9}{str(x['family']):>6}{kp_s:>6}{cost_s:>8}  "
              f"{x['replay']:<22}{x['replay_cost']:>12.4f}  {x['verdict']}")

    counts = {}
    for x in rows:
        counts[x["verdict"]] = counts.get(x["verdict"], 0) + 1
    scored = [x for x in rows if x["verdict"] != "no replay"]
    print(f"\n{len(scored)} pairs with a replay to score against: "
          + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    for x in rows:
        if x["note"]:
            print(f"\n  {x['pair']}: {x['note']}")
    print("\nThe replays were run and inspected before this script existed, so agreement is a "
          "CONSISTENCY result, not a pre-registered prediction. Quote it as 'the planner's "
          "family decision is retrospectively consistent with the replayed winner on "
          f"{counts.get('match', 0)} of {len(scored)} pairs', with the exceptions named.")
    for p in problems:
        print("  ", p)

    if a.csv:
        os.makedirs(a.csv, exist_ok=True)
        out = os.path.join(a.csv, "plan.csv")
        cols = ["pair", "quality_gap", "family", "kprime", "cost", "ceiling", "replay",
                "replay_cost", "verdict", "tau", "eps", "excluded", "why", "note"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {out}")

    bad = [x for x in rows if x["verdict"] in ("MISS", "CONTRADICTION")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
