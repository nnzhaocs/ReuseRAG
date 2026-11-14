"""Turn result JSONs into the paper's tables (markdown on stdout, optional CSVs).

  python scripts/summarize.py results/*.json --exclude P0 P6 --csv results/tables

Tables:
  map      per pair: models, N, eval queries, M_S/M_T/M_A, kappa, CKA, rho and G at the
           strategy window, winner/C_eps/floor at the primary setting, warnings
  grid     winner in every (traffic_ratio, repeat_rate) setting -- RQ1 sensitivity
  probe    RQ3: estimator, sample budget, total encoding cost, rho_hat vs rho_true, error
  heat     RQ4: AUC with bootstrap CI, gate decision, label rule
  calib    measure-only runs (e.g. the NQ comparison): G at every K'
Paper tables refuse inputs that are leaky (v1 replay), development runs (--set overrides)
or from uncommitted code, unless --include-nonpaper is given; such rows are marked.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from reusemap import provenance as prov  # noqa: E402
from reusemap import winner as winner_mod  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--exclude", nargs="*", default=[], help="pair names to leave out (e.g. P0 P6)")
ap.add_argument("--csv", default=None, help="directory to write one CSV per table")
ap.add_argument("--include-nonpaper", action="store_true",
                help="keep leaky / development / dirty-code results (marked) instead of refusing")
a = ap.parse_args()


def f(x, n=4):
    # np.floating too: runner._jsonable converts on the way out, but a hand-written result or an
    # older file can still carry numpy scalars, and those would print unrounded via str()
    return "" if x is None else (f"{x:.{n}f}" if isinstance(x, (float, np.floating)) else str(x))


def nonpaper_reasons(r):
    why = []
    if r.get("leaky", False) or "leaky" not in r:
        why.append("leaky or pre-v2 replay")
    if r.get("config", {}).get("_dev_overrides"):
        why.append("development overrides")
    if "-dirty" in str(r.get("provenance", {}).get("git", "")):
        why.append("uncommitted code")
    return why


res, refused = [], []
for path in a.results:
    r = json.load(open(path))
    if "reference" not in r or r.get("name") in a.exclude:
        continue
    why = nonpaper_reasons(r)
    if why and not a.include_nonpaper:
        refused.append((r.get("name"), path, why))
        continue
    r["_flag"] = ";".join(why)
    res.append(r)
if refused:
    print("REFUSED (not paper-grade; use --include-nonpaper to inspect them):", file=sys.stderr)
    for name, path, why in refused:
        print(f"  {name}: {', '.join(why)}  [{path}]", file=sys.stderr)

tables = {k: [] for k in ("map", "grid", "probe", "heat", "calib")}
for r in sorted(res, key=lambda x: x["name"]):
    m, ref, w = r.get("meta", {}), r["reference"], str(r["window"])
    if r.get("measure_only"):
        for k in sorted(r["G"], key=int):
            tables["calib"].append({"pair": r["name"], "source": m.get("source"), "target": m.get("target"),
                                    "M_S": ref["M_S"], "M_T": ref["M_T"], "Kprime": int(k),
                                    "rho": r["rho"].get(k), "G": r["G"][k]})
        continue
    s = r["summaries"].get(r["winner"] or "", {})
    v3 = r.get("replay_version") == 3
    row = {
        "pair": r["name"], "source": m.get("source"), "target": m.get("target"), "N": r["N"],
        "n_eval": r["n_eval"], "M_S": ref["M_S"], "M_T": ref["M_T"], "M_A": ref["M_A"],
        "kappa": r["compat"]["kappa"], "cka": r["compat"]["cka"], "window": int(w),
        "rho_window": r["rho"].get(w), "G_window": r["G"].get(w), "winner": r["winner"]}
    if v3:
        # the objective is the quality lost during the migration; cost is N for everything that
        # completes, so C_serve_frac is a diagnostic (see DESIGN.md "Replay protocol v3")
        # C_serve is not a saving and not one quantity: C_serve_basis says which of the three
        # situations produced it, and C_total (= 1.0000 for everything admissible) shows that
        # the remainder is paid later. See summarize_arrivals' docstring.
        # Select by the PRIMARY SETTING's traffic ratio, never by winner equality. Window
        # strategies win at every ratio, so the same winner string appears in several settings
        # and "first setting whose winner matches" returned ratio 4 while primary_setting is
        # ratio 64 -- on 10 of 13 pairs in the 2026-11-07 run, with tied_exact wrong on 8 of
        # them (P11 reported 4 ties instead of 6; P2/P4/P9/P12/P13 reported 1 instead of 4).
        # tied_exact is a column the paper prints, so this was a wrong number in Table 1.
        prim_st = winner_mod.primary_setting(r)
        row.update({"debt": s.get("debt"), "served_mean": s.get("served_mean"),
                    "source_mean": s.get("source_mean"), "C_serve_frac": s.get("C_serve_frac"),
                    "C_serve_basis": s.get("C_serve_basis"), "C_total_frac": s.get("C_total_frac"),
                    "repairs_per_query": s.get("repairs_per_query_mean"),
                    "band": len(r.get("winner_ties", [])),
                    "tied_exact": len(prim_st.get("tied_exact", [])),
                    "tied_families": ",".join(prim_st.get("tied_exact_families", []))})
    else:
        row.update({"C_eps_frac": s.get("C_eps_frac"), "floor": s.get("floor")})
    row.update({"warnings": len(r.get("warnings", [])), "flag": r["_flag"]})
    tables["map"].append(row)
    for st in r.get("settings", []):
        ws = st["summaries"].get(st["winner"] or "", {})
        g = {"pair": r["name"], "traffic_ratio": st["traffic_ratio"],
             "repeat_rate": st["repeat_rate"], "stream_len": st["stream_len"],
             "winner": st["winner"]}
        if v3:
            g.update({"debt": ws.get("debt"), "C_serve_frac": ws.get("C_serve_frac"),
                      "C_serve_basis": ws.get("C_serve_basis"),
                      "C_total_frac": ws.get("C_total_frac"),
                      "eval_served_frac": st.get("eval_served_frac"),
                      "non_completing": len(st.get("non_completing", [])),
                      "no_winner_reason": st.get("no_winner_reason"),
                      "n_in_band": len(st.get("winner_ties", [])),
                      "n_tied_exact": len(st.get("tied_exact", [])),
                      "cost_basis_conflict": ",".join(st.get("cost_basis_conflict", []))})
        else:
            g.update({"C_eps_frac": ws.get("C_eps_frac"), "n_ties": len(st.get("winner_ties", []))})
        tables["grid"].append(g)
    pr = r.get("probe") or {}
    for e in pr.get("estimates", []):
        tables["probe"].append({"pair": r["name"], "Kprime": pr["kprime"], "n_queries": pr["n_queries"],
                                "estimator": e["kind"], "sample_frac": e["budget_frac"],
                                "total_frac": e["total_encodings_frac"], "rho_true": pr["rho_true"],
                                "rho_hat": e["rho_hat"], "abs_err": e["abs_err"]})
    hg = r.get("heat_gate") or {}
    tables["heat"].append({"pair": r["name"], "available": hg.get("available"), "auc": hg.get("auc"),
                           "ci_lo": (hg.get("ci") or [None, None])[0], "ci_hi": (hg.get("ci") or [None, None])[1],
                           "gate_open": hg.get("gate_open"), "label_rule": hg.get("label_rule"),
                           "n_docs": hg.get("n_docs")})

for name, rows in tables.items():
    if not rows:
        continue
    cols = list(dict.fromkeys(k for row in rows for k in row))  # union, first-seen order
    rows = [{c: row.get(c) for c in cols} for row in rows]
    print(f"\n### {name}\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for row in rows:
        print("| " + " | ".join(f(row[c]) for c in cols) + " |")
    if a.csv:
        os.makedirs(a.csv, exist_ok=True)
        out = os.path.join(a.csv, f"{name}.csv")
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            wr.writerows(rows)
        # which results this view came from, so check_tables.py can catch it going stale
        prov.write_table_inputs(out, res, ROOT)
if not res:
    sys.exit("no paper-grade results to summarize")
