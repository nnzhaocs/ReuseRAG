"""Does every table in results/tables still describe the results sitting beside it?

    python scripts/check_tables.py results/tables results/P*.json

A table there is a view of the result JSONs, and two scripts write into that one directory:
summarize.py makes map/grid/probe/heat/calib, tau_curve.py makes tau_curve. Re-run one and not
the other and the directory holds a mix of generations that looks uniformly current. That
shipped. In the 2026-11-07 archive map.csv and grid.csv matched the P*.json row for row while
tau_curve.csv still described the previous generation: 177 of its 416 rows disagreed, over ten
pairs, and the stale version was the flattering one -- it showed no S4 winner anywhere, where
the current results have S4 winning all four P11 cells at 3.2x the window's cost.

Each generator now writes `<table>.csv.inputs.json` recording the run_key of every result it
read. This compares those against the results given. A table that is missing its sidecar is
reported too: it cannot be checked, which is the state every table was in before.

Run it as the last step of the freeze check, after regenerating the tables, and again on any
archive before its numbers go into the paper.

Exit codes: 0 all tables current, 1 at least one stale or uncheckable, 2 refused (bad inputs).
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from reusemap import provenance as prov  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tables", help="the tables directory (e.g. results/tables)")
ap.add_argument("results", nargs="+", help="the result JSONs the tables should describe")
a = ap.parse_args()

if not os.path.isdir(a.tables):
    print(f"refused: {a.tables} is not a directory", file=sys.stderr)
    sys.exit(2)

results, unreadable = [], []
for p in a.results:
    try:
        with open(p, encoding="utf-8") as fh:
            results.append(json.load(fh))
    except (OSError, json.JSONDecodeError) as e:
        unreadable.append(f"{os.path.basename(p)}: {e}")
if not results:
    for u in unreadable:
        print("  unreadable", u, file=sys.stderr)
    print("refused: no readable result JSONs", file=sys.stderr)
    sys.exit(2)

csvs = sorted(glob.glob(os.path.join(a.tables, "*.csv")))
if not csvs:
    print(f"refused: no CSVs in {a.tables}", file=sys.stderr)
    sys.exit(2)

now = prov.table_inputs(results)
print(f"{len(csvs)} tables in {a.tables} against {len(results)} results "
      f"({', '.join(sorted(now))})\n")

problems = []
for c in csvs:
    msgs = prov.check_table_inputs(c, results)
    problems += msgs
    if msgs:
        for m in msgs:
            print(f"  STALE  {m}")
    else:
        with open(c + ".inputs.json", encoding="utf-8") as fh:
            rec = json.load(fh)
        print(f"  ok     {os.path.basename(c):<16} {len(rec.get('inputs') or {})} inputs, "
              f"built {rec.get('built')} at {rec.get('git')}")

for u in unreadable:
    print("  unreadable", u)
if unreadable:
    # a result the check cannot read is a result it cannot vouch for: a partly corrupt archive
    # used to pass here whenever the readable files matched the sidecars
    problems.append(f"{len(unreadable)} result file(s) unreadable")

if problems:
    print(f"\n{len(problems)} problem(s). A stale table is not a wrong number you can correct in "
          f"the text: regenerate it (summarize.py for map/grid/probe/heat/calib, tau_curve.py "
          f"for tau_curve) and read the numbers again, because a table and the results beside it "
          f"disagreeing means one of them is not this run.")
    sys.exit(1)
print(f"\nall {len(csvs)} tables describe these results")
