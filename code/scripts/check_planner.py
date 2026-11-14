"""Score a COMMITTED planner output (scripts/plan.py's plan.csv) against replays run after it.

    python scripts/check_planner.py --plan planner/plan.csv --results results \
        --committed-at "$(git log -1 --format=%cI -- planner/plan.csv)" \
        --runs results/RUNS.tsv --runs-tz +08:00 --md planner_check.md

What the planner claims, and so what is checked, per pair:
  family   the replay's winner at the primary setting is in the planned family (S4/S5 wins
           are reported as 'excluded by D2', the scope rule, never silently counted as a match);
  cost     for a window plan at depth K', the window strategy whose background waits for the
           workload (S2/demand_then_rand) pays C_serve/N <= cov(K') -- the bound -- and at least
           (1 - --cost-rel-tol) of it, i.e. the bound is attained;
  quality  that strategy's debt is within --debt-tol of G(K') from the replay's own ladder.
A pair is scorable only when the replay ran at the planned depth. A pair counts as a PREDICTION
only when the plan's commit time precedes the replay's start in RUNS.tsv; otherwise it is
reported as post-hoc. File modification times are never used: archives and copies reset them.
Exit 0 if every pre-committed, scorable claim holds; 1 if any fails; 2 if none was scorable.
"""
import argparse, csv, json, os, re, sys
from datetime import datetime, timedelta, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--committed-at", help="ISO-8601 commit time of the plan (git log -1 --format=%%cI)")
ap.add_argument("--runs", help="RUNS.tsv of the replays")
ap.add_argument("--runs-tz", default="local",
                help="UTC offset of RUNS.tsv's naive timestamps, e.g. +08:00; default: this machine's "
                     "offset, which is right when the check runs where the replays ran (RUNS.tsv is "
                     "written in local time). It used to default to +00:00, wrong off UTC.")
ap.add_argument("--cost-rel-tol", type=float, default=0.001)
ap.add_argument("--debt-tol", type=float, default=0.005)
ap.add_argument("--strategy", default="S2/demand_then_rand")
ap.add_argument("--md")
a = ap.parse_args()

if a.runs_tz == "local":
    _off = datetime.now().astimezone().utcoffset()
    _m = int(_off.total_seconds() // 60)
    a.runs_tz = f"{'+' if _m >= 0 else '-'}{abs(_m) // 60:02d}:{abs(_m) % 60:02d}"
    print(f"RUNS.tsv timestamps read at this machine's UTC offset {a.runs_tz} (--runs-tz)")
m = re.fullmatch(r"([+-])(\d{2}):(\d{2})", a.runs_tz)
if not m:
    sys.exit(f"--runs-tz must look like +08:00 or -05:00 (got {a.runs_tz!r})")
sign = 1 if m.group(1) == "+" else -1
tz = timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))
committed = datetime.fromisoformat(a.committed_at) if a.committed_at else None
started = {}
if a.runs:
    for r in csv.DictReader(open(a.runs), delimiter="\t"):
        if r.get("status", "").startswith("ok"):
            dt = datetime.fromisoformat(r["date"])
            # RUNS.tsv is written in naive local time; a stamp that already carries an offset
            # (hand-edited, or from another tool) keeps it rather than being re-labelled
            started[r["name"]] = dt if dt.tzinfo is not None else dt.replace(tzinfo=tz)

rows, scored, failed, pre_n = [], 0, 0, 0
for p in csv.DictReader(open(a.plan)):
    name, fam, kp = p["pair"], p["family"], p["kprime"]
    rp = os.path.join(a.results, name + ".json")
    # decide from the replay file, not the plan's 'verdict' column: a plan committed BEFORE its
    # replays (the point of a prediction) says "no replay" on every row, truthfully, when written
    if not os.path.exists(rp):
        rows.append((name, fam, kp or "-", "no replay", "", "", "", "")); continue
    R = json.load(open(rp))
    if R.get("measure_only") or not R.get("settings"):
        rows.append((name, fam, kp or "-", "no replay (measure-only result)", "", "", "", "")); continue
    tr = (R.get("primary_setting") or {}).get("traffic_ratio")
    s = next((x for x in R.get("settings", []) if x.get("traffic_ratio") == tr), None)
    if s is None:   # unscorable replay: report it, never crash on it
        rows.append((name, fam, kp or "-", f"no setting at primary traffic ratio {tr}", "", "", "", "")); continue
    win = s.get("winner")
    if not win:   # NO WINNER at the primary setting (runner warns): report it, never crash on it
        rows.append((name, fam, kp or "-", "replay has no winner at the primary setting", "", "", "", "")); continue
    wfam = win.split("/")[0]
    pre = bool(committed and name in started and committed < started[name])
    timing = "pre-committed" if pre else ("post-hoc" if committed and name in started else "timing unknown")
    fam_ok = wfam == fam
    fam_txt = "match" if fam_ok else ("excluded by D2 (" + win + ")" if wfam in ("S4", "S5") and fam == "S2" else "MISS (" + win + ")")
    cost_txt = qual_txt = ""
    ok = fam_ok or fam_txt.startswith("excluded")
    if fam == "S2":
        if not kp or int(kp) != R["window"]:
            rows.append((name, fam, kp, f"replay at K'={R['window']}: depth not scorable", timing, fam_txt, "", "")); continue
        if a.strategy not in s.get("summaries", {}):
            rows.append((name, fam, kp, f"strategy {a.strategy} not in summaries", timing, fam_txt, "", "")); continue
        w = s["summaries"][a.strategy]; cost = float(p["cost"]) if p.get("cost") not in (None, "") else None
        if cost is not None and w["C_serve_basis"] == "at_last_arrival":
            c_ok = cost * (1 - a.cost_rel_tol) <= w["C_serve_frac"] <= cost + 1e-9
            cost_txt = f"{w['C_serve_frac']:.6f} vs cov {cost:.6f}" + ("" if c_ok else " x"); ok &= c_ok
        G = R["G"].get(str(R["window"]))
        if G is not None:
            q_ok = abs(w["debt"] - G) <= a.debt_tol
            qual_txt = f"debt {w['debt']:+.4f} vs G {G:+.4f}" + ("" if q_ok else " x"); ok &= q_ok
    scored += 1; pre_n += pre; failed += (pre and not ok)
    rows.append((name, fam, kp or "-", "PASS" if ok else "FAIL", timing, fam_txt, cost_txt, qual_txt))

hdr = "| pair | planned | K' | verdict | timing | family | cost (bound) | quality |\n|---|---|---|---|---|---|---|---|\n"
body = "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)
summary = f"{scored} pairs scorable, {pre_n} pre-committed; {failed} pre-committed claims failed"
print(hdr + body + "\n" + summary)
if a.md:
    open(a.md, "w").write("# Planner check\n\n" + hdr + body + "\n" + summary + "\n")
sys.exit(2 if scored == 0 else (1 if failed else 0))
