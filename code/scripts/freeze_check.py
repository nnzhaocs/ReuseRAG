"""Emit FREEZE_CHECK.md from the results, completely, and verify its own output.

    python scripts/freeze_check.py results/*.json --out FREEZE_CHECK.md
    python scripts/freeze_check.py results/*.json results_measure/*.json --out FREEZE_CHECK.md
    python scripts/freeze_check.py results/P{2,3,4,5,9,10,11,12,13,14}.json sweeps_fine/sweep_P*.json --out FREEZE_CHECK.md

Why this script exists. Nothing in the repo wrote FREEZE_CHECK.md -- it was assembled by hand,
and the 2026-11-07 copy was **truncated mid-JSON** at `"G_at_10_25_50": {... "P11": [`. The
four pairs it cut off, P11 to P14, were the only ones that mattered: the pre-registered rule
picks the smallest K' admissible on *every* pair, so the depth the paper reports rests entirely
on P11, and P11's number was inside the truncation. A freeze record that silently loses its own
binding constraint is worse than no freeze record.

Two design choices follow from that, and they are the point of the script:

  1. **A missing measurement is printed, not omitted.** Every (pair, K') cell the frontier
     wants appears in the table; one that was never measured reads `not measured` and is
     counted in the verdict. Absence you can see is a finding; absence you cannot see is the
     bug above.
  2. **The output is re-read and checked before the script exits.** Every fenced JSON block
     must parse, and every input pair must appear. A truncated or malformed write fails here
     instead of six days later in a review.

There is deliberately **no option to supply G values by hand.** Everything in this file comes
from a result JSON carrying a `run_key`. The six readable numbers in the 2026-11-07 record exist
only as prose inside that truncated markdown — the Phase 11.1 sweep's own outputs
(`results_measure/`) are in no archive — and a freeze record that accepts transcribed numbers
would launder exactly that into something a referee reads as measured. Re-run the sweep.

Exit codes: 0 written and verified, 1 written but the frontier is incomplete, 2 refused.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from reusemap import provenance as prov  # noqa: E402


def git_log(n=12):
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "log", f"-{n}", "--oneline"],
            stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(no git)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="result and/or measure JSONs")
    ap.add_argument("--out", required=True, help="path to write (e.g. FREEZE_CHECK.md)")
    ap.add_argument("--kprimes", default="10,25,50",
                    help="the frontier depths the window rule is decided on")
    ap.add_argument("--eps", type=float, default=None,
                    help="admissibility threshold; default: each result's own config")
    a = ap.parse_args()

    kprimes = [int(k) for k in a.kprimes.split(",") if k.strip()]
    loaded, unreadable = {}, []
    for p in a.results:
        try:
            with open(p, encoding="utf-8") as fh:
                r = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            unreadable.append(f"{os.path.basename(p)}: {e}")
            continue
        if not r.get("name") and r.get("pair") and isinstance(r.get("depths"), list):
            # a sweep.py output (sweeps_fine/sweep_P*.json): the Phase 11.1 measurement of G at
            # the shallow rungs. Read it as a measure-only ladder for that pair.
            r = {"name": r["pair"], "measure_only": True,
                 "G": {str(int(d["kprime"])): d["G"] for d in r["depths"] if d.get("G") is not None}}
        name = r.get("name")
        if not name:
            unreadable.append(f"{os.path.basename(p)}: no 'name'")
            continue
        # a replay result supersedes a measure-only one for the same pair, but either can
        # carry the G ladder, so merge the ladders rather than letting one shadow the other
        if name in loaded:
            merged = {**(loaded[name].get("G") or {}), **(r.get("G") or {})}
            if r.get("measure_only") and not loaded[name].get("measure_only"):
                loaded[name]["G"] = merged
                continue
            r["G"] = merged
        loaded[name] = r
    if not loaded:
        for u in unreadable:
            print("  unreadable", u, file=sys.stderr)
        print("refused: no readable results", file=sys.stderr)
        sys.exit(2)

    eps = a.eps
    if eps is None:
        eps = next((float((r.get("config") or {}).get("eps", 0.01))
                    for r in loaded.values() if (r.get("config") or {}).get("eps")), 0.01)

    # ---- the frontier, with every cell present ----
    frontier, missing = {}, []
    for name in sorted(loaded):
        g = {int(k): float(v) for k, v in (loaded[name].get("G") or {}).items()
             if v is not None}
        row = {}
        for k in kprimes:
            row[k] = g.get(k)
            if g.get(k) is None:
                missing.append((name, k))
        frontier[name] = row

    def smallest_admissible(row):
        ok = [k for k in kprimes if row.get(k) is not None and row[k] <= eps]
        return min(ok) if ok else None

    per_pair = {n: smallest_admissible(r) for n, r in frontier.items()}
    # K* requires BOTH conditions, and each failed a different way:
    #
    #  - every pair must have an admissible depth. The first form took max() over the pairs
    #    that had one and skipped those that did not, so a pair whose G exceeds eps at every
    #    rung was dropped from the maximum and the record printed a K* under the sentence
    #    "every pair is admissible at this depth", which was false.
    #  - no cell may be unmeasured. `smallest_admissible` can only see measured rungs, so a
    #    missing shallower rung might have been admissible and K* would be too large. (The
    #    first repair of the bug above dropped this condition and reported K* = 50 for a pair
    #    measured only at 50.)
    #
    # Certifying a depth no measurement supports is the one thing this file must not do, so
    # both conditions withhold K*, and the table tells the reader which one fired.
    inadmissible = sorted(n for n in frontier if per_pair[n] is None
                          and all(frontier[n][k] is not None for k in kprimes))
    k_star = (max(per_pair.values())
              if not missing and all(v is not None for v in per_pair.values()) else None)

    L = []
    L.append("# FREEZE_CHECK")
    L.append("")
    L.append(f"Generated by `scripts/freeze_check.py` at "
             f"{datetime.datetime.now().isoformat(timespec='seconds')}, "
             f"repo {prov.git_version(ROOT)}.")
    L.append("")
    L.append(f"Inputs: {len(loaded)} pairs — {', '.join(sorted(loaded))}.")
    L.append(f"Admissibility threshold eps = {eps}.")
    L.append("")

    L.append("## Window frontier")
    L.append("")
    L.append("`G(K')` at each frontier depth. The pre-registered rule takes the **smallest K'"
             " with `G <= eps` on every pair**, so a single unmeasured cell can leave the"
             " depth undecided — which is why they are printed rather than dropped.")
    L.append("")
    L.append("| pair | " + " | ".join(f"K'={k}" for k in kprimes) + " | smallest admissible |")
    L.append("|---|" + "---|" * (len(kprimes) + 1))
    for name in sorted(frontier):
        cells = []
        for k in kprimes:
            v = frontier[name][k]
            cells.append("**not measured**" if v is None
                         else (f"{v:+.5f}" + ("" if v <= eps else " ✗")))
        pp = per_pair[name]
        cells.append(str(pp) if pp is not None
                     else ("undecided" if any(frontier[name][k] is None for k in kprimes)
                           else "none admissible"))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")

    if k_star is not None:
        binders = sorted(n for n in frontier if per_pair[n] == k_star)
        L.append(f"**K\\* = {k_star}**, bound by {', '.join(binders)}. Every pair is admissible "
                 f"at this depth and at least one is not admissible below it.")
    elif inadmissible:
        L.append(f"**K\\* does not exist on this ladder.** {', '.join(inadmissible)} "
                 f"{'is' if len(inadmissible) == 1 else 'are'} fully measured and admissible at "
                 f"*no* depth (G > eps at every rung), so there is no depth admissible on every "
                 f"pair. Either widen the ladder, or exclude "
                 f"{'that pair' if len(inadmissible) == 1 else 'those pairs'} from the depth "
                 f"rule and say so in the paper — do not report the maximum over the pairs that "
                 f"do have one.")
        if missing:
            L.append("")
            L.append(f"{len(missing)} cell(s) are also unmeasured:")
            for n, k in missing:
                L.append(f"- {n} at K'={k}")
    else:
        L.append(f"**K\\* undecided.** {len(missing)} frontier cell(s) were never measured, so "
                 f"the smallest depth admissible on every pair cannot be computed. The depth "
                 f"the paper reports is therefore the depth that was *used*, not the smallest "
                 f"admissible one — say so, or measure the missing cells.")
        L.append("")
        L.append("Missing cells:")
        for n, k in missing:
            L.append(f"- {n} at K'={k}")
    L.append("")
    L.append("```json")
    L.append(json.dumps({"eps": eps, "kprimes": kprimes, "k_star": k_star,
                         "inadmissible": inadmissible,
                         "unmeasured": [[n, k] for n, k in missing],
                         "smallest_admissible_per_pair": per_pair,
                         "G": {n: {str(k): frontier[n][k] for k in kprimes}
                               for n in sorted(frontier)}},
                        indent=1, sort_keys=True))
    L.append("```")
    L.append("")

    L.append("## Provenance")
    L.append("")
    L.append("| pair | window | N | run_key | git | measure-only |")
    L.append("|---|---|---|---|---|---|")
    for name in sorted(loaded):
        r = loaded[name]
        pv = r.get("provenance") or {}
        L.append(f"| {name} | {r.get('window')} | {r.get('N')} | {pv.get('run_key')} "
                 f"| {pv.get('git')} | {bool(r.get('measure_only'))} |")
    L.append("")
    dirty = [n for n in sorted(loaded)
             if prov.is_dirty((loaded[n].get("provenance") or {}).get("git", ""))]
    L.append(f"Results from uncommitted code: {', '.join(dirty) if dirty else 'none'}.")
    L.append("")
    L.append("## Recent commits")
    L.append("")
    L.append("```")
    L.append(git_log())
    L.append("```")
    if unreadable:
        L.append("")
        L.append("## Inputs that could not be read")
        L.append("")
        for u in unreadable:
            L.append(f"- {u}")
    L.append("")

    text = "\n".join(L)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)

    # ---- verify what landed on disk, not what we meant to write ----
    with open(a.out, encoding="utf-8") as fh:
        back = fh.read()
    problems = []
    if back != text:
        problems.append("the file on disk differs from what was generated (short write?)")
    for i, block in enumerate(re.findall(r"```json\n(.*?)```", back, flags=re.S)):
        try:
            json.loads(block)
        except json.JSONDecodeError as e:
            problems.append(f"JSON block {i} does not parse: {e}")
    for name in loaded:
        if not re.search(rf"^\| {re.escape(name)} \|", back, flags=re.M):
            problems.append(f"pair {name} is missing from the output tables")
    if problems:
        print("WROTE A BAD FILE — do not use it:", file=sys.stderr)
        for p in problems:
            print("  ", p, file=sys.stderr)
        sys.exit(2)

    print(f"wrote {a.out} ({len(back):,} bytes, {len(back.splitlines())} lines); "
          f"every JSON block parses and all {len(loaded)} pairs appear")
    if k_star is not None:
        print(f"K* = {k_star}")
        sys.exit(0)
    if inadmissible:
        print(f"K* DOES NOT EXIST: {', '.join(inadmissible)} admissible at no measured depth")
    if missing:
        print(f"K* UNDECIDED: {len(missing)} frontier cell(s) not measured — "
              + ", ".join(f"{n}@{k}" for n, k in missing[:8])
              + (" ..." if len(missing) > 8 else ""))
    sys.exit(1)


if __name__ == "__main__":
    main()
