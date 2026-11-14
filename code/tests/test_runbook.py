"""Check RUNBOOK.md against the code: every `python scripts/X.py ...` command must name a
script that exists and use only flags that script defines; every configs/*.json it names
must exist (golden files the user creates are exempt); the stated test count must match.

  python tests/test_runbook.py
"""
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_rb = os.path.join(ROOT, "RUNBOOK.md")
if not os.path.exists(_rb):
    sys.exit(f"refused: {_rb} is missing (this check compares RUNBOOK.md against the code)")
text = open(_rb, encoding="utf-8").read()
problems = []

# fenced bash blocks, with backslash continuations joined
blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.S)
lines = []
for b in blocks:
    lines += re.sub(r"\\\n\s*", " ", b).splitlines()

defined, required, positional = {}, {}, {}
for line in lines:
    for m in re.finditer(r"python (scripts/[\w./-]+\.py)([^;&|]*)", line):
        script, args = m.group(1), m.group(2)
        path = os.path.join(ROOT, script)
        if not os.path.exists(path):
            problems.append(f"missing script: {script}")
            continue
        if script not in defined:
            src = open(path, encoding="utf-8").read()
            defined[script] = set(re.findall(r"add_argument\(\s*[\"'](--[\w-]+)[\"']", src))
            # required flags and whether the script takes a positional at all. Checking only
            # that flag NAMES exist let a command missing every required flag through: Phase
            # 11.2 shipped `predict.py results_measure/$P.json --out ...` against a script whose
            # --result and --coverage are both required, so it failed on the first line anyone
            # ran, and this checker passed it.
            # scan each add_argument(...) separately. [^)]* would stop at the first newline
            # or ")", hiding required=True in a multi-line call -- and this is the check that
            # exists to catch the Phase 11.2 class of error. A plain [\s\S]*? is worse: it
            # runs ACROSS calls, so a flag inherits a later call's required=True.
            required[script] = set()
            starts = [m.start() for m in re.finditer(r"add_argument\(", src)] + [len(src)]
            for lo, hi in zip(starts, starts[1:]):
                call = src[lo:hi]
                m = re.match(r"add_argument\(\s*[\"'](--[\w-]+)[\"']", call)
                if m and re.search(r"required\s*=\s*True", call):
                    required[script].add(m.group(1))
            positional[script] = bool(re.search(r"add_argument\(\s*[\"'][^-]", src))
        given = set(re.findall(r"(?<![\w'\"{])(--[a-z][\w-]*)", args))
        for flag in given:
            if flag not in defined[script]:
                problems.append(f"{script}: unknown flag {flag}  [{line.strip()[:90]}]")
        for flag in sorted(required[script] - given):
            problems.append(f"{script}: missing required {flag}  [{line.strip()[:90]}]")
        # a bare path where the script defines no positional is the same mistake wearing
        # different clothes -- argparse rejects it as an unrecognised argument
        if not positional[script]:
            # walk left to right: every non-flag token after a --flag is one of its values
            # (nargs="+" means there can be several), so only a token appearing before any
            # flag is a genuine positional
            seen_flag = False
            for tok in args.split():
                if tok.startswith("-"):
                    seen_flag = True
                    continue
                if seen_flag or "=" in tok or tok.startswith(("$", "'", '"')):
                    continue
                if "/" in tok or tok.endswith((".json", ".npy", ".tsv")):
                    problems.append(f"{script}: positional {tok!r} but the script defines none"
                                    f"  [{line.strip()[:90]}]")
                    break

for cfg in sorted(set(re.findall(r"configs/[\w./-]+\.json", text))):
    if cfg.startswith("configs/golden/") and "TEMPLATE" not in cfg:
        continue  # created by the user in Phase 2.2
    if not os.path.exists(os.path.join(ROOT, cfg)):
        problems.append(f"missing config: {cfg}")

# the drivers run Phases 13 and 14: every config and script they name must exist, every
# EXPECTED_TESTS default must match the suite, and the NQ configs cfg_for builds must exist
drv_scripts = set()
for drv in ("submission_runs.sh", "extension_runs.sh", "qwen_rerun.sh", "final_runs.sh", "final2_runs.sh"):
    src = open(os.path.join(ROOT, "scripts", drv)).read()
    for cfg in sorted(set(re.findall(r"configs/[\w./-]+\.json", src))):
        if not os.path.exists(os.path.join(ROOT, cfg)):
            problems.append(f"{drv}: missing config {cfg}")
    for sc in sorted(set(re.findall(r"scripts/[\w./-]+\.(?:py|sh)", src))):
        drv_scripts.add(sc)
        if not os.path.exists(os.path.join(ROOT, sc)):
            problems.append(f"{drv}: missing script {sc}")
    m = re.search(r"EXPECTED_TESTS=\$\{EXPECTED_TESTS:-(\d+)\}", src)
    n_inv = len(re.findall(r"^def test_", open(os.path.join(ROOT, "tests", "test_invariants.py")).read(), re.M))
    if not m or int(m.group(1)) != n_inv:
        problems.append(f"{drv}: EXPECTED_TESTS default {m and m.group(1)}, test_invariants.py has {n_inv}")
    if "cfg_for" in src:
        ks = re.search(r"for K in ([\d ]+); do \[ -f \"\$\(cfg_for", src)
        for k in (ks.group(1).split() if ks else []):
            c = "configs/nq_100k_v3.json" if k == "50" else f"configs/nq_100k_v3_k{k}.json"
            if not os.path.exists(os.path.join(ROOT, c)):
                problems.append(f"{drv}: cfg_for {k} -> missing {c}")
        if not ks:
            problems.append(f"{drv}: cannot find the cfg_for depth list in preflight")


# Final2 has its own run guide and a stricter contract because it is the first dynamic 8.84M replay.
f2doc = os.path.join(ROOT, "RUN_FINAL2.md")
f2drv = os.path.join(ROOT, "scripts", "final2_runs.sh")
if not os.path.exists(f2doc):
    problems.append("missing RUN_FINAL2.md")
elif not os.path.exists(f2drv):
    problems.append("missing scripts/final2_runs.sh")
else:
    f2text = open(f2doc, encoding="utf-8").read()
    f2src = open(f2drv, encoding="utf-8").read()
    import subprocess
    r = subprocess.run(["bash", "-n", f2drv], capture_output=True, text=True)
    if r.returncode != 0:
        problems.append(f"final2_runs.sh: bash -n failed: {r.stderr.strip()}")
    if 'SIZES=${SIZES-"2000000 4000000"}' not in f2src or 'SIZES=${SIZES:-"2000000 4000000"}' in f2src:
        problems.append("final2_runs.sh: explicit SIZES= must be preserved (use ${SIZES-default}, not :-)")
    if "check_full_inputs ||" not in f2src or "FULL_N=8841823" not in f2src:
        problems.append("final2_runs.sh: preflight does not hard-require/validate the 8.84M inputs")
    if '[ -f "results_$lab/$p.json" ] && continue' in f2src:
        problems.append("final2_runs.sh: scale replay still trusts an existing filename instead of run_pair provenance")
    if re.search(r"replay_prereg\.py score[^\n]*\n?[^\n]*\|\| true", f2src):
        problems.append("final2_runs.sh: replay prediction-score failure is still swallowed by || true")
    m_fp = re.search(r'^FPAIRS="([^"]+)"', f2src, re.M)
    m_rf = re.search(r'^REPLAY_FULL=\$\{REPLAY_FULL:-"([^"]+)"\}', f2src, re.M)
    expected = {"P2f", "P3f", "P9f"}
    if not m_fp or set(m_fp.group(1).split()) != expected:
        problems.append(f"final2_runs.sh: FPAIRS must be exactly {sorted(expected)}")
    if not m_rf or set(m_rf.group(1).split()) != expected:
        problems.append(f"final2_runs.sh: REPLAY_FULL must contain exactly {sorted(expected)}")
    for cfg in ("configs/msmarco_full_v3_replay.json", "configs/msmarco_s2m_v3_replay.json",
                "configs/msmarco_s4m_v3_replay.json"):
        try:
            import json
            c = json.load(open(os.path.join(ROOT, cfg)))
            if int(c.get("window", -1)) != 50:
                problems.append(f"{cfg}: scale replay window must stay fixed at K'=50")
            if "no replay is run at 8.8M" in c.get("_preregistered_scale", ""):
                problems.append(f"{cfg}: stale metadata still says no 8.8M replay")
        except Exception as e:
            problems.append(f"{cfg}: unreadable ({e})")
    if "all scale replays intentionally use fixed K'=50" not in f2text:
        problems.append("RUN_FINAL2.md: must state that all scale replays intentionally use fixed K'=50")
    if "full-scale inputs ok" not in f2text or "8,841,823" not in f2text:
        problems.append("RUN_FINAL2.md: must document the strict 8.84M preflight pass condition")
    # Every driver stage shown in RUN_FINAL2 must be accepted by final2_runs.sh.
    case = re.search(r'case "\$STAGE" in(.*?)esac', f2src, re.S)
    accepted = set(re.findall(r'[A-Za-z0-9_]+', case.group(1).split(')')[0])) if case else set()
    for stage in re.findall(r'bash scripts/final2_runs\.sh ([a-z0-9]+)', f2text):
        if stage not in accepted:
            problems.append(f"RUN_FINAL2.md: stage {stage!r} is not accepted by final2_runs.sh")

n_tests = len(re.findall(r"^def test_", open(os.path.join(ROOT, "tests", "test_invariants.py")).read(), re.M))
stated = re.search(r'"(\d+) tests passed"', text)
for m_ in re.finditer(r"test count is not (\d+)", text):
    if int(m_.group(1)) != n_tests:
        problems.append(f"RUNBOOK says 'the test count is not {m_.group(1)}'; the suite has {n_tests}")
if not stated or int(stated.group(1)) != n_tests:
    problems.append(f"RUNBOOK states {stated and stated.group(1)} tests; test_invariants.py has {n_tests}")

if problems:
    print("\n".join(problems))
    sys.exit(1)
print(f"runbook ok ({sum(1 for line in lines if 'python scripts/' in line)} commands, "
      f"{len(defined)} scripts, {n_tests} tests; drivers name {len(drv_scripts)} scripts, all present)")
