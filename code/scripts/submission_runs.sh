#!/usr/bin/env bash
# Every experiment for the ICDE 2027 Round 2 submission, in dependency order.
#
#   cd /root/reusemap_v3n            # the instance tree holding enc/ pairs/ results/ coverage/ sweeps_fine/ ...
#   (install: unzip reusemap_submission_changes.zip over the 57de2da checkout as its APPLY.md says, and commit)
#   nohup bash scripts/submission_runs.sh all > logs/submission.log 2>&1 &
#
# Stages (run one with `bash scripts/submission_runs.sh <stage>`):
#   preflight  clean tree, all tests, runbook check, inputs present
#   plan       13.2a  merge frontier, run plan.py, check it, COMMIT it (must precede every d3 replay)
#   d3         13.2b  the 8 replays at K'=25 (parallel)   ~45-90 min (measured at K'=50: 34-73 min each)
#   adapters   13.1   every Drift-Adapter family on every pair (one process per pair)  ~25-40 min
#   p0         13.3   P0 under the final config, alone     ~5-10 min (measured: 9 min at window 500)
#   mass       13.5   window-multiplicity histograms + model-vs-replay    ~5-15 min
#   trecdl     TREC-DL G at K'=25/50 for P11, P14 (the d3 replays carry the other 8)  ~20-40 min
#   sample     13.6   D1 sample-size sweep                                ~25-40 min
#   xarch      full replays of the encoder->LLM pairs P5p, P5q (on by default; XARCH=0 skips) ~1-2 h
#   llm        LLM->LLM pair P5r (Qwen3-0.6B -> 8B): runs in 'all' only if its GPU encodings exist
#              (extension_runs.sh makes them and runs this stage itself; RUNBOOK 13.10, 14)
#   check      after everything: planner check, tables, tau curve, curve check
#   collect    zip every output for the paper archive
# `all` runs preflight, p0, plan (serially: P0 is final before the plan is computed and before anything
# reads results/), then d3 + adapters + mass + trecdl + sample + xarch (+ llm if its encodings exist)
# in parallel, waits, then check and collect. Every stage logs to logs/<stage>.log.
# Expected total on the 224-core instance: ~1.5-2.5 h wall-clock, bound by d3 and xarch. CPU only;
# the optional llm stage needs GPU encodings made beforehand (RUNBOOK 13.10).
# Threads: the parallel stages split the machine's cores (nproc) by a fixed share, so the peak never
# exceeds it: d3 13/32 (8 jobs), adapters 9/32 (14 jobs, one per pair), mass 2/32 (6 jobs), and
# trecdl, sample, xarch and llm 2/32 each (32/32; 212 threads on 224 cores). A stage run alone gets
# every core. THREADS=n forces n threads per job everywhere (your responsibility).
set -euo pipefail
EXPECTED_TESTS=${EXPECTED_TESTS:-81}   # preflight refuses if the suite shrank or grew
ROOT=$(pwd)
STAGE=${1:-all}
mkdir -p logs
TEXT_D3="P2 P3 P4 P5 P9 P10 P12 P13"           # D3 sends these to K'=25
TEXT_ALL="P2 P3 P4 P5 P9 P10 P11 P12 P13 P14"
CODE="P7 P8"
TEXT_SRC="all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5 bge-m3"
# code sources with the encodings their pairs used (pairs/P7/meta.json: 256; pairs/P8: 512)
CODE_ENC="enc_csn256/all-MiniLM-L6-v2 enc_csn512/codesage-small"
# Cores this container may use. `nproc` reports the HOST's CPUs inside many containers (AutoDL:
# 224 on a 20-vCPU instance), and sizing threads from it oversubscribed the quota ~11x on the
# 2026-11-09 runs (and throttled the latency kit). Read the cgroup quota first.
cpu_quota() {
  local q p
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    if [ "$q" != "max" ] && [ "${p:-0}" -gt 0 ]; then echo $(( (q + p - 1) / p )); return; fi
  fi
  if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [ "$q" -gt 0 ] && [ "$p" -gt 0 ]; then echo $(( (q + p - 1) / p )); return; fi
  fi
  nproc
}
NCORES=${NCORES:-$(cpu_quota)}
PARALLEL=0                                      # 1 while the stages share the machine (set by `all`)
XPAIRS="P5p P5q"                                # encoder->LLM pairs (Qwen3-Embedding-8B target)
XARCH=${XARCH:-1}                               # replay them in `all` (XARCH=0 skips)
LLM_SRC=${LLM_SRC:-enc/Qwen3-Embedding-0.6B}    # optional LLM->LLM pair P5r: source encodings (GPU, RUNBOOK 13.10)
LLM_TGT=${LLM_TGT:-enc/Qwen3-Embedding-8B}      # its target: the 1M Qwen3-8B encodings P5p/P5q use
log() { echo "[$(date -Is)] $*"; }
use_threads() {  # use_threads <jobs in this stage> <share of the cores, in 32nds, when stages run together>
  local jobs=$1 share=$2 t
  if [ -n "${THREADS:-}" ]; then t=$THREADS
  elif [ "$PARALLEL" = 1 ]; then t=$(( NCORES * share / 32 / jobs ))
  else t=$(( NCORES / jobs )); fi
  [ "$t" -ge 1 ] || t=1
  export OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t
  log "threads: $jobs job(s) x $t of $NCORES cores"
}
need_outputs() {  # need_outputs <stage> <min count> <glob>: a stage that exits 0 but writes nothing fails
  local n; n=$(ls $3 2>/dev/null | wc -l)
  [ "$n" -ge "$2" ] || { log "$1: expected at least $2 file(s) matching $3, found $n"; return 1; }
  log "$1: $n output file(s)"
}
init_registry() {  # write RUNS.tsv's header once, before parallel runs append to it
  python -c "import os,sys; sys.path.insert(0,'.'); from reusemap.provenance import REG_COLS as C; r=os.path.join(sys.argv[1],'RUNS.tsv'); os.path.exists(r) or open(r,'w').write(chr(9).join(C)+chr(10))" "$1"
}
llm_ready() {  # the P5r source vectors exist AND passed encode_check (its JSON record, for this folder)
  python -c "import json,os,sys; p='checks/Qwen3-Embedding-0.6B.json'; d=json.load(open(p)) if os.path.exists(p) else None; s=sys.argv[1]; sys.exit(0 if d and d.get('hard_failures')==0 and os.path.realpath(d['enc'])==os.path.realpath(s) and all(os.path.exists(os.path.join(s,f)) for f in ('doc.npy','q.npy')) else 1)" "$LLM_SRC"
}
FAILED=()
waitall() {  # wait on EVERY job (none is abandoned), record failures, return 1 if any failed
  local pid n0=${#FAILED[@]}
  for pid in "$@"; do wait "$pid" || FAILED+=("$pid"); done
  [ ${#FAILED[@]} -eq "$n0" ]
}

preflight() {
  log "preflight"
  [ "$(git rev-parse --abbrev-ref HEAD)" = "submission-eab" ] || { echo "check out submission-eab first"; exit 1; }
  [ -z "$(git status --porcelain -- reusemap scripts configs tests compat)" ] || { echo "uncommitted code: results would be ok_dirty"; exit 1; }
  # run_pair.py refuses to run when ANY tracked file differs from HEAD (git diff HEAD), not only
  # code: a hand-edited DECISIONS.md would pass the check above and then stop every replay
  [ -z "$(git diff HEAD --name-only)" ] || { echo "tracked files differ from HEAD (run_pair.py would refuse): $(git diff HEAD --name-only | tr '\n' ' ')-- commit or restore them"; exit 1; }
  local t; t=$(python tests/test_invariants.py | tail -1)
  [ "$t" = "$EXPECTED_TESTS tests passed" ] || { echo "tests: got '$t', expected '$EXPECTED_TESTS tests passed'"; exit 1; }
  echo "$t"
  python tests/test_runbook.py
  for d in enc pairs results coverage sweeps_fine sweeps_csn configs sub1m/query_sets subcsn/query_sets; do
    [ -e "$d" ] || { echo "missing $d"; exit 1; }
  done
  for p in $TEXT_ALL $CODE P0; do [ -d pairs/$p ] || { echo "missing pairs/$p"; exit 1; }; done
  python - <<'PY'
import json, sys
a = json.load(open("configs/msmarco_1m_v3.json")); b = json.load(open("configs/msmarco_1m_v3_d3.json"))
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
if diff != ["_preregistered_d3", "kprime_frac", "window_frac"] or abs(b["window_frac"] * 1e6 - 25) > 1e-9:
    sys.exit(f"configs/msmarco_1m_v3_d3.json is not the registered config at K'=25 (differs in {diff})")
print("d3 config: registered config with window K'=25")
PY
  for e in $CODE_ENC $(for m in $TEXT_SRC; do echo enc/$m; done); do
    for f in doc.npy q.npy; do [ -f $e/$f ] || { echo "missing $e/$f (encodings for the window-mass pass)"; exit 1; }; done
  done
  for f in sub1m/query_sets/trecdl.txt subcsn/query_sets/cosqa.txt; do [ -s $f ] || { echo "missing $f"; exit 1; }; done
  for p in $XPAIRS; do [ -f pairs/$p/meta.json ] || { echo "missing pairs/$p (encoder->LLM pair; adapters and xarch use it)"; exit 1; }; done
  if llm_ready; then echo "LLM->LLM pair: $LLM_SRC checked, the llm stage will run (target $LLM_TGT)"
  else echo "LLM->LLM pair: no $LLM_SRC/doc.npy, the llm stage is skipped here (extension_runs.sh runs it; RUNBOOK 13.10)"; fi
  log "preflight ok"
}

plan() {
  log "13.2a: plan, then commit it BEFORE any K'=25 replay"
  python -c "import json; w=json.load(open('results/P0.json'))['window']; print('plan input: results/P0.json at window', w, '' if w == 50 else '(NOT the final P0: run the p0 stage first)')"
  mkdir -p planner/cov
  python scripts/merge_frontier.py sweeps_fine/sweep_P*.json sweeps_csn/sweep_P*.json --out planner/frontier_all.json
  for f in coverage/*.json; do b=$(basename "$f"); cp "$f" "planner/cov/cov_${b#cov_}"; done  # plan.py globs cov_*.json; never doubles the prefix
  python scripts/plan.py results/P*.json --coverage-dir planner/cov --frontier planner/frontier_all.json --csv planner
  python - <<'PY'
import csv, sys
P = {r["pair"]: r for r in csv.DictReader(open("planner/plan.csv"))}
want25 = "P2 P3 P4 P5 P9 P10 P12 P13".split()
absent = [p for p in want25 + ["P0", "P8", "P11", "P14"] if p not in P]
if absent:
    sys.exit(f"plan.csv has no row for {absent}: plan.py writes a row per results/<pair>.json, so "
             f"check those files exist (ls results/P*.json) and re-run this stage")
bad = [p for p in want25 if (P[p]["family"], P[p]["kprime"]) != ("S2", "25")]
bad += [p for p in ("P0", "P8", "P14") if P[p]["family"] != "S0"]
bad += [] if (P["P11"]["family"], P["P11"]["kprime"]) == ("S2", "50") else ["P11"]
bad += [p + " (unpriced)" for p in want25 + ["P11"] if not P[p].get("cost")]
if bad:
    sys.exit(f"plan.csv is not the plan the paper describes, check: {bad}")
print("plan.csv: 8 pairs at K'=25, P11 at 50, P0/P8/P14 rebuild, all window plans priced -- as expected")
PY
  if git diff --quiet HEAD -- planner/plan.csv 2>/dev/null && git ls-files --error-unmatch planner/plan.csv >/dev/null 2>&1; then
    log "plan.csv already committed and unchanged"
  else
    git add -f planner/plan.csv planner/frontier_all.json
    git commit -q -m "D3 plan, committed before the K'=25 replays"
  fi
  git log -1 --format=%cI -- planner/plan.csv | tee logs/plan_committed_at.txt
}

d3() {
  [ -s logs/plan_committed_at.txt ] || { echo "run the plan stage first: the plan must be committed before the replays"; exit 1; }
  log "13.2b: $TEXT_D3 at K'=25 (per-pair logs: logs/d3_P*.log)"
  mkdir -p results_d3; init_registry results_d3; use_threads 8 13
  local pids=()
  for p in $TEXT_D3; do
    python scripts/run_pair.py pairs/$p --config configs/msmarco_1m_v3_d3.json --name $p \
      --out results_d3/$p.json > logs/d3_$p.log 2>&1 & pids+=($!)
  done
  waitall "${pids[@]}" || { log "13.2b: replay(s) failed -- see logs/d3_P*.log"; return 1; }
  need_outputs 13.2b 8 "results_d3/P*.json" || return 1
  log "13.2b done"
}

adapters() {
  # one process per pair: each pair runs ~15 full-corpus retrievals (one per fitted adapter), and
  # the single-threaded argpartition in each is the floor, so pairs in parallel, not threads
  log "13.1: adapter families (per-pair logs: logs/adapters_P*.log)"; use_threads 14 9
  # every file in adapters/ must come from this code: move earlier fits aside first (the first
  # run's affine/low-rank fits used a fixed ridge that collapsed the map; see compat.fit_affine)
  if ls adapters/adapters_P*.json >/dev/null 2>&1; then
    local old="adapters_superseded/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$old"; mv adapters/adapters_P*.json "$old"/
    log "13.1: moved earlier adapter fits to $old"
  fi
  mkdir -p adapters
  # ADAPTER_JOBS caps how many pairs fit at once (each holds two 1M-vector sets in memory; a
  # 90 GB container OOMs at 14). Default: all 14 together.
  local J=${ADAPTER_JOBS:-14} pids=() p cfg rc=0 n=0
  for p in $TEXT_ALL $CODE $XPAIRS; do
    case " $CODE " in *" $p "*) cfg=configs/csn_py_v3.json ;; *) cfg=configs/msmarco_1m_v3.json ;; esac
    python scripts/adapter_families.py pairs/$p --config $cfg --out-dir adapters > logs/adapters_$p.log 2>&1 & pids+=($!)
    n=$((n + 1))
    if [ "$n" -ge "$J" ]; then waitall "${pids[@]}" || rc=1; pids=(); n=0; fi
  done
  [ ${#pids[@]} -eq 0 ] || waitall "${pids[@]}" || rc=1
  [ "$rc" = 0 ] || { log "13.1: adapter fit(s) failed -- see logs/adapters_P*.log"; return 1; }
  need_outputs 13.1 14 "adapters/adapters_P*.json" || return 1
  log "13.1 done"
}

p0() {
  log "13.3: P0 under the final config"
  mkdir -p superseded; use_threads 1 32
  [ -f superseded/P0_window500.json ] || cp results/P0.json superseded/P0_window500.json
  cat > superseded/README.txt <<'EOF'
P0_window500.json is the identity control from the window-500 pass, replaced by results/P0.json at the
final window (K'=50). The p0 stage runs before the plan stage, so planner/plan.csv is computed from the
final P0. D1 sends the identity control to S0 (rebuild) at any window, so P0's plan row does not depend
on which of the two files it was computed from. Do not use this file for any reported number.
EOF
  if ! python scripts/run_pair.py pairs/P0 --config configs/msmarco_1m_v3.json --name P0 --identity \
       --out results/P0.json --force; then
    cp superseded/P0_window500.json results/P0.json
    log "13.3 failed: results/P0.json restored to the window-500 run. results/RUNS.tsv keeps the failed"
    log "     run's row as a record; the paper's P0 is then the restored run, whose own row is ok"; return 1
  fi
  if ! python -c "import json,sys; r=json.load(open('results/P0.json')); sys.exit(0 if r['window']==50 else 'P0 window is %s, not 50' % r['window'])"; then
    cp superseded/P0_window500.json results/P0.json
    log "13.3 failed: re-run is not at window 50; results/P0.json restored"; return 1
  fi
  log "13.3 done"
}

mass() {
  log "13.5: window multiplicities"
  mkdir -p window_mass window_mass_csn; use_threads 6 2
  local pids=()
  for M in $TEXT_SRC; do
    python scripts/window_mass.py --docs enc/$M/doc.npy --queries enc/$M/q.npy \
      --exclude-qids sub1m/query_sets/trecdl.txt --kprime 25,50,100,200,500 \
      --zipf 0.6,0.9,1.2 --sample-fracs 0.1,0.25,0.5 --name $M --out window_mass/$M.json & pids+=($!)
  done
  for E in $CODE_ENC; do
    M=$(basename $E)
    python scripts/window_mass.py --docs $E/doc.npy --queries $E/q.npy \
      --exclude-qids subcsn/query_sets/cosqa.txt --kprime 25,50,100,200 \
      --name $M --out window_mass_csn/$M.json & pids+=($!)
  done
  waitall "${pids[@]}" || { log "13.5: window_mass failed -- see logs/mass.log"; return 1; }
  need_outputs 13.5 4 "window_mass/*.json" && need_outputs 13.5 2 "window_mass_csn/*.json" || return 1
  python scripts/coverage_curve_check.py --window-mass window_mass --results results --md curve_check.md
  python scripts/coverage_curve_check.py --window-mass window_mass_csn --results results --md curve_check_csn.md
  log "13.5 done"
}

trecdl() {
  log "TREC-DL G at K'=25/50 for P11 and P14 (measure-only; the d3 replays carry the other eight)"
  mkdir -p results_measure_d3; init_registry results_measure_d3; use_threads 2 2
  local pids=()
  for p in P11 P14; do
    python scripts/run_pair.py pairs/$p --config configs/msmarco_1m_v3_d3.json --name $p --measure-only \
      --out results_measure_d3/$p.json > logs/trecdl_$p.log 2>&1 & pids+=($!)
  done
  waitall "${pids[@]}" || { log "trecdl: measure-only run failed -- see logs/trecdl_P*.log"; return 1; }
  need_outputs trecdl 2 "results_measure_d3/P*.json" || return 1
  log "trecdl done"
}

sample() {
  log "13.6: D1 sample-size sweep"; use_threads 1 2
  python scripts/sample_sweep.py $(for p in $TEXT_ALL; do echo pairs/$p; done) \
    --config configs/msmarco_1m_v3.json --results results --fracs 0.01,0.05,0.25 --seeds 3 --out-dir sample_sweep
  python scripts/sample_sweep.py $(for p in $CODE; do echo pairs/$p; done) \
    --config configs/csn_py_v3.json --results results --fracs 0.01,0.05,0.25 --seeds 3 --out-dir sample_sweep
  need_outputs 13.6 12 "sample_sweep/*.json" || return 1
  log "13.6 done"
}

xarch() {
  log "xarch: full replays of the encoder->LLM pairs P5p, P5q (XARCH=0 skips)"
  mkdir -p results_xarch; init_registry results_xarch; use_threads 2 2
  local pids=()
  for p in $XPAIRS; do
    python scripts/run_pair.py pairs/$p --config configs/msmarco_1m_v3.json --name $p \
      --out results_xarch/$p.json > logs/xarch_$p.log 2>&1 & pids+=($!)
  done
  waitall "${pids[@]}" || { log "xarch: replay failed -- see logs/xarch_P*.log"; return 1; }
  need_outputs xarch 2 "results_xarch/P*.json" || return 1
  log "xarch done"
}

llm() {  # OPTIONAL LLM->LLM pair P5r (RUNBOOK 13.10): runs only once the 0.6B encodings exist
  log "13.10: LLM->LLM pair P5r = $LLM_SRC -> $LLM_TGT (static measurements + adapter families)"
  use_threads 1 2
  llm_ready || { log "13.10: $LLM_SRC has no passing encode_check record (checks/Qwen3-Embedding-0.6B.json); not running P5r on it"; return 1; }
  for f in "$LLM_SRC/doc.npy" "$LLM_SRC/q.npy" "$LLM_TGT/doc.npy" "$LLM_TGT/q.npy"; do
    [ -f "$f" ] || { log "13.10: missing $f"; return 1; }
  done
  if [ ! -f pairs/P5r/meta.json ]; then
    python scripts/assemble_pair.py --src "$LLM_SRC" --tgt "$LLM_TGT" --data sub1m --bm25 sub1m/bm25.npy --out pairs/P5r \
      --meta '{"source":"Qwen/Qwen3-Embedding-0.6B","target":"Qwen/Qwen3-Embedding-8B","corpus":"msmarco-passage 1M","max_seq_len":512}'
  fi
  mkdir -p results_llm coverage_llm; init_registry results_llm
  python scripts/coverage.py --docs "$LLM_SRC/doc.npy" --queries "$LLM_SRC/q.npy" \
    --exclude-qids sub1m/query_sets/trecdl.txt --kprime 25,50,100,200,350,500,1000,2000 \
    --out coverage_llm/Qwen3-Embedding-0.6B.json --name Qwen3-Embedding-0.6B
  # measure-only under the d3 config: its ladder carries G and rho at K' = 25, 50, 500, ...
  python scripts/run_pair.py pairs/P5r --config configs/msmarco_1m_v3_d3.json --name P5r --measure-only \
    --out results_llm/P5r.json
  python scripts/adapter_families.py pairs/P5r --config configs/msmarco_1m_v3.json --out-dir adapters
  need_outputs 13.10 1 "results_llm/P5r.json" && need_outputs 13.10 1 "coverage_llm/*.json" \
    && need_outputs 13.10 1 "adapters/adapters_P5r.json" || return 1
  log "13.10 done"
}

check() {
  log "checks"
  if [ ! -f results_d3/RUNS.tsv ]; then
    log "no K'=25 replays in results_d3/ yet: run the d3 stage first"
  else
  python scripts/check_planner.py --plan planner/plan.csv --results results_d3 \
    --committed-at "$(cat logs/plan_committed_at.txt)" --runs results_d3/RUNS.tsv --runs-tz "$(date +%:z)" \
    --md planner_check.md || log "planner check reported failures: read planner_check.md (report them, do not re-plan)"
  fi
  # Every check runs and reports, whatever an earlier one said: under set -e -o pipefail a failing
  # check would otherwise end this function before the next one ran (check_tables failing used to
  # skip freeze_check, leaving an old FREEZE_CHECK.md in place), and `check && collect` would then
  # not collect. A failure is logged; the function still returns non-zero at the end.
  local bad=0
  python scripts/summarize.py results/P*.json --exclude P0 P6 --csv results/tables > logs/summarize.log \
    || { bad=1; log "summarize failed: read logs/summarize.log"; }
  python scripts/tau_curve.py results/P*.json --csv results/tables > logs/tau_curve.log \
    || { bad=1; log "tau_curve failed: read logs/tau_curve.log"; }
  python scripts/check_tables.py results/tables results/P*.json | tail -2 \
    || { bad=1; log "check_tables: the tables do not match the results (see above)"; }
  python scripts/freeze_check.py $(for p in $TEXT_ALL; do echo results/$p.json; done) sweeps_fine/sweep_P*.json \
    --out FREEZE_CHECK.md | tail -1 || log "freeze check did not certify K*: read FREEZE_CHECK.md"
  if [ -d window_mass ]; then
    python scripts/coverage_curve_check.py --window-mass window_mass --results results --md curve_check.md | tail -1 \
      || { bad=1; log "coverage_curve_check failed: read curve_check.md"; }
  fi
  log "checks done$([ $bad = 0 ] || echo ' (with failures: see above)')"
  return $bad
}

collect() {
  log "collect"
  out=submission_outputs_$(date +%Y%m%d_%H%M).zip
  local pjson=(); for f in results/P*.json; do [ -e "$f" ] && pjson+=("$f"); done
  [ ${#pjson[@]} -gt 0 ] || log "no results/P*.json: the zip cannot reproduce the tables"
  local want=(planner planner_check.md results_d3 adapters "${pjson[@]}" superseded results/tables
              results/RUNS.tsv FREEZE_CHECK.md window_mass window_mass_csn curve_check.md curve_check_csn.md
              results_measure_d3 sample_sweep results_xarch results_llm coverage_llm logs) have=() missing=()
  for f in "${want[@]}"; do if [ -e "$f" ]; then have+=("$f"); else missing+=("$f"); fi; done
  [ ${#missing[@]} -eq 0 ] || log "not produced (stage failed or not run): ${missing[*]}"
  zip -qr "$out" "${have[@]}" || { log "zip failed: $out not written"; return 1; }
  [ -s "$out" ] || { log "$out is empty"; return 1; }
  log "wrote $out ($(du -h "$out" | cut -f1)) -- send this file"
}

case "$STAGE" in
  all)
    preflight
    # P0 first and alone: the plan is then computed from the final P0, and nothing running in
    # parallel reads results/P0.json while it is being rewritten. If it fails, the window-500 file
    # is restored, the failure is recorded, and the rest still runs.
    log "p0 (serial, before the plan): logs/p0.log"
    p0 > logs/p0.log 2>&1 || { FAILED+=(p0); log "p0 failed: plan uses the window-500 P0 (see logs/p0.log)"; }
    plan
    PARALLEL=1; jobs_=()
    ( d3 > logs/d3.log 2>&1 ) & jobs_+=($!)
    ( adapters > logs/adapters.log 2>&1 ) & jobs_+=($!)
    ( mass > logs/mass.log 2>&1 ) & jobs_+=($!)
    ( trecdl > logs/trecdl.log 2>&1 ) & jobs_+=($!)
    ( sample > logs/sample.log 2>&1 ) & jobs_+=($!)
    if [ "$XARCH" = 1 ]; then ( xarch > logs/xarch.log 2>&1 ) & jobs_+=($!); fi
    if llm_ready; then ( llm > logs/llm.log 2>&1 ) & jobs_+=($!)
    else log "llm: skipped here, $LLM_SRC not encoded and checked yet (extension_runs.sh runs it; RUNBOOK 13.10)"; fi
    waitall "${jobs_[@]}" || true   # failures are in FAILED; still check and collect what finished
    if [ ${#FAILED[@]} -gt 0 ]; then log "${#FAILED[@]} stage(s) failed: read logs/*.log; running check/collect on what finished"; fi
    check || true; collect
    [ ${#FAILED[@]} -eq 0 ] ;;
  preflight|plan|d3|adapters|p0|mass|trecdl|sample|xarch|llm|check|collect) "$STAGE" ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
