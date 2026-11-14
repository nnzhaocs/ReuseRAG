#!/usr/bin/env bash
# Extension experiments for the ICDE 2027 submission. Run AFTER scripts/submission_runs.sh, on the
# same instance and branch, with the same committed code. RUNBOOK.md Phase 14 is the guide.
#
#   bash scripts/extension_runs.sh <stage>
#
#   preflight  branch, clean tree, tests, inputs
#   encode     GPU: the four text encoders missing on NQ (MiniLM exists from P5c) + encode_check
#   bm25       BM25 top-1000 for the NQ queries
#   pairs      N0 (identity) and N2..N14: the ten MS MARCO text pairs, on NQ
#   measure    the planner's static inputs: coverage per source, sweeps, measure-only runs
#   plan       D1-D3 on NQ; COMMITS planner_nq/plan.csv before any NQ replay
#   replay     each pair at the depth its committed plan names; adapter families
#   seeds      P11, the closest call on MS MARCO, at seeds 1-4 (robustness; dev-override runs)
#   zipf       P2, P5, P11, P13 replayed under a pre-registered Zipf(0.9) workload; Theorem 2's bound
#              checked on their curves (it holds for any workload)
#   qsweep     D1/D3 recomputed from q = 50..2000 judged queries: how many does the planner need?
#   encode_llm GPU: Qwen3-Embedding-0.6B on MS MARCO, for the LLM->LLM pair P5r (+ encode_check)
#   p5r        the LLM->LLM pair P5r (Qwen3-0.6B -> 8B): runs submission_runs.sh's llm stage
#   throughput GPU: measured encoder throughput and inline-repair latency (optional; needs torch)
#   core       ONLY the NQ prediction: preflight, encode, bm25, pairs, measure, plan, replay,
#              check, collect. Use it if time runs short; everything else is a strengthening.
#   check      planner check on NQ, tables, tau curve, freeze check
#   collect    extension_outputs_<date>.zip
#   all        preflight; GPU in priority order: encode (NQ, required), encode_llm (P5r), throughput;
#              then bm25, pairs, measure, plan; then replay, seeds, zipf, qsweep and p5r in parallel;
#              then check and collect. A failed encode_llm or throughput never blocks the NQ test.
#
# Expected on the 224-core instance: encode ~20-40 GPU-min, throughput ~10 GPU-min; the rest
# ~1.5-2 h CPU, bound by the MS MARCO replays (P11 seeds, Zipf; 34-73 min each at K'=50).
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-}
EXPECTED_TESTS=${EXPECTED_TESTS:-81}
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
PARALLEL=0
FAILED=()
NQ_DATA=${NQ_DATA:-subnq}
NQ_ENC=${NQ_ENC:-enc_nq}
NQ_MAXLEN=${NQ_MAXLEN:-512}                     # what enc_nq/all-MiniLM-L6-v2 (P5c) was encoded at
NQ_PAIRS="N0 N2 N3 N4 N5 N9 N10 N11 N12 N13 N14"
SEEDS="1 2 3 4"
ZIPF_PAIRS="P2 P5 P11 P13"
QSWEEP_PAIRS="P2 P3 P4 P5 P9 P10 P11 P12 P13 P14"
LLM_SRC=${LLM_SRC:-enc/Qwen3-Embedding-0.6B}
declare -A HF=( [all-MiniLM-L6-v2]=sentence-transformers/all-MiniLM-L6-v2 [bge-small-en-v1.5]=BAAI/bge-small-en-v1.5
                [bge-base-en-v1.5]=BAAI/bge-base-en-v1.5 [bge-m3]=BAAI/bge-m3 [e5-base-v2]=intfloat/e5-base-v2 )
# the ten MS MARCO text pairs (pairs/P*/meta.json), same names with N for NQ; N0 is the identity control
declare -A SRC=( [N0]=bge-base-en-v1.5 [N2]=all-MiniLM-L6-v2 [N3]=bge-small-en-v1.5 [N4]=bge-m3 [N5]=bge-m3
                 [N9]=all-MiniLM-L6-v2 [N10]=all-MiniLM-L6-v2 [N11]=all-MiniLM-L6-v2 [N12]=bge-m3
                 [N13]=bge-small-en-v1.5 [N14]=bge-base-en-v1.5 )
declare -A TGT=( [N0]=bge-base-en-v1.5 [N2]=bge-base-en-v1.5 [N3]=bge-base-en-v1.5 [N4]=bge-base-en-v1.5
                 [N5]=bge-small-en-v1.5 [N9]=bge-small-en-v1.5 [N10]=bge-m3 [N11]=e5-base-v2 [N12]=e5-base-v2
                 [N13]=e5-base-v2 [N14]=e5-base-v2 )
NQ_SOURCES="all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5 bge-m3"
log() { echo "[$(date -Is)] $*"; }
use_threads() {  # use_threads <jobs> <share in 32nds when stages run together>
  local jobs=$1 share=$2 t
  if [ -n "${THREADS:-}" ]; then t=$THREADS
  elif [ "$PARALLEL" = 1 ]; then t=$(( NCORES * share / 32 / jobs ))
  else t=$(( NCORES / jobs )); fi
  [ "$t" -ge 1 ] || t=1
  export OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t
  log "threads: $jobs job(s) x $t of $NCORES cores"
}
need_outputs() {  # need_outputs <stage> <min count> <glob>
  local n; n=$(ls $3 2>/dev/null | wc -l)
  [ "$n" -ge "$2" ] || { log "$1: expected at least $2 file(s) matching $3, found $n"; return 1; }
  log "$1: $n output file(s)"
}
init_registry() {
  python -c "import os,sys; sys.path.insert(0,'.'); from reusemap.provenance import REG_COLS as C; r=os.path.join(sys.argv[1],'RUNS.tsv'); os.path.exists(r) or open(r,'w').write(chr(9).join(C)+chr(10))" "$1"
}
waitall() {
  local pid n0=${#FAILED[@]}
  for pid in "$@"; do wait "$pid" || FAILED+=("$pid"); done
  [ ${#FAILED[@]} -eq "$n0" ]
}
llm_ready() {  # the P5r source vectors exist AND passed encode_check (its JSON record, for this folder)
  python -c "import json,os,sys; p='checks/Qwen3-Embedding-0.6B.json'; d=json.load(open(p)) if os.path.exists(p) else None; s=sys.argv[1]; sys.exit(0 if d and d.get('hard_failures')==0 and os.path.realpath(d['enc'])==os.path.realpath(s) and all(os.path.exists(os.path.join(s,f)) for f in ('doc.npy','q.npy')) else 1)" "$LLM_SRC"
}
cfg_for() {  # cfg_for <K'>: the pre-registered NQ config for that window depth
  if [ "$1" = 50 ]; then echo configs/nq_100k_v3.json; else echo "configs/nq_100k_v3_k$1.json"; fi
}

preflight() {
  [ "$(git branch --show-current)" = submission-eab ] || { echo "not on branch submission-eab"; exit 1; }
  [ -z "$(git status --porcelain -- reusemap scripts configs tests compat)" ] || { echo "uncommitted code: results would be ok_dirty"; exit 1; }
  [ -z "$(git diff HEAD --name-only)" ] || { echo "tracked files differ from HEAD (run_pair.py would refuse): $(git diff HEAD --name-only | tr '\n' ' ')"; exit 1; }
  local t; t=$(python tests/test_invariants.py | tail -1)
  [ "$t" = "$EXPECTED_TESTS tests passed" ] || { echo "tests: got '$t', expected '$EXPECTED_TESTS tests passed'"; exit 1; }
  for f in corpus.tsv queries.tsv qrels.tsv; do [ -s "$NQ_DATA/$f" ] || { echo "missing $NQ_DATA/$f (scripts/build_nq_prefix.py output)"; exit 1; }; done
  for K in 25 50 100 200 500 1000; do [ -f "$(cfg_for $K)" ] || { echo "missing $(cfg_for $K)"; exit 1; }; done
  [ -f configs/msmarco_1m_v3_zipf.json ] || { echo "missing configs/msmarco_1m_v3_zipf.json"; exit 1; }
  local p; for p in $QSWEEP_PAIRS; do [ -f pairs/$p/meta.json ] || { echo "missing pairs/$p"; exit 1; }; done
  local M miss=""
  for M in "${!HF[@]}"; do [ -f "$NQ_ENC/$M/doc.npy" ] && [ -f "$NQ_ENC/$M/q.npy" ] || miss+=" $M"; done
  if [ -n "$miss" ]; then echo "NQ encodings missing (the encode stage makes them on the GPU):$miss"
  else echo "NQ encodings: all five present"; fi
  if llm_ready; then echo "P5r: $LLM_SRC encoded and checked"
  else echo "P5r: $LLM_SRC not encoded and checked yet (encode_llm does it on the GPU)"; fi
  log "preflight ok"
}

encode() {  # GPU. Skips a model whose encodings exist; a failed check stops the stage.
  log "encode: NQ encodings at max-seq-len $NQ_MAXLEN"
  mkdir -p checks_nq
  local M
  for M in "${!HF[@]}"; do
    if [ -f "$NQ_ENC/$M/doc.npy" ] && [ -f "$NQ_ENC/$M/q.npy" ]; then log "  $M: present"; continue; fi
    local ov=()
    [ "$M" = all-MiniLM-L6-v2 ] && ov=(--override "{\"max_seq_len_cap\": $NQ_MAXLEN}")   # as P5c's MiniLM
    python scripts/encode.py --model "${HF[$M]}" --kind doc --tsv "$NQ_DATA/corpus.tsv" --out "$NQ_ENC/$M/doc.npy" --max-seq-len "$NQ_MAXLEN" "${ov[@]}" \
      || { log "encode: $M document encoding failed"; return 1; }
    python scripts/encode.py --model "${HF[$M]}" --kind query --tsv "$NQ_DATA/queries.tsv" --out "$NQ_ENC/$M/q.npy" --max-seq-len "$NQ_MAXLEN" "${ov[@]}" \
      || { log "encode: $M query encoding failed"; return 1; }
    python scripts/encode_check.py --enc "$NQ_ENC/$M" --data "$NQ_DATA" --json "checks_nq/$M.json" \
      || { log "encode: $M failed encode_check -- see checks_nq/$M.json; stop here"; return 1; }
  done
  log "encode done"
}

encode_llm() {  # GPU: the source side of the LLM->LLM pair P5r, at the 512 tokens the 8B vectors used
  if llm_ready; then log "encode_llm: $LLM_SRC encoded and checked"; return 0; fi
  mkdir -p checks
  if [ -f "$LLM_SRC/doc.npy" ] && [ -f "$LLM_SRC/q.npy" ]; then
    log "encode_llm: $LLM_SRC exists without a passing check record; checking it (no re-encode)"
  else
    log "encode_llm: Qwen/Qwen3-Embedding-0.6B on MS MARCO 1M"
    python scripts/encode.py --model Qwen/Qwen3-Embedding-0.6B --kind doc --tsv sub1m/corpus.tsv \
      --out "$LLM_SRC/doc.npy" --max-seq-len 512 --batch 128 || { log "encode_llm: document encoding failed"; return 1; }
    python scripts/encode.py --model Qwen/Qwen3-Embedding-0.6B --kind query --tsv sub1m/queries.tsv \
      --out "$LLM_SRC/q.npy" --max-seq-len 512 --batch 128 || { log "encode_llm: query encoding failed"; return 1; }
  fi
  if ! python scripts/encode_check.py --enc "$LLM_SRC" --data sub1m --json checks/Qwen3-Embedding-0.6B.json; then
    # a query file made under the old Qwen3 contract fails only the contract check: the documents
    # are fine, so keep them and let qwen_rerun.sh re-encode the queries instead
    if python -c "import json,sys; d=json.load(open('checks/Qwen3-Embedding-0.6B.json')); f=[c['check'] for c in d['checks'] if c['hard'] and not c['ok']]; sys.exit(0 if f==['query: encoded under the current contract'] else 1)"; then
      log "encode_llm: $LLM_SRC queries predate the Qwen3 contract fix; run 'bash scripts/qwen_rerun.sh requery' (documents kept)"
    else
      log "encode_llm: encode_check failed; P5r is skipped (moving the vectors aside)"; mv "$LLM_SRC" "$LLM_SRC.failed_check"
    fi
    return 1
  fi
  log "encode_llm done"
}

p5r() {  # the llm stage of submission_runs.sh, with this run's thread share
  llm_ready || { log "p5r: $LLM_SRC not encoded and checked; skipped"; return 0; }
  if [ -f results_llm/P5r.json ] && [ -f adapters/adapters_P5r.json ]; then
    log "p5r: already done (results_llm/P5r.json, adapters/adapters_P5r.json)"; return 0
  fi
  local t=$NCORES; [ "$PARALLEL" = 1 ] && t=$(( NCORES * 2 / 32 )); [ "$t" -ge 1 ] || t=1
  THREADS=${THREADS:-$t} LLM_SRC="$LLM_SRC" bash scripts/submission_runs.sh llm
}

bm25() {
  if [ -f "$NQ_DATA/bm25.npy" ]; then log "bm25: $NQ_DATA/bm25.npy present"; return 0; fi
  log "bm25: NQ"; use_threads 1 32
  python scripts/bm25_run.py --corpus "$NQ_DATA/corpus.tsv" --queries "$NQ_DATA/queries.tsv" --k 1000 --out "$NQ_DATA/bm25.npy"
}

pairs() {
  log "pairs: $NQ_PAIRS"
  local p
  for p in $NQ_PAIRS; do
    python scripts/assemble_pair.py --src "$NQ_ENC/${SRC[$p]}" --tgt "$NQ_ENC/${TGT[$p]}" --data "$NQ_DATA" \
      --bm25 "$NQ_DATA/bm25.npy" --out "pairs/$p" \
      --meta "{\"source\":\"${HF[${SRC[$p]}]}\",\"target\":\"${HF[${TGT[$p]}]}\",\"corpus\":\"BEIR NQ 100k\",\"max_seq_len\":$NQ_MAXLEN}"
  done
  need_outputs pairs 11 "pairs/N*/meta.json"
}

measure() {  # the planner's static inputs, nothing replayed
  log "measure: coverage, sweeps, measure-only runs (per-pair logs: logs/nq_*_N*.log)"
  mkdir -p coverage_nq sweeps_nq results_nq_static; init_registry results_nq_static
  local M p pids=(); use_threads 15 32
  for M in $NQ_SOURCES; do
    python scripts/coverage.py --docs "$NQ_ENC/$M/doc.npy" --queries "$NQ_ENC/$M/q.npy" \
      --kprime 10,25,50,100,200,500,1000 --out "coverage_nq/$M.json" --name "$M" > "logs/nq_cov_$M.log" 2>&1 & pids+=($!)
  done
  for p in $NQ_PAIRS; do
    [ "$p" = N0 ] && continue
    python scripts/sweep.py "pairs/$p" --config configs/nq_100k_v3.json --kprime 10,25,50,100,200,500,1000 \
      --out-dir "sweeps_nq/$p" > "logs/nq_sweep_$p.log" 2>&1 & pids+=($!)
  done
  waitall "${pids[@]}" || { log "measure: coverage or sweep failed -- see logs/nq_*.log"; return 1; }
  for p in $NQ_PAIRS; do [ "$p" = N0 ] || cp "sweeps_nq/$p/sweep_$p.json" sweeps_nq/; done
  pids=()
  for p in $NQ_PAIRS; do
    local id=(); [ "$p" = N0 ] && id=(--identity)
    python scripts/run_pair.py "pairs/$p" --config configs/nq_100k_v3.json --name "$p" --measure-only "${id[@]}" \
      --out "results_nq_static/$p.json" > "logs/nq_static_$p.log" 2>&1 & pids+=($!)
  done
  waitall "${pids[@]}" || { log "measure: measure-only run failed -- see logs/nq_static_N*.log"; return 1; }
  need_outputs measure 4 "coverage_nq/*.json" && need_outputs measure 10 "sweeps_nq/sweep_N*.json" \
    && need_outputs measure 11 "results_nq_static/N*.json" || return 1
  log "measure done"
}

plan() {  # D1-D3 from static inputs only; committed before any NQ replay
  log "plan: NQ, then commit BEFORE any NQ replay"
  if [ -f results_nq/RUNS.tsv ] && [ "$(wc -l < results_nq/RUNS.tsv)" -gt 1 ]; then
    # replays exist: keep the plan they were scored against (a re-run of 'all' after a partial
    # failure lands here); never compute a new one, which would be post-hoc
    if [ -s logs/nq_plan_committed_at.txt ] && git ls-files --error-unmatch planner_nq/plan.csv >/dev/null 2>&1 \
       && git diff --quiet HEAD -- planner_nq/plan.csv; then
      log "plan: NQ replays exist; keeping the plan committed at $(cat logs/nq_plan_committed_at.txt)"; return 0
    fi
    log "plan: NQ replays exist but no unchanged committed plan; a new plan now would be post-hoc. Refusing."; return 1
  fi
  mkdir -p planner_nq/cov
  python scripts/merge_frontier.py sweeps_nq/sweep_N*.json --out planner_nq/frontier_all.json
  local f b
  for f in coverage_nq/*.json; do b=$(basename "$f"); cp "$f" "planner_nq/cov/cov_${b#cov_}"; done
  python scripts/plan.py results_nq_static/N*.json --coverage-dir planner_nq/cov --frontier planner_nq/frontier_all.json \
    --config configs/nq_100k_v3.json --csv planner_nq
  python - <<'PY'
import csv, sys
P = {r["pair"]: r for r in csv.DictReader(open("planner_nq/plan.csv"))}
want = "N0 N2 N3 N4 N5 N9 N10 N11 N12 N13 N14".split()
bad = [p for p in want if p not in P or not P[p]["family"]]
bad += [f"{p} (S2 at K'={P[p]['kprime']}: no pre-registered config)" for p in want
        if p in P and P[p]["family"] == "S2" and P[p]["kprime"] not in ("25", "50", "100", "200", "500", "1000")]
bad += [f"{p} (unpriced)" for p in want if p in P and P[p]["family"] == "S2" and not P[p].get("cost")]
if bad:
    sys.exit("plan_nq is not complete, refusing to commit: " + ", ".join(bad))
print("plan_nq: " + ", ".join(f"{p} {P[p]['family']}" + (f"@{P[p]['kprime']}" if P[p]["family"] == "S2" else "") for p in want))
PY
  git add -f planner_nq/plan.csv planner_nq/frontier_all.json
  git diff --cached --quiet || git commit -q -m "NQ plan (D1-D3 from static inputs), committed before any NQ replay"
  git log -1 --format=%cI -- planner_nq/plan.csv > logs/nq_plan_committed_at.txt
  log "plan committed at $(cat logs/nq_plan_committed_at.txt)"
}

replay() {  # each pair at the depth the committed plan names; S0/S1 plans at the default 50
  [ -s logs/nq_plan_committed_at.txt ] || { log "replay: no committed NQ plan (run the plan stage first)"; return 1; }
  log "replay: NQ pairs at their planned depths (per-pair logs: logs/nq_replay_N*.log)"
  mkdir -p results_nq adapters_nq; init_registry results_nq
  local n_jobs; n_jobs=$(( $(echo $NQ_PAIRS | wc -w) * 2 ))
  use_threads "$n_jobs" 10
  local p K id pids=()
  for p in $NQ_PAIRS; do
    K=$(python -c "import csv,sys; P={r['pair']:r for r in csv.DictReader(open('planner_nq/plan.csv'))}; r=P[sys.argv[1]]; print(r['kprime'] if r['family']=='S2' else 50)" "$p")
    id=(); [ "$p" = N0 ] && id=(--identity)
    log "  $p at K'=$K ($(cfg_for "$K"))"
    python scripts/run_pair.py "pairs/$p" --config "$(cfg_for "$K")" --name "$p" "${id[@]}" \
      --out "results_nq/$p.json" > "logs/nq_replay_$p.log" 2>&1 & pids+=($!)
    if [ "$p" != N0 ]; then
      python scripts/adapter_families.py "pairs/$p" --config configs/nq_100k_v3.json --out-dir adapters_nq \
        > "logs/nq_adapters_$p.log" 2>&1 & pids+=($!)
    fi
  done
  waitall "${pids[@]}" || { log "replay: a replay or adapter fit failed -- see logs/nq_replay_N*.log, logs/nq_adapters_N*.log"; return 1; }
  need_outputs replay 11 "results_nq/N*.json" && need_outputs replay 10 "adapters_nq/adapters_N*.json" || return 1
  log "replay done"
}

seeds() {  # robustness of the closest MS MARCO call; --set marks them dev-override runs, not paper-table data
  log "seeds: P11 at seeds $SEEDS (per-seed logs: logs/seed_P11_*.log)"
  mkdir -p results_seeds; init_registry results_seeds
  use_threads 4 8
  local s pids=()
  for s in $SEEDS; do
    python scripts/run_pair.py pairs/P11 --config configs/msmarco_1m_v3.json --name "P11_seed$s" --set "seed=$s" \
      --out "results_seeds/P11_seed$s.json" > "logs/seed_P11_$s.log" 2>&1 & pids+=($!)
  done
  waitall "${pids[@]}" || { log "seeds: a replay failed -- see logs/seed_P11_*.log"; return 1; }
  need_outputs seeds 4 "results_seeds/P11_seed*.json" || return 1
  log "seeds done"
}

zipf() {  # the paper's only workload was uniform; same pairs, same everything, Zipf(0.9) arrivals
  log "zipf: $ZIPF_PAIRS under configs/msmarco_1m_v3_zipf.json (per-pair logs: logs/zipf_P*.log)"
  mkdir -p results_zipf; init_registry results_zipf
  use_threads 4 6
  local p pids=()
  for p in $ZIPF_PAIRS; do
    python scripts/run_pair.py "pairs/$p" --config configs/msmarco_1m_v3_zipf.json --name "$p" \
      --out "results_zipf/$p.json" > "logs/zipf_$p.log" 2>&1 & pids+=($!)
  done
  waitall "${pids[@]}" || { log "zipf: a replay failed -- see logs/zipf_P*.log"; return 1; }
  need_outputs zipf 4 "results_zipf/P*.json" || return 1
  python scripts/coverage_curve_check.py --bound-only --results results_zipf --md curve_check_zipf.md | tail -1
  log "zipf done"
}

qsweep() {  # how many judged queries D1 and D3 need, from the vectors that exist
  log "qsweep: $QSWEEP_PAIRS (per-pair logs: logs/qsweep_P*.log)"
  mkdir -p query_sweep
  use_threads 10 6
  local p pids=() planarg=()
  [ -f planner/plan.csv ] && planarg=(--plan planner/plan.csv)
  for p in $QSWEEP_PAIRS; do
    python scripts/query_sweep.py "pairs/$p" --config configs/msmarco_1m_v3.json --q 50,100,250,500,1000,2000 \
      --seeds 5 "${planarg[@]}" --out-dir query_sweep > "logs/qsweep_$p.log" 2>&1 & pids+=($!)
  done
  waitall "${pids[@]}" || { log "qsweep: failed -- see logs/qsweep_P*.log"; return 1; }
  need_outputs qsweep 10 "query_sweep/P*.json" || return 1
  log "qsweep done"
}

throughput() {  # GPU. Optional: a failure here never blocks the measurements
  log "throughput: the MS MARCO target encoders at 256 tokens, Qwen3-8B at 512"
  python scripts/encode_throughput.py --tsv sub1m/corpus.tsv --max-seq-len 256 \
    --model BAAI/bge-base-en-v1.5 --model BAAI/bge-small-en-v1.5 --model BAAI/bge-m3 --model intfloat/e5-base-v2 \
    --out throughput.json
  python scripts/encode_throughput.py --tsv sub1m/corpus.tsv --max-seq-len 512 --batch 16 \
    --model Qwen/Qwen3-Embedding-8B --out throughput_qwen8b.json || log "throughput: Qwen3-8B skipped (see log)"
  log "throughput done"
}

check() {
  log "checks (NQ)"
  local bad=0   # every check runs and reports; see submission_runs.sh check()
  if [ -s logs/nq_plan_committed_at.txt ] && [ -f results_nq/RUNS.tsv ]; then
    python scripts/check_planner.py --plan planner_nq/plan.csv --results results_nq \
      --committed-at "$(cat logs/nq_plan_committed_at.txt)" --runs results_nq/RUNS.tsv --runs-tz "$(date +%:z)" \
      --md planner_check_nq.md || log "NQ planner check reported failures: read planner_check_nq.md (report them, do not re-plan)"
  fi
  if ls results_nq/N*.json > /dev/null 2>&1; then
    mkdir -p results_nq/tables
    python scripts/summarize.py results_nq/N*.json --exclude N0 --csv results_nq/tables > logs/nq_summarize.log \
      || { bad=1; log "NQ summarize failed: read logs/nq_summarize.log"; }
    python scripts/tau_curve.py results_nq/N*.json --csv results_nq/tables > logs/nq_tau_curve.log \
      || { bad=1; log "NQ tau_curve failed: read logs/nq_tau_curve.log"; }
    python scripts/check_tables.py results_nq/tables results_nq/N*.json | tail -2 \
      || { bad=1; log "NQ check_tables: the tables do not match the results (see above)"; }
  fi
  if ls results_seeds/P11_seed*.json > /dev/null 2>&1 && [ -f results/P11.json ]; then
    python scripts/seed_summary.py results/P11.json results_seeds/P11_seed*.json --md seeds_summary.md | tail -1 \
      || { bad=1; log "seed_summary failed (see above)"; }
  fi
  if ls sweeps_nq/sweep_N*.json > /dev/null 2>&1; then
    python scripts/freeze_check.py $(for p in $NQ_PAIRS; do [ "$p" = N0 ] || echo "results_nq_static/$p.json"; done) \
      sweeps_nq/sweep_N*.json --kprimes 10,25,50,100,200,500,1000 --out FREEZE_CHECK_NQ.md | tail -1 \
      || log "NQ freeze check did not certify a single K*: read FREEZE_CHECK_NQ.md"
  fi
  log "checks done$([ $bad = 0 ] || echo ' (with failures: see above)')"
  return $bad
}

collect() {
  local out="extension_outputs_$(date +%Y%m%d_%H%M).zip"   # with the time: a second collect never overwrites
  local want=(planner_nq planner_check_nq.md results_nq results_nq_static sweeps_nq coverage_nq adapters_nq
              checks_nq results_seeds results_zipf curve_check_zipf.md query_sweep throughput.json
              throughput_qwen8b.json seeds_summary.md results_llm coverage_llm FREEZE_CHECK_NQ.md logs) have=() missing=()
  local f
  for f in "${want[@]}"; do if [ -e "$f" ]; then have+=("$f"); else missing+=("$f"); fi; done
  for f in pairs/N*/meta.json pairs/P5r/meta.json adapters/adapters_P5r.json checks/Qwen3-Embedding-0.6B.json; do
    [ -e "$f" ] && have+=("$f"); done
  [ ${#missing[@]} -eq 0 ] || log "not produced: ${missing[*]}"
  [ ${#have[@]} -gt 0 ] || { log "nothing to collect"; return 1; }
  rm -f "$out"; zip -qr "$out" "${have[@]}" || { log "collect: zip failed"; return 1; }
  log "wrote $out ($(du -h "$out" | cut -f1)) -- send this file back"
}

case "$STAGE" in
  all)
    preflight
    encode > logs/nq_encode.log 2>&1 || { log "encode failed: logs/nq_encode.log (GPU stage; fix it and rerun 'all')"; exit 1; }
    encode_llm > logs/encode_llm.log 2>&1 || log "encode_llm failed (P5r skipped): logs/encode_llm.log"
    if [ ! -f throughput.json ] && python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      throughput > logs/throughput.log 2>&1 || log "throughput failed (optional): logs/throughput.log"
    fi
    bm25 > logs/nq_bm25.log 2>&1
    pairs > logs/nq_pairs.log 2>&1
    measure > logs/nq_measure.log 2>&1
    plan > logs/nq_plan.log 2>&1
    PARALLEL=1; jobs_=()
    ( replay > logs/nq_replay.log 2>&1 ) & jobs_+=($!)
    ( seeds > logs/seeds.log 2>&1 ) & jobs_+=($!)
    ( zipf > logs/zipf.log 2>&1 ) & jobs_+=($!)
    ( qsweep > logs/qsweep.log 2>&1 ) & jobs_+=($!)
    if llm_ready; then ( p5r > logs/p5r.log 2>&1 ) & jobs_+=($!); fi
    waitall "${jobs_[@]}" || true
    if [ ${#FAILED[@]} -gt 0 ]; then log "${#FAILED[@]} stage(s) failed: read logs/nq_replay.log, logs/seeds.log, logs/zipf.log, logs/qsweep.log, logs/p5r.log"; fi
    check > logs/nq_check.log 2>&1 || true; collect
    [ ${#FAILED[@]} -eq 0 ] ;;
  core)
    preflight
    encode > logs/nq_encode.log 2>&1 || { log "encode failed: logs/nq_encode.log"; exit 1; }
    bm25 > logs/nq_bm25.log 2>&1
    pairs > logs/nq_pairs.log 2>&1
    measure > logs/nq_measure.log 2>&1
    plan > logs/nq_plan.log 2>&1
    replay > logs/nq_replay.log 2>&1 || FAILED+=(replay)
    check > logs/nq_check.log 2>&1 || true; collect
    [ ${#FAILED[@]} -eq 0 ] ;;
  preflight|encode|encode_llm|throughput|bm25|pairs|measure|plan|replay|seeds|zipf|qsweep|p5r|check|collect) "$STAGE" ;;
  *) echo "usage: bash scripts/extension_runs.sh {all|core|preflight|encode|encode_llm|throughput|bm25|pairs|measure|plan|replay|seeds|zipf|qsweep|p5r|check|collect}"; exit 2 ;;
esac
