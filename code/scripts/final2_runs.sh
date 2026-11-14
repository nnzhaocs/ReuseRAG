#!/usr/bin/env bash
# The second round of final experiments (2026-11-11). RUN_FINAL2.md is the guide. CPU only: no
# new encoding; every vector used here exists on the instance.
#
#   bash scripts/final2_runs.sh <stage>
#
#   preflight  clean tree, tests, inputs, cores, disk
#   union      static prices of every window family (S2-S5) from the serving side, uniform and
#              Zipf(0.3/0.6/0.9/1.2), scored against every replay we have (union_price*/)
#   prereg     predictions for the new Zipf replays, committed before any of them runs
#   zipf       Zipf replays: alpha 0.3, 0.6, 1.2 on the ten text pairs; 0.9 on the six not yet
#              replayed (36 replays, $ZJOBS at a time); then scored (ZIPF2_SCORE.md)
#   p5r        the LLM->LLM pair P5r replayed like the text pairs (results_p5r/)
#   subset     nested 2M and 4M corpora cut from the 8.8M encodings (no re-encoding)
#   diag       why the gap moves with corpus size: per-query paired bootstrap and TREC-DL
#              condensed nDCG at 1M, 2M, 4M, 8.8M (SCALE_DIAG.json)
#   sweep2     static sweeps of P2f, P3f, P9f at 2M and 4M (sweeps_s2m/, sweeps_s4m/)
#   replay2    replays at 8.8M (P9f, P2f, P3f) and 2M, 4M, each predicted from its sweep and the
#              prediction committed first (PREREG_REPLAY_<size>.md, REPLAY_<size>_SCORE.md)
#   collect    final2_outputs_<time>.zip
#   all        preflight; then two lanes in parallel:
#                lane 1: union, prereg, zipf, p5r
#                lane 2: subset, diag, sweep2, replay2
#              then collect. A stage that fails is reported and the lane goes on where it can.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-}
EXPECTED_TESTS=${EXPECTED_TESTS:-81}
cpu_quota() {  # cores this container may use (nproc reports the host's 224 on AutoDL)
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
LANE1_CORES=${LANE1_CORES:-$(( NCORES * 3 / 5 ))}         # 12 of 20
LANE2_CORES=${LANE2_CORES:-$(( NCORES - LANE1_CORES ))}   # 8 of 20
ZJOBS=${ZJOBS:-3}
TEXT=${TEXT:-"P2 P3 P4 P5 P9 P10 P11 P12 P13 P14"}
Z09_NEW=${Z09_NEW:-"P3 P4 P9 P10 P12 P14"}
NQ=${NQ:-"N0 N2 N3 N4 N5 N9 N10 N11 N12 N13 N14"}
DATA_ROOT=${DATA_ROOT:-$( [ -d /root/autodl-tmp ] && echo /root/autodl-tmp || echo . )}
ENCF=${ENCF:-$DATA_ROOT/enc_full}
SIZES=${SIZES-"2000000 4000000"}  # explicit SIZES= means: skip intermediate sizes
FPAIRS="P2f P3f P9f"
REPLAY_FULL=${REPLAY_FULL:-"P9f P2f P3f"}
declare -A FSRC=( [P2f]=all-MiniLM-L6-v2 [P3f]=bge-small-en-v1.5 [P9f]=all-MiniLM-L6-v2 )
declare -A FTGT=( [P2f]=bge-base-en-v1.5 [P3f]=bge-base-en-v1.5 [P9f]=bge-small-en-v1.5 )
MODELS="all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5"
FULL_N=8841823
KLADDER="10,25,50,100,200,350,500,1000"
FAILED=()
log() { echo "[$(date -Is)] $*"; }
threads() {  # threads per job: <cores of this lane> / <jobs>
  local t=$(( $1 / $2 )); [ "$t" -ge 1 ] || t=1
  export OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t; log "threads: $2 job(s) x $t of $1 cores"; }
init_registry() {
  python -c "import os,sys; sys.path.insert(0,'.'); from reusemap.provenance import REG_COLS as C; r=os.path.join(sys.argv[1],'RUNS.tsv'); os.path.exists(r) or open(r,'w').write(chr(9).join(C)+chr(10))" "$1"
}
size_lab() { python -c "import sys; print(f's{int(sys.argv[1])/1e6:g}m')" "$1"; }
has() { [ -e "$1" ] || [ -L "$1" ]; }

# Commit one file without ever leaving the working tree dirty: a replay starting in the other lane
# checks `git diff HEAD`, which a staged-but-uncommitted file would fail. The commit is built in a
# private index and HEAD moved in one step; the real index is synced after.
commit_file() {
  local f=$1 msg=$2
  exec 8> logs/.git.lock; flock 8
  local blob tree commit tmp; tmp=$(mktemp)
  blob=$(git hash-object -w "$f")
  GIT_INDEX_FILE=$tmp git read-tree HEAD
  GIT_INDEX_FILE=$tmp git update-index --add --cacheinfo "100644,$blob,$f"
  tree=$(GIT_INDEX_FILE=$tmp git write-tree)
  commit=$(git commit-tree "$tree" -p HEAD -m "$msg")
  git update-ref HEAD "$commit"
  git reset -q
  rm -f "$tmp"; flock -u 8
  log "committed $f as $(git rev-parse --short HEAD) at $(git log -1 --format=%cI)"
  echo "$f $(git rev-parse HEAD) $(git log -1 --format=%cI)" >> final2_outputs_commits.txt
}

current_result() {  # $1 result json, $2 pair dir, $3 config
  python - "$1" "$2" "$3" <<'PYCUR'
import json, os, subprocess, sys
sys.path.insert(0, ".")
from reusemap import provenance
r, pdir, cfg = sys.argv[1:4]
pv = json.load(open(r)).get("provenance") or {}
commit = str(pv.get("git", "")).split("-")[0].split("+")[0]
if not commit or subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"]).returncode:
    sys.exit(1)
if subprocess.run(["git", "diff", "--quiet", commit, "HEAD", "--", "reusemap", "scripts/run_pair.py"]).returncode:
    sys.exit(1)                                   # measurement code changed since that run
if pv.get("config_file") != cfg or pv.get("data_fingerprint") != provenance.data_fingerprint(pdir):
    sys.exit(1)
from reusemap import runner
c = dict(runner.DEFAULT_CFG); c.update(json.load(open(cfg)))       # exactly what run_pair hashes
if pv.get("config_hash") != provenance.config_hash({**c, "dev_overrides": {}}):
    sys.exit(1)
PYCUR
}

check_full_inputs() {
  local qn
  for f in subfull/corpus.tsv subfull/queries.tsv subfull/qrels.tsv subfull/query_sets/trecdl.txt; do
    [ -s "$f" ] || { echo "missing required 8.8M input: $f"; return 1; }
  done
  [ "$(wc -l < subfull/corpus.tsv)" -eq "$FULL_N" ] \
    || { echo "subfull/corpus.tsv has $(wc -l < subfull/corpus.tsv) rows; expected $FULL_N"; return 1; }
  cmp -s subfull/queries.tsv sub1m/queries.tsv \
    || { echo "subfull/queries.tsv differs from sub1m/queries.tsv; scale replay must hold queries fixed"; return 1; }
  cmp -s subfull/query_sets/trecdl.txt sub1m/query_sets/trecdl.txt \
    || { echo "subfull TREC-DL query set differs from sub1m"; return 1; }
  qn=$(wc -l < subfull/queries.tsv)
  python - "$ENCF" "$FULL_N" "$qn" <<'PYFULL'
import json, os, sys
import numpy as np
enc, N, Q = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
models = {"all-MiniLM-L6-v2": 384, "bge-small-en-v1.5": 384, "bge-base-en-v1.5": 768}
for m, dim in models.items():
    d = os.path.join(enc, m)
    for fn, rows in (("doc.npy", N), ("q.npy", Q)):
        path = os.path.join(d, fn)
        if not os.path.isfile(path):
            raise SystemExit(f"missing required 8.8M vector: {path}")
        x = np.load(path, mmap_mode="r")
        if x.ndim != 2 or x.shape[0] != rows or x.shape[1] != dim:
            raise SystemExit(f"wrong shape for {path}: {tuple(x.shape)}, expected ({rows}, {dim})")
    chk = os.path.join("checks_full", m + ".json")
    if not os.path.isfile(chk):
        raise SystemExit(f"missing encode check: {chk}")
    c = json.load(open(chk))
    if c.get("hard_failures") != 0:
        raise SystemExit(f"{chk}: hard_failures={c.get('hard_failures')}, expected 0")
    if os.path.realpath(c.get("enc", "")) != os.path.realpath(d):
        raise SystemExit(f"{chk}: checked enc path does not match {d}")
    if os.path.realpath(c.get("data", "")) != os.path.realpath("subfull"):
        raise SystemExit(f"{chk}: checked data path does not match subfull")
for p in ("P2f", "P3f", "P9f"):
    pd = os.path.join("pairs_full", p)
    for fn, rows in (("doc_src.npy", N), ("doc_tgt.npy", N), ("q_src.npy", Q), ("q_tgt.npy", Q)):
        path = os.path.join(pd, fn)
        if not os.path.isfile(path):
            raise SystemExit(f"missing full pair input: {path}")
        x = np.load(path, mmap_mode="r")
        if x.ndim != 2 or x.shape[0] != rows:
            raise SystemExit(f"wrong first dimension for {path}: {tuple(x.shape)}, expected {rows} rows")
    sw = os.path.join("sweeps_full", f"sweep_{p}.json")
    if not os.path.isfile(sw):
        raise SystemExit(f"missing full sweep: {sw}")
    s = json.load(open(sw))
    if s.get("pair") != p or int(s.get("N", -1)) != N:
        raise SystemExit(f"{sw}: pair/N mismatch ({s.get('pair')}, {s.get('N')})")
    if not any(int(r.get("kprime", -1)) == 50 for r in s.get("depths", [])):
        raise SystemExit(f"{sw}: no K'=50 row")
for cfg in ("configs/msmarco_full_v3_replay.json", "configs/msmarco_s2m_v3_replay.json", "configs/msmarco_s4m_v3_replay.json"):
    c = json.load(open(cfg))
    if int(c.get("window", -1)) != 50:
        raise SystemExit(f"{cfg}: window={c.get('window')}, expected fixed K'=50")
print(f"full-scale inputs ok: N={N:,}, Q={Q:,}, 3 encodings, 3 checked pairs, 3 sweeps, replay K'=50")
PYFULL
}

preflight() {
  mkdir -p logs
  log "cores: $NCORES (lane 1 $LANE1_CORES, lane 2 $LANE2_CORES); nproc says $(nproc)"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { git status --short --untracked-files=no; echo "uncommitted code: install with scripts/install_changes.sh first"; exit 1; }
  local n; n=$(python tests/test_invariants.py 2>&1 | tail -1)
  echo "$n" | grep -q "$EXPECTED_TESTS tests passed" || { echo "tests: $n (expected $EXPECTED_TESTS)"; exit 1; }
  python tests/test_final2.py | tail -1
  python tests/test_runbook.py | tail -1
  local miss=0 p
  for p in $TEXT; do has "pairs/$p/doc_src.npy" || { echo "missing pairs/$p"; miss=1; }; has "results/$p.json" || { echo "missing results/$p.json"; miss=1; }; done
  for d in results_d3 results_zipf zipf_scan sub1m; do has "$d" || { echo "missing $d/"; miss=1; }; done
  [ "$miss" = 0 ] || { echo "inputs missing (above)"; exit 1; }
  # The full-corpus replay is a required Final2 result, not an optional stage.  Refuse to start
  # unless the exact 8.84M corpus, checked vectors, all three full pairs and all three static
  # sweeps are present and internally consistent.  Intermediate 2M/4M sizes remain optional.
  check_full_inputs || { echo "8.8M preflight FAILED; do not start Final2"; exit 1; }
  for d in results_xarch results_nq results_llm pairs/P5r; do has "$d" && log "found $d" || log "absent: $d (optional stage is skipped)"; done
  log "disk: $(df -BG --output=avail "$DATA_ROOT" | tail -1 | tr -dc 0-9) GB free on $DATA_ROOT; 2M needs ~13 GB, 4M ~26 GB"
  log "RAM: $(free -g | awk '/Mem:/{print $2" GB, "$7" GB available"}')"
  log "preflight ok"
}

union() {
  mkdir -p union_price union_price_d3 logs
  threads "$LANE1_CORES" 3
  local p pids=()
  for p in $TEXT; do
    [ -f "union_price/$p.json" ] && { log "union: $p present"; continue; }
    python scripts/union_price.py "pairs/$p" --config configs/msmarco_1m_v3.json --kprime 25,50,100 \
      --workloads uniform,zipf:0.3,zipf:0.6,zipf:0.9,zipf:1.2 --results uniform=results --results zipf:0.9=results_zipf \
      --out-dir union_price > "logs/union_$p.log" 2>&1 & pids+=($!)
    if [ ${#pids[@]} -ge 3 ]; then wait "${pids[0]}" || FAILED+=("union:${pids[0]}"); pids=("${pids[@]:1}"); fi
  done
  local x; for x in "${pids[@]}"; do wait "$x" || FAILED+=("union:$x"); done
  # the K'=25 replays (the plan's depth), the encoder->LLM replays and NQ, uniform only
  for p in P2 P3 P4 P5 P9 P10 P11 P12 P13; do
    [ -f "union_price_d3/$p.json" ] || python scripts/union_price.py "pairs/$p" --config configs/msmarco_1m_v3_d3.json \
      --kprime 25,50 --results uniform=results_d3 --out-dir union_price_d3 > "logs/union_d3_$p.log" 2>&1 || FAILED+=("union_d3:$p")
  done
  if has results_xarch; then mkdir -p union_price_xarch
    for p in P5p P5q; do has "pairs/$p/doc_src.npy" || continue
      [ -f "union_price_xarch/$p.json" ] || python scripts/union_price.py "pairs/$p" --config configs/msmarco_1m_v3.json \
        --kprime 25,50,100 --results uniform=results_xarch --out-dir union_price_xarch > "logs/union_xarch_$p.log" 2>&1 || FAILED+=("union_xarch:$p")
    done
  fi
  if has results_nq; then mkdir -p union_price_nq
    for p in $NQ; do has "pairs/$p/doc_src.npy" || continue
      [ -f "union_price_nq/$p.json" ] || python scripts/union_price.py "pairs/$p" --config configs/nq_100k_v3.json \
        --kprime 25,50,100,200,500,1000 --results uniform=results_nq --out-dir union_price_nq > "logs/union_nq_$p.log" 2>&1 || FAILED+=("union_nq:$p")
    done
  fi
  log "union done: $(ls union_price/P*.json 2>/dev/null | wc -l) text pairs"
  grep -h "replay at\|agree" logs/union_*.log | head -40 || true
}

prereg() {
  [ -f PREREG_ZIPF2.md ] && git ls-files --error-unmatch PREREG_ZIPF2.md > /dev/null 2>&1 && { log "prereg: PREREG_ZIPF2.md already committed"; return 0; }
  ls results_zipf_a* > /dev/null 2>&1 && { log "prereg: Zipf replays already exist; a prediction written now would not be one -- stop"; return 1; }
  python scripts/zipf2.py prereg --scan zipf_scan --price union_price \
    --plan "0.3,0.6,1.2=$(echo $TEXT | tr ' ' ',')" --plan "0.9=$(echo $Z09_NEW | tr ' ' ',')" \
    --window 50 --md PREREG_ZIPF2.md
  commit_file PREREG_ZIPF2.md "Pre-register the Zipf replays at alpha 0.3, 0.6, 1.2 (and 0.9 on six pairs) before running them"
}

zipf() {
  git ls-files --error-unmatch PREREG_ZIPF2.md > /dev/null 2>&1 || { log "zipf: PREREG_ZIPF2.md is not committed; run prereg first"; return 1; }
  threads "$LANE1_CORES" "$ZJOBS"
  local jobs=() a p cfg out
  for a in 0.3 0.6 1.2; do for p in $TEXT; do jobs+=("$a:$p"); done; done
  for p in $Z09_NEW; do jobs+=("0.9:$p"); done
  local pids=() j
  for j in "${jobs[@]}"; do
    a=${j%%:*}; p=${j#*:}; out="results_zipf_a$a"; mkdir -p "$out"; init_registry "$out"
    [ -f "$out/$p.json" ] && continue
    if [ "$a" = 0.9 ]; then cfg=configs/msmarco_1m_v3_zipf.json; else cfg="configs/msmarco_1m_v3_zipf${a//./}.json"; fi
    python scripts/run_pair.py "pairs/$p" --config "$cfg" --name "$p" --out "$out/$p.json" > "logs/zipf2_${a}_$p.log" 2>&1 & pids+=($!)
    if [ ${#pids[@]} -ge "$ZJOBS" ]; then wait "${pids[0]}" || FAILED+=("zipf:${pids[0]}"); pids=("${pids[@]:1}"); fi
  done
  local x; for x in "${pids[@]}"; do wait "$x" || FAILED+=("zipf:$x"); done
  # the six new 0.9 replays sit beside the four pre-registered on 2026-11-09 in results_zipf/
  python scripts/zipf2.py score --prereg PREREG_ZIPF2.md --results-pattern "results_zipf_a{A}" --md ZIPF2_SCORE.md \
    || FAILED+=("zipf-score")
  python scripts/coverage_curve_check.py --bound-only --results "results_zipf_a0.3" --md curve_check_zipf03.md 2>/dev/null | tail -1 || true
  log "zipf done"
}

p5r() {
  has pairs/P5r/doc_src.npy || { log "p5r: no pairs/P5r; skipped"; return 0; }
  mkdir -p results_p5r; init_registry results_p5r
  [ -f results_p5r/P5r.json ] && { log "p5r: present"; return 0; }
  threads "$LANE1_CORES" 1
  python scripts/run_pair.py pairs/P5r --config configs/msmarco_1m_v3.json --name P5r --out results_p5r/P5r.json \
    > logs/p5r_replay.log 2>&1 || { log "p5r: failed, see logs/p5r_replay.log"; return 1; }
  tail -5 logs/p5r_replay.log
}

subset() {
  [ -n "${SIZES//[[:space:]]/}" ] || { log "subset: SIZES is empty; intermediate 2M/4M corpora intentionally skipped"; return 0; }
  has subfull/corpus.tsv || { log "subset: no subfull/; skipped"; return 1; }
  threads "$LANE2_CORES" 1
  python scripts/scale_subset.py --full subfull --small sub1m --enc "$ENCF" --models $MODELS \
    --sizes "$(echo $SIZES | tr ' ' ',')" --out-root "$DATA_ROOT" --tag s 2>&1 | tee logs/subset.log | tail -12
  local S lab p
  for S in $SIZES; do
    lab=$(size_lab "$S")
    has "$lab" || ln -s "$DATA_ROOT/$lab" "$lab"
    for p in $FPAIRS; do
      has "$DATA_ROOT/enc_$lab/${FSRC[$p]}/doc.npy" && has "$DATA_ROOT/enc_$lab/${FTGT[$p]}/doc.npy" || { log "subset: $lab vectors for $p missing (disk?)"; continue; }
      [ -f "pairs_$lab/$p/meta.json" ] || python scripts/assemble_pair.py --src "$DATA_ROOT/enc_$lab/${FSRC[$p]}" \
        --tgt "$DATA_ROOT/enc_$lab/${FTGT[$p]}" --data "$lab" --out "pairs_$lab/$p" \
        --meta "{\"source\":\"${FSRC[$p]}\",\"target\":\"${FTGT[$p]}\",\"corpus\":\"msmarco nested $lab\"}"
    done
  done
}

diag() {
  threads "$LANE2_CORES" 1
  local args=(--size "1M=pairs/{P}:configs/msmarco_1m_v3.json") S lab
  for S in $SIZES; do lab=$(size_lab "$S"); has "pairs_$lab/P2f/doc_src.npy" && args+=(--size "${lab}=pairs_$lab/{P}f:configs/msmarco_${lab}_v3.json"); done
  has pairs_full/P2f/doc_src.npy && args+=(--size "8.8M=pairs_full/{P}f:configs/msmarco_full_v3.json")
  python scripts/scale_diag.py --pairs P2 P3 P9 "${args[@]}" --out SCALE_DIAG.json 2>&1 | tee logs/scale_diag.log | tail -20
}

sweep2() {
  threads "$LANE2_CORES" 1
  local S lab p
  for S in $SIZES; do
    lab=$(size_lab "$S"); mkdir -p "sweeps_$lab"
    for p in $FPAIRS; do
      has "pairs_$lab/$p/doc_src.npy" || continue
      [ -f "sweeps_$lab/sweep_$p.json" ] && continue
      python scripts/sweep.py "pairs_$lab/$p" --config "configs/msmarco_${lab}_v3.json" --kprime "$KLADDER" \
        --out-dir "sweeps_$lab" > "logs/sweep_${lab}_$p.log" 2>&1 || { FAILED+=("sweep:$lab:$p"); continue; }
      tail -3 "logs/sweep_${lab}_$p.log"
    done
  done
}

replay_size() {  # $1 label (full|s2m|s4m), $2 pair dir prefix, $3 sweeps dir, $4 config, $5 pairs
  local lab=$1 pdir=$2 sw=$3 cfg=$4 prs=$5 p sweeps=() miss=0
  # A requested scale replay is all-or-nothing: never silently turn a 3-pair experiment into 1/3 or 2/3.
  for p in $prs; do
    [ -f "$sw/sweep_$p.json" ] || { log "replay $lab: missing $sw/sweep_$p.json"; miss=1; continue; }
    has "$pdir/$p/doc_src.npy" || { log "replay $lab: missing $pdir/$p"; miss=1; continue; }
    sweeps+=("$sw/sweep_$p.json")
  done
  [ "$miss" = 0 ] && [ ${#sweeps[@]} -eq $(wc -w <<< "$prs") ] \
    || { log "replay $lab: required inputs are incomplete; refusing a partial replay"; return 1; }
  python - "$cfg" <<'PYCFG'
import json, sys
c=json.load(open(sys.argv[1]))
if int(c.get("window", -1)) != 50:
    raise SystemExit(f"{sys.argv[1]}: window={c.get('window')}, expected fixed K'=50")
PYCFG
  if ! git ls-files --error-unmatch "PREREG_REPLAY_$lab.md" > /dev/null 2>&1; then
    ls "results_$lab"/P*.json > /dev/null 2>&1 && { log "replay $lab: results exist without a committed prediction -- stop (archive/remove those old results first)"; return 1; }
    # Intentionally fixed at K'=50 for the scale replay: compare the same depth across corpus sizes.
    python scripts/replay_prereg.py prereg --sweeps "${sweeps[@]}" --window 50 --label "$lab" --md "PREREG_REPLAY_$lab.md"
    commit_file "PREREG_REPLAY_$lab.md" "Pre-register the replays at $lab from their static sweeps, before running them"
  fi
  mkdir -p "results_$lab"; init_registry "results_$lab"
  threads "$LANE2_CORES" 1
  for p in $prs; do
    # Keep an existing result only if it is current: same pair data, same config file content, and
    # no change to the measurement code since the commit that produced it. The run key alone would
    # force a rerun after every unrelated commit (the other lane commits its predictions while this
    # one runs), and an 8.8M replay costs hours.
    if [ -f "results_$lab/$p.json" ] && current_result "results_$lab/$p.json" "$pdir/$p" "$cfg"; then
      log "replay $lab: $p is current (measurement code, config and data unchanged); kept"; continue
    fi
    log "replay $lab: $p"
    python scripts/run_pair.py "$pdir/$p" --config "$cfg" --name "$p" --out "results_$lab/$p.json" \
      > "logs/replay_${lab}_$p.log" 2>&1 || { FAILED+=("replay:$lab:$p"); log "replay $lab $p failed"; }
  done
  for p in $prs; do [ -f "results_$lab/$p.json" ] || { log "replay $lab: missing result for $p"; miss=1; }; done
  [ "$miss" = 0 ] || return 1
  python scripts/replay_prereg.py score --prereg "PREREG_REPLAY_$lab.md" --results "results_$lab" --md "REPLAY_${lab}_SCORE.md" \
    || { FAILED+=("replay-score:$lab"); log "replay $lab: scoring failed"; return 1; }
}

replay2() {
  replay_size full pairs_full sweeps_full configs/msmarco_full_v3_replay.json "$REPLAY_FULL" || FAILED+=("replay:full")
  local S lab
  for S in $SIZES; do lab=$(size_lab "$S")
    replay_size "$lab" "pairs_$lab" "sweeps_$lab" "configs/msmarco_${lab}_v3_replay.json" "$FPAIRS" || FAILED+=("replay:$lab")
  done
}

collect() {
  local z; z="final2_outputs_$(date +%Y%m%d_%H%M).zip"
  local items=() f
  for f in union_price union_price_d3 union_price_xarch union_price_nq PREREG_ZIPF2.md ZIPF2_SCORE.md \
           results_zipf_a0.3 results_zipf_a0.6 results_zipf_a0.9 results_zipf_a1.2 curve_check_zipf03.md \
           results_p5r SCALE_DIAG.json sweeps_s2m sweeps_s4m results_full results_s2m results_s4m \
           PREREG_REPLAY_full.md PREREG_REPLAY_s2m.md PREREG_REPLAY_s4m.md REPLAY_full_SCORE.md REPLAY_s2m_SCORE.md \
           REPLAY_s4m_SCORE.md final2_outputs_commits.txt logs; do has "$f" && items+=("$f"); done
  for f in s2m/rows.npy s4m/rows.npy s2m/qrels.tsv s4m/qrels.tsv; do has "$f" && items+=("$f"); done
  [ ${#items[@]} -gt 0 ] || { log "collect: nothing to collect"; return 1; }
  git log --format='%H %cI %s' -15 > final2_git_log.txt; items+=(final2_git_log.txt)
  zip -qr "$z" "${items[@]}" -x 'pairs*' -x '*/doc*.npy'
  log "wrote $z ($(du -h "$z" | cut -f1)) -- send this file"
}

lane1() {
  local st before st_rc rc=0
  for st in union prereg zipf p5r; do
    log "===== lane 1: $st"; before=${#FAILED[@]}
    if "$st"; then st_rc=0; else st_rc=$?; fi
    if [ "$st_rc" -ne 0 ]; then
      [ ${#FAILED[@]} -gt "$before" ] || FAILED+=("$st")
      log "$st failed (exit $st_rc)"; rc=1
      [ "$st" = prereg ] && break
    elif [ ${#FAILED[@]} -gt "$before" ]; then
      log "$st recorded one or more failed sub-jobs"; rc=1
    fi
  done
  if [ "$rc" -ne 0 ]; then log "lane 1 finished with failures: ${FAILED[*]}"; return 1; fi
}

lane2() {
  local st before st_rc rc=0
  for st in subset diag sweep2 replay2; do
    log "===== lane 2: $st"; before=${#FAILED[@]}
    if "$st"; then st_rc=0; else st_rc=$?; fi
    if [ "$st_rc" -ne 0 ]; then
      [ ${#FAILED[@]} -gt "$before" ] || FAILED+=("$st")
      log "$st failed (exit $st_rc)"; rc=1
    elif [ ${#FAILED[@]} -gt "$before" ]; then
      log "$st recorded one or more failed sub-jobs"; rc=1
    fi
  done
  if [ "$rc" -ne 0 ]; then log "lane 2 finished with failures: ${FAILED[*]}"; return 1; fi
}

all() {
  local p1 p2 rc=0
  preflight
  lane1 > logs/final2_lane1.log 2>&1 & p1=$!
  lane2 > logs/final2_lane2.log 2>&1 & p2=$!
  log "lane 1 (pid $p1): logs/final2_lane1.log; lane 2 (pid $p2): logs/final2_lane2.log"
  if wait "$p1"; then log "lane 1 completed successfully"; else log "lane 1 FAILED -- see logs/final2_lane1.log"; rc=1; fi
  if wait "$p2"; then log "lane 2 completed successfully"; else log "lane 2 FAILED -- see logs/final2_lane2.log"; rc=1; fi
  grep -h "failed\|FAILED\|stop" logs/final2_lane1.log logs/final2_lane2.log | grep -v "^$" | tail -20 || true
  if ! collect; then log "collect FAILED"; rc=1; fi
  if [ "$rc" -ne 0 ]; then
    log "ALL FINISHED WITH FAILURES -- partial outputs were collected; inspect both lane logs"
    return 1
  fi
  log "ALL DONE -- both lanes completed successfully"
}

case "$STAGE" in
  preflight|union|prereg|zipf|p5r|subset|diag|sweep2|replay2|collect|all|lane1|lane2) "$STAGE" ;;
  *) echo "usage: bash scripts/final2_runs.sh preflight|union|prereg|zipf|p5r|subset|diag|sweep2|replay2|collect|all"; exit 2 ;;
esac
