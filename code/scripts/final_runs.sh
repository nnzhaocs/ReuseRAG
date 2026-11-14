#!/usr/bin/env bash
# Last two experiments for the ICDE 2027 submission (2026-11-10). RUN_FINAL.md is the guide.
#
#   bash scripts/final_runs.sh <stage>
#
#   preflight  branch, clean tree, tests, inputs, disk, cores
#   prereg     record the commit of PREREG_SCALE.md before any full-corpus vector exists
#   seeds      CPU: the eight windowed text pairs other than P11 at seeds 1-4 (32 replays,
#              $SEED_JOBS at a time); P11's seeds exist from the extension run
#   build      the full MS MARCO collection (8.8M) with the same query files as sub1m/
#   encode     GPU: MiniLM-L6, bge-small, bge-base documents at 8.8M (chunked, memory-mapped);
#              their query vectors are the 1M ones (the queries are identical) + encode_check
#   pairs      P2f, P3f, P9f: P2, P3, P9 on the full corpus (no BM25: static measurements only)
#   sweep      rho, G, coverage and the gap on K' = 10..1000 (sweeps_full/)
#   mass       window-multiplicity histograms at 8.8M for both sources (window_mass_full/)
#   summary    seeds_all.md; SCALE_CHECK.md if the 1M sweeps are present
#   collect    final_outputs_<time>.zip
#   all        preflight, prereg; seeds in the background; build, encode, pairs, sweep, mass;
#              wait for the seeds; summary, collect. SKIP_SCALE=1 runs only the seeds.
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
SEED_PAIRS=${SEED_PAIRS:-"P2 P3 P4 P5 P9 P10 P12 P13"}
SEEDS=${SEEDS:-"1 2 3 4"}
SEED_JOBS=${SEED_JOBS:-4}
SUB1M=${SUB1M:-sub1m}
# the big files go on the data disk: on AutoDL /root/sub13run is on the 30 GB system disk, and
# enc/ there is already a link to /root/autodl-tmp/enc
DATA_ROOT=${DATA_ROOT:-$( [ -d /root/autodl-tmp ] && echo /root/autodl-tmp || echo . )}
FULL=subfull                                # configs/msmarco_full_v3.json names subfull/query_sets/
ENCF=${ENCF:-$DATA_ROOT/enc_full}
full_dir() {  # subfull/ is a link to $DATA_ROOT/subfull when the data disk is elsewhere
  if [ -e "$FULL" ] || [ -L "$FULL" ]; then return; fi
  if [ "$DATA_ROOT" != . ]; then mkdir -p "$DATA_ROOT/subfull"; ln -s "$DATA_ROOT/subfull" "$FULL"; fi
}
CHUNK=${CHUNK:-500000}
ONE_M_SWEEPS=${ONE_M_SWEEPS:-"sweeps sweeps_fine"}
KLADDER="10,25,50,100,200,350,500,1000"
declare -A HF=( [all-MiniLM-L6-v2]=sentence-transformers/all-MiniLM-L6-v2 [bge-small-en-v1.5]=BAAI/bge-small-en-v1.5
                [bge-base-en-v1.5]=BAAI/bge-base-en-v1.5 )
DIMS="all-MiniLM-L6-v2:384 bge-small-en-v1.5:384 bge-base-en-v1.5:768"
# the 1M pair whose link gives each model's 1M query vectors (pairs/<P>/q_src.npy or q_tgt.npy)
declare -A QFROM=( [all-MiniLM-L6-v2]=P2/q_src.npy [bge-base-en-v1.5]=P2/q_tgt.npy [bge-small-en-v1.5]=P9/q_tgt.npy )
declare -A FSRC=( [P2f]=all-MiniLM-L6-v2 [P3f]=bge-small-en-v1.5 [P9f]=all-MiniLM-L6-v2 )
declare -A FTGT=( [P2f]=bge-base-en-v1.5 [P3f]=bge-base-en-v1.5 [P9f]=bge-small-en-v1.5 )
FPAIRS="P2f P3f P9f"
FAILED=()
log() { echo "[$(date -Is)] $*"; }
threads() { local t=$(( NCORES / $1 )); [ "$t" -ge 1 ] || t=1
  export OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t; log "threads: $1 job(s) x $t of $NCORES cores"; }
init_registry() {
  python -c "import os,sys; sys.path.insert(0,'.'); from reusemap.provenance import REG_COLS as C; r=os.path.join(sys.argv[1],'RUNS.tsv'); os.path.exists(r) or open(r,'w').write(chr(9).join(C)+chr(10))" "$1"
}
raw_dir() {  # where the MS MARCO files sub1m/ was built from
  if [ -n "${RAW:-}" ]; then echo "$RAW"; return; fi
  local f; f=$(find /root -maxdepth 4 -name collection.tsv -size +1G 2>/dev/null | head -1)
  [ -n "$f" ] && dirname "$f"
}
RAWFILES="collection.tsv qrels.dev.tsv 2019qrels-pass.txt 2020qrels-pass.txt queries.dev.tsv msmarco-test2019-queries.tsv msmarco-test2020-queries.tsv"
need_gb() {  # disk the scale study needs on $ENCF's filesystem: three document matrices + corpus.tsv
  local r n; r=$(raw_dir); n=$(wc -l < "$r/collection.tsv")
  python -c "import sys; n=int(sys.argv[1]); print(int(n*(384+384+768)*4/1e9 + 2*$(stat -c %s "$r/collection.tsv")/1e9 + 3))" "$n"
}
free_gb() { mkdir -p "$ENCF"; df -BG --output=avail "$ENCF" | tail -1 | tr -dc 0-9; }

preflight() {
  [ "$(git branch --show-current)" = submission-eab ] || { echo "not on branch submission-eab"; exit 1; }
  [ -z "$(git status --porcelain -- reusemap scripts configs tests compat)" ] || { echo "uncommitted code: results would be ok_dirty"; exit 1; }
  [ -z "$(git diff HEAD --name-only)" ] || { echo "tracked files differ from HEAD: $(git diff HEAD --name-only | tr '\n' ' ')"; exit 1; }
  local t; t=$(python tests/test_invariants.py | tail -1)
  [ "$t" = "$EXPECTED_TESTS tests passed" ] || { echo "tests: got '$t', expected '$EXPECTED_TESTS tests passed'"; exit 1; }
  log "cores: $NCORES (nproc says $(nproc)); set NCORES=<n> to override"
  local p; for p in $SEED_PAIRS P11; do [ -f "pairs/$p/meta.json" ] || { echo "missing pairs/$p"; exit 1; }; done
  [ -f configs/msmarco_1m_v3.json ] && [ -f configs/msmarco_full_v3.json ] || { echo "missing configs"; exit 1; }
  log "seeds: inputs present"
  if [ "${SKIP_SCALE:-0}" = 1 ]; then log "preflight ok (SKIP_SCALE=1: seeds only)"; return; fi
  local r; r=$(raw_dir)
  [ -n "$r" ] || { echo "MS MARCO files not found under /root: set RAW=<dir with $RAWFILES>"; exit 1; }
  for f in $RAWFILES; do [ -s "$r/$f" ] || { echo "missing $r/$f (RAW=$r)"; exit 1; }; done
  for f in queries.tsv qrels.tsv query_sets/trecdl.txt; do [ -s "$SUB1M/$f" ] || { echo "missing $SUB1M/$f"; exit 1; }; done
  local M; for M in "${!QFROM[@]}"; do [ -f "pairs/${QFROM[$M]}" ] || { echo "missing pairs/${QFROM[$M]}"; exit 1; }; done
  local need free; need=$(need_gb); free=$(free_gb)
  if [ "$free" -lt "$need" ] && ! ls "$ENCF"/*/doc.npy > /dev/null 2>&1; then
    echo "disk: $ENCF has ${free} GB free, the 8.8M encodings need ~${need} GB."
    echo "Expand the data disk (or set ENCF=<dir on a disk with room>), or run SKIP_SCALE=1 for the seeds only."
    exit 1
  fi
  log "disk: ${free} GB free in $ENCF, ~${need} GB needed; MS MARCO files in $r"
  if [ "${SKIP_GPU_CHECK:-0}" != 1 ]; then
    python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))" \
      || { echo "no GPU visible"; exit 1; }
  fi
  log "preflight ok"
}

prereg() {
  mkdir -p scale_outputs
  local c; c=$(git log -1 --format='%H %cI' -- PREREG_SCALE.md)
  [ -n "$c" ] || { echo "PREREG_SCALE.md is not committed"; exit 1; }
  if ls "$ENCF"/*/doc.npy > /dev/null 2>&1 && [ ! -f scale_outputs/PREREG_COMMIT.txt ]; then
    log "prereg: full-corpus vectors already exist and no earlier record -- recording anyway, marked late"
    echo "LATE: recorded after $ENCF existed" >> scale_outputs/PREREG_COMMIT.txt
  fi
  [ -f scale_outputs/PREREG_COMMIT.txt ] && grep -q "$c" scale_outputs/PREREG_COMMIT.txt \
    || echo "PREREG_SCALE.md committed $c; recorded $(date -Is)" >> scale_outputs/PREREG_COMMIT.txt
  cat scale_outputs/PREREG_COMMIT.txt
}

seeds() {  # 32 replays; each is the paper's replay of that pair with only the seed changed
  mkdir -p results_seeds logs
  # one seeds stage at a time: `seeds` started on its own and `all` started later must not run
  # the same replay twice into the same file
  exec 9> logs/.seeds.lock
  if ! flock -n 9; then
    log "seeds: another seeds stage is running; waiting for it to finish"; flock 9
  fi
  log "seeds: $SEED_PAIRS at seeds $SEEDS, $SEED_JOBS at a time (per-run logs: logs/seed_<P>_<s>.log)"
  init_registry results_seeds
  # leave 4 cores to the GPU encoder's tokenizer when the scale study runs alongside
  local t=$(( (NCORES > 8 ? NCORES - 4 : NCORES) / SEED_JOBS )); [ "$t" -ge 1 ] || t=1
  export OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t
  log "threads: $SEED_JOBS replay(s) x $t of $NCORES cores"
  local p s running=0 pids=() n=0
  for p in $SEED_PAIRS; do for s in $SEEDS; do
    if [ -f "results_seeds/${p}_seed$s.json" ]; then continue; fi
    python scripts/run_pair.py "pairs/$p" --config configs/msmarco_1m_v3.json --name "${p}_seed$s" --set "seed=$s" \
      --out "results_seeds/${p}_seed$s.json" > "logs/seed_${p}_$s.log" 2>&1 & pids+=($!)
    n=$((n + 1))
    if [ ${#pids[@]} -ge "$SEED_JOBS" ]; then wait "${pids[0]}" || FAILED+=("seed:${pids[0]}"); pids=("${pids[@]:1}"); fi
  done; done
  local pid; for pid in "${pids[@]}"; do wait "$pid" || FAILED+=("seed:$pid"); done
  local want=0; for p in $SEED_PAIRS; do for s in $SEEDS; do want=$((want + 1)); done; done
  local have; have=$(for p in $SEED_PAIRS; do for s in $SEEDS; do [ -f "results_seeds/${p}_seed$s.json" ] && echo x; done; done | wc -l)
  log "seeds: $have of $want replays present ($n started now)"
  [ "$have" -eq "$want" ] || { log "seeds: missing replays -- read logs/seed_<P>_<s>.log, then re-run this stage (done ones are skipped)"; return 1; }
  log "seeds done"
}

build() {
  full_dir
  if [ -s "$FULL/corpus.tsv" ] && [ -s "$FULL/qrels.tsv" ]; then log "build: $FULL present"; else
    local r; r=$(raw_dir); local n; n=$(wc -l < "$r/collection.tsv")
    log "build: the whole collection ($n passages) from $r"
    python scripts/build_subsample.py --collection "$r/collection.tsv" \
      --qrels "$r/qrels.dev.tsv:dev" "$r/2019qrels-pass.txt:trecdl" "$r/2020qrels-pass.txt:trecdl" \
      --queries "$r/queries.dev.tsv:dev" "$r/msmarco-test2019-queries.tsv:trecdl" "$r/msmarco-test2020-queries.tsv:trecdl" \
      --N "$n" --out "$FULL" | tee logs/build_full.log
    grep -q "judged coverage: 100.00%" logs/build_full.log || { log "build: judged coverage is not 100% -- stop"; return 1; }
  fi
  # the scale comparison holds the queries fixed: same texts, same order, same TREC-DL set
  cmp -s "$FULL/queries.tsv" "$SUB1M/queries.tsv" || { log "build: $FULL/queries.tsv differs from $SUB1M/queries.tsv -- the query files are not the ones sub1m/ was built from"; return 1; }
  cmp -s "$FULL/query_sets/trecdl.txt" "$SUB1M/query_sets/trecdl.txt" || { log "build: TREC-DL query set differs from $SUB1M"; return 1; }
  log "build: queries identical to $SUB1M ($(wc -l < "$FULL/queries.tsv")); build done"
}

encode() {  # GPU; skips a model whose document vectors exist and passed encode_check
  mkdir -p checks_full logs
  local M
  for M in all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5; do
    local d="$ENCF/$M"; mkdir -p "$d"
    local q; q=$(readlink -f "pairs/${QFROM[$M]}")
    for f in q.npy q.npy.json q.npy.txt; do
      [ -e "$(dirname "$q")/$f" ] && ln -sfn "$(dirname "$q")/$f" "$d/$f"
    done
    if [ -f "$d/doc.npy" ] && [ -f "checks_full/$M.json" ] && python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('hard_failures')==0 else 1)" "checks_full/$M.json"; then
      log "encode: $M present and checked"; continue
    fi
    log "encode: $M documents ($(wc -l < "$FULL/corpus.tsv") passages, chunks of $CHUNK)"
    export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=true   # the tokenizer's pool, not the host's 224
    python scripts/encode.py --model "${HF[$M]}" --kind doc --tsv "$FULL/corpus.tsv" --out "$d/doc.npy" \
      --max-seq-len 256 --chunk "$CHUNK" > "logs/encode_full_$M.log" 2>&1 \
      || { log "encode: $M failed -- see logs/encode_full_$M.log"; return 1; }
    tail -2 "logs/encode_full_$M.log"
    python scripts/encode_check.py --enc "$d" --data "$FULL" --json "checks_full/$M.json" | tail -2 \
      || { log "encode: $M failed encode_check -- see checks_full/$M.json; stop here"; return 1; }
  done
  log "encode done"
}

pairs() {
  local p
  for p in $FPAIRS; do
    [ -f "pairs_full/$p/meta.json" ] && { log "pairs: $p present"; continue; }
    python scripts/assemble_pair.py --src "$ENCF/${FSRC[$p]}" --tgt "$ENCF/${FTGT[$p]}" --data "$FULL" \
      --out "pairs_full/$p" --meta "{\"source\":\"${HF[${FSRC[$p]}]}\",\"target\":\"${HF[${FTGT[$p]}]}\",\"corpus\":\"msmarco_full\"}"
  done
  log "pairs done"
}

sweep() {  # one pair at a time: each holds two 8.8M document matrices in the page cache
  mkdir -p sweeps_full; threads 1
  local p
  for p in $FPAIRS; do
    [ -f "sweeps_full/sweep_$p.json" ] && { log "sweep: $p present"; continue; }
    python scripts/sweep.py "pairs_full/$p" --config configs/msmarco_full_v3.json --kprime "$KLADDER" \
      --out-dir sweeps_full > "logs/sweep_full_$p.log" 2>&1 || { log "sweep: $p failed -- see logs/sweep_full_$p.log"; return 1; }
    tail -12 "logs/sweep_full_$p.log"
  done
  log "sweep done"
}

mass() {
  mkdir -p window_mass_full; threads 1
  local M
  for M in all-MiniLM-L6-v2 bge-small-en-v1.5; do
    [ -f "window_mass_full/$M.json" ] && { log "mass: $M present"; continue; }
    python scripts/window_mass.py --docs "$ENCF/$M/doc.npy" --queries "$ENCF/$M/q.npy" \
      --exclude-qids "$FULL/query_sets/trecdl.txt" --kprime 10,25,50,100,200,500,1000 \
      --out "window_mass_full/$M.json" --name "$M" > "logs/mass_full_$M.log" 2>&1 \
      || { log "mass: $M failed -- see logs/mass_full_$M.log"; return 1; }
  done
  log "mass done"
}

summary() {
  python scripts/seeds_all_summary.py --results results --seeds results_seeds \
    --pairs $SEED_PAIRS P11 --plan planner/plan.csv --md seeds_all.md | tail -1 || log "seeds summary failed (see above)"
  local ones=() d p
  for d in $ONE_M_SWEEPS; do for p in P2 P3 P9; do [ -f "$d/sweep_$p.json" ] && ones+=("$d/sweep_$p.json"); done; done
  if [ ${#ones[@]} -gt 0 ] && ls sweeps_full/sweep_P*f.json > /dev/null 2>&1; then
    python scripts/scale_check.py --one-m "${ones[@]}" --full sweeps_full/sweep_P*f.json \
      --mass window_mass_full --md SCALE_CHECK.md | tail -6 || log "scale check failed (see above)"
  else
    log "summary: no 1M sweeps in '$ONE_M_SWEEPS' (set ONE_M_SWEEPS) or no full sweeps: SCALE_CHECK.md skipped (it is recomputed from the zip)"
  fi
}

collect() {
  local z; z="final_outputs_$(date +%Y%m%d_%H%M).zip"
  local items=() f
  for f in results_seeds seeds_all.md sweeps_full window_mass_full checks_full SCALE_CHECK.md scale_outputs logs \
           "$FULL/query_sets" PREREG_SCALE.md; do [ -e "$f" ] && items+=("$f"); done
  for f in "$ENCF"/*/doc.npy.json "$ENCF"/*/doc.npy.txt; do [ -e "$f" ] && items+=("$f"); done
  [ -f "$FULL/corpus.tsv" ] && { wc -l "$FULL/corpus.tsv" > scale_outputs/corpus_lines.txt; items+=(scale_outputs/corpus_lines.txt); }
  [ ${#items[@]} -gt 0 ] || { log "collect: nothing to collect (no outputs yet)"; return 1; }
  zip -qr "$z" "${items[@]}" -x '*.npy'
  log "wrote $z ($(du -h "$z" | cut -f1)) -- send this file"
}

all() {
  preflight; mkdir -p logs
  [ "${SKIP_SCALE:-0}" = 1 ] || prereg
  seeds > logs/seeds_all.log 2>&1 & local spid=$!
  log "seeds running in the background (pid $spid, logs/seeds_all.log)"
  if [ "${SKIP_SCALE:-0}" != 1 ]; then
    for st in build encode pairs sweep mass; do
      log "===== $st"; "$st" || { FAILED+=("$st"); log "$st failed: the later scale stages need it, stopping the scale part"; break; }
    done
  fi
  log "waiting for the seeds"; wait "$spid" || FAILED+=(seeds)
  tail -3 logs/seeds_all.log
  summary; collect
  [ ${#FAILED[@]} -eq 0 ] && log "ALL DONE" || { log "done with failures: ${FAILED[*]}"; return 1; }
}

case "$STAGE" in
  preflight|prereg|seeds|build|encode|pairs|sweep|mass|summary|collect|all) "$STAGE" ;;
  *) echo "usage: bash scripts/final_runs.sh preflight|prereg|seeds|build|encode|pairs|sweep|mass|summary|collect|all"; exit 2 ;;
esac
