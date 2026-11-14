#!/usr/bin/env bash
# Rerun of every measurement that used Qwen3 QUERY vectors encoded with the instruction twice.
# RUN_QWEN.md is the guide. Run from the repository root, with the final code committed.
#
#   bash scripts/qwen_rerun.sh <stage>
#
#   check     which Qwen3 query encodings are stale (made under the old contract); inputs; tests
#   requery   GPU: re-encode ONLY the Qwen3 query files under the fixed contract (documents are
#             unaffected: their prefix and prompt were empty), each with encode_check; the old
#             query files are kept in superseded_double_instruction/
#   verify    every Qwen3 query encoding in use was made under the current contract, and the
#             model saw the instruction once (from the sidecar's recorded model input)
#   measure   CPU, in parallel: P5p/P5q static sweeps, P5p/P5q adapter families, P5p/P5q full
#             replays (xarch), and the LLM->LLM pair P5r (submission_runs.sh llm); the results
#             they replace are moved to superseded_double_instruction/ first
#   consistent  every redone output read the fixed query files (also run at the end of measure)
#   collect   qwen_rerun_outputs_<time>.zip
#   all       check, requery, verify, measure, collect
#
# Affected and redone here: P5p, P5q (MiniLM / bge-base -> Qwen3-Embedding-8B) and P5r
# (Qwen3-Embedding-0.6B -> 8B). Not affected: every other pair, all document vectors, NQ (N*),
# the latency kit. P5c (NQ, MiniLM -> Qwen3-8B, measure-only) is not in the paper; WITH_NQ=1
# also re-encodes its query file and re-runs it.
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
Q8=${Q8:-enc/Qwen3-Embedding-8B}
Q06=${Q06:-enc/Qwen3-Embedding-0.6B}
QNQ=${QNQ:-enc_nq/Qwen3-Embedding-8B}
WITH_NQ=${WITH_NQ:-0}
OLD=superseded_double_instruction
FAILED=()
log() { echo "[$(date -Is)] $*"; }
waitall() { local pid n0=${#FAILED[@]}; for pid in "$@"; do wait "$pid" || FAILED+=("$pid"); done; [ ${#FAILED[@]} -eq "$n0" ]; }

# the query encodings this script owns: "<enc dir>|<model id>|<data dir>"
targets() {
  echo "$Q8|Qwen/Qwen3-Embedding-8B|sub1m"
  [ -d "$Q06" ] && echo "$Q06|Qwen/Qwen3-Embedding-0.6B|sub1m"
  [ "$WITH_NQ" = 1 ] && echo "$QNQ|Qwen/Qwen3-Embedding-8B|subnq"
  return 0
}

# stale <enc dir> <model id>: exit 0 if the query sidecar disagrees with the current contract
stale() {
  python - "$1" "$2" <<'PY'
import json, os, sys
sys.path.insert(0, ".")
from reusemap import models
enc, mid = sys.argv[1], sys.argv[2]
side = os.path.join(enc, "q.npy.json")
if not os.path.exists(side):
    print(f"  {enc}/q.npy: no sidecar -- treated as stale"); sys.exit(0)
rec = json.load(open(side))
st = models.stale_encoding(rec, models.get_contract(mid), "query")
print(f"  {enc}/q.npy: prefix {rec.get('text_prefix', '')[:24]!r}, prompt_name {rec.get('prompt_name')!r} -> "
      + ("STALE: " + "; ".join(st) if st else "current"))
sys.exit(0 if st else 1)
PY
}

pairs_point_here() {  # the pairs must read the query files this script rewrites, not other copies
  python - "$Q8" "$Q06" <<'PY'
import os, sys
q8, q06 = sys.argv[1], sys.argv[2]
want = {("P5p", "q_tgt.npy"): q8, ("P5q", "q_tgt.npy"): q8, ("P5r", "q_src.npy"): q06, ("P5r", "q_tgt.npy"): q8}
bad = []
for (p, f), enc in want.items():
    link = os.path.join("pairs", p, f)
    if not os.path.lexists(link):
        continue   # P5r is assembled by the llm stage, from these same directories
    a, b = os.path.realpath(link), os.path.realpath(os.path.join(enc, "q.npy"))
    if a != b:
        bad.append(f"pairs/{p}/{f} -> {a}, but this script rewrites {b}")
if bad:
    sys.exit("the pairs read different query files than the ones this script re-encodes:\n  " + "\n  ".join(bad)
             + "\nset Q8=/the/dir/the/pairs/use (and Q06 likewise), then run check again")
print("pairs read the query files this script rewrites (P5p, P5q, and P5r if assembled)")
PY
}

check() {
  log "check"
  [ "$(git branch --show-current)" = submission-eab ] || { echo "not on branch submission-eab"; exit 1; }
  [ -z "$(git status --porcelain -- reusemap scripts configs tests compat)" ] || { echo "uncommitted code"; exit 1; }
  [ -z "$(git diff HEAD --name-only)" ] || { echo "tracked files differ from HEAD: $(git diff HEAD --name-only | tr '\n' ' ')"; exit 1; }
  local t; t=$(python tests/test_invariants.py | tail -1)
  [ "$t" = "$EXPECTED_TESTS tests passed" ] || { echo "tests: got '$t', expected '$EXPECTED_TESTS tests passed' (install the final code first)"; exit 1; }
  echo "$t"
  python -c "import sys; sys.path.insert(0,'.'); from reusemap import models; c=models.get_contract('Qwen/Qwen3-Embedding-8B'); assert c['query_prefix']=='' and c['query_prompt_name']=='query', c; print('contract: Qwen3 queries take the instruction from the model prompt only')"
  for p in P5p P5q; do [ -f pairs/$p/meta.json ] || { echo "missing pairs/$p (link it from /root/reusemap_v3n/pairs)"; exit 1; }; done
  for f in "$Q8/doc.npy" "$Q8/q.npy" sub1m/queries.tsv; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
  [ -d "$Q06" ] || echo "note: $Q06 does not exist yet; P5r is redone only once encode_llm has made it"
  pairs_point_here
  local line enc mid n=0
  while IFS='|' read -r enc mid _; do
    if stale "$enc" "$mid"; then n=$((n + 1)); fi
  done < <(targets)
  log "check ok: $n stale query encoding(s) to redo"
}

requery() {  # GPU
  mkdir -p "$OLD" checks logs
  local enc mid data
  while IFS='|' read -r enc mid data; do
    if ! stale "$enc" "$mid"; then log "requery: $enc/q.npy already current"; continue; fi
    # same text file, token limit and precision as the original encoding (from its sidecar)
    read -r tsv msl fp16 batch < <(python - "$enc" "$data" <<'PY'
import json, os, sys
enc, data = sys.argv[1], sys.argv[2]
rec = json.load(open(os.path.join(enc, "q.npy.json"))) if os.path.exists(os.path.join(enc, "q.npy.json")) else {}
print(rec.get("tsv") or os.path.join(data, "queries.tsv"), rec.get("max_seq_len") or 512,
      "--fp16" if (rec.get("fp16") or os.environ.get("FORCE_FP16") == "1") else "-", rec.get("batch") or 16)
PY
)
    [ -s "$tsv" ] || tsv="$data/queries.tsv"
    local keep="$OLD/$(echo "$enc" | tr '/' '_')"; mkdir -p "$keep"
    for f in q.npy q.npy.json q.npy.txt; do [ -e "$enc/$f" ] && mv "$enc/$f" "$keep/"; done
    log "requery: $mid queries from $tsv (max-seq-len $msl, precision ${fp16}, batch $batch); old file in $keep/"
    if ! python scripts/encode.py --model "$mid" --kind query --tsv "$tsv" --out "$enc/q.npy" \
         --max-seq-len "$msl" --batch "$batch" $( [ "$fp16" = --fp16 ] && echo --fp16 ) > "logs/requery_$(basename "$enc").log" 2>&1; then
      log "requery: $mid failed -- see logs/requery_$(basename "$enc").log; restoring the old file"
      for f in q.npy q.npy.json q.npy.txt; do [ -e "$keep/$f" ] && mv "$keep/$f" "$enc/"; done
      return 1
    fi
    grep -h "model input for the first text" "logs/requery_$(basename "$enc").log" || true
    python scripts/encode_check.py --enc "$enc" --data "$data" --json "checks/$(basename "$enc")$( [ "$data" = subnq ] && echo _nq).json" \
      || { log "requery: encode_check failed for $enc -- see checks/"; return 1; }
  done < <(targets)
  log "requery done"
}

verify() {
  log "verify"
  python - "$Q8" "$Q06" "$QNQ" "$WITH_NQ" <<'PY'
import json, os, sys
sys.path.insert(0, ".")
from reusemap import models
q8, q06, qnq, with_nq = sys.argv[1:5]
dirs = [(q8, "Qwen/Qwen3-Embedding-8B")] + ([(q06, "Qwen/Qwen3-Embedding-0.6B")] if os.path.isdir(q06) else []) \
       + ([(qnq, "Qwen/Qwen3-Embedding-8B")] if with_nq == "1" else [])
bad = []
for enc, mid in dirs:
    rec = json.load(open(os.path.join(enc, "q.npy.json")))
    st = models.stale_encoding(rec, models.get_contract(mid), "query")
    ex = rec.get("model_input_example", "")
    once = ex.count("Instruct:") == 1
    print(f"  {enc}: contract {'current' if not st else 'STALE'}; model input {ex[:90]!r}; "
          f"instruction {'once' if once else 'NOT once'}")
    if st or not once:
        bad.append(enc)
sys.exit(f"not fixed: {bad}" if bad else 0)
PY
  log "verify ok: every Qwen3 query encoding in use carries the instruction once"
}

manifest() {  # fingerprints of the fixed inputs; make_numbers accepts a P5p/P5q/P5r number only if it matches
  python - "$WITH_NQ" <<'PY'
import datetime, hashlib, json, os, sys
sys.path.insert(0, ".")
from reusemap import provenance
pairs = ["P5p", "P5q"] + (["P5r"] if os.path.isdir("pairs/P5r") else []) + (["P5c"] if sys.argv[1] == "1" else [])
def sha(path, n=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:16]
doc = {"written": datetime.datetime.now().isoformat(timespec="seconds"),
       "git": provenance.git_version("."),
       "pairs": {p: provenance.data_fingerprint(f"pairs/{p}") for p in pairs},
       "query_files": {p: {"path": os.path.realpath(f"pairs/{p}/q_tgt.npy") if os.path.exists(f"pairs/{p}/q_tgt.npy") else None}
                       for p in pairs}}
for p, d in doc["query_files"].items():
    if d["path"]:
        d["sha256_16MB"] = sha(d["path"])
json.dump(doc, open("qwen_manifest.json", "w"), indent=1)
print("  qwen_manifest.json: " + ", ".join(f"{p} {fp}" for p, fp in doc["pairs"].items()))
PY
}

consistent() {  # every redone output must have read the inputs in qwen_manifest.json
  python - <<'PY'
import json, os, sys
if not os.path.exists("qwen_manifest.json"):
    sys.exit("no qwen_manifest.json: measure writes it (after verify)")
man = json.load(open("qwen_manifest.json"))["pairs"]
files = {"P5p": ["sweeps_qwen/sweep_P5p.json", "adapters/adapters_P5p.json", "results_xarch/P5p.json"],
         "P5q": ["sweeps_qwen/sweep_P5q.json", "adapters/adapters_P5q.json", "results_xarch/P5q.json"],
         "P5r": ["results_llm/P5r.json", "adapters/adapters_P5r.json"], "P5c": ["results/P5c.json"]}
bad = 0
for p, fp in man.items():
    for f in files[p]:
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        got = d.get("data_fingerprint") or (d.get("provenance") or {}).get("data_fingerprint")
        ok = got == fp
        bad += not ok
        print(f"  {f}: {'reads the fixed inputs' if ok else f'MISMATCH (read {got}, fixed inputs are {fp})'}")
sys.exit(f"{bad} output(s) did not read the fixed query files" if bad else 0)
PY
}

measure() {  # CPU
  verify
  pairs_point_here
  manifest
  mkdir -p "$OLD" logs sweeps_qwen
  # results made from the old query vectors are moved aside ONCE; a second 'measure' keeps what the
  # first one produced (its replays are registered, so run_pair skips them as already run)
  if [ ! -f "$OLD/.results_moved" ]; then
    for f in results_xarch/P5p.json results_xarch/P5q.json adapters/adapters_P5p.json adapters/adapters_P5q.json \
             results_llm/P5r.json adapters/adapters_P5r.json; do
      [ -e "$f" ] && { mkdir -p "$OLD/$(dirname "$f")"; mv "$f" "$OLD/$f"; log "moved stale $f to $OLD/"; }
    done
    date -Is > "$OLD/.results_moved"
  fi
  local t=$(( NCORES / 8 )); [ "$t" -ge 1 ] || t=1
  export OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t
  log "measure: sweeps, adapters, replays for P5p/P5q, and P5r, in parallel ($t threads each)"
  local pids=()
  python scripts/sweep.py pairs/P5p pairs/P5q --config configs/msmarco_1m_v3.json --kprime 10,25,50,100,200,500 \
    --mapped --out-dir sweeps_qwen > logs/qwen_sweeps.log 2>&1 & pids+=($!)
  for p in P5p P5q; do
    python scripts/adapter_families.py pairs/$p --config configs/msmarco_1m_v3.json --out-dir adapters \
      > logs/qwen_adapters_$p.log 2>&1 & pids+=($!)
  done
  THREADS=$t bash scripts/submission_runs.sh xarch > logs/qwen_xarch.log 2>&1 & pids+=($!)
  if [ -d "$Q06" ]; then
    THREADS=$t bash scripts/submission_runs.sh llm > logs/qwen_p5r.log 2>&1 & pids+=($!)
  else
    log "measure: $Q06 absent, P5r not redone"
  fi
  if [ "$WITH_NQ" = 1 ]; then
    python scripts/run_pair.py pairs/P5c --config configs/nq_100k_v2.json --out results/P5c.json --name P5c \
      > logs/qwen_p5c.log 2>&1 & pids+=($!)
  fi
  waitall "${pids[@]}" || log "measure: ${#FAILED[@]} job(s) failed -- read logs/qwen_*.log"
  local n_ok=0
  for f in sweeps_qwen/sweep_P5p.json sweeps_qwen/sweep_P5q.json adapters/adapters_P5p.json adapters/adapters_P5q.json \
           results_xarch/P5p.json results_xarch/P5q.json; do
    if [ -s "$f" ]; then n_ok=$((n_ok + 1)); else log "measure: missing $f"; fi
  done
  [ -d "$Q06" ] && for f in results_llm/P5r.json adapters/adapters_P5r.json; do
    [ -s "$f" ] && n_ok=$((n_ok + 1)) || log "measure: missing $f"; done
  log "measure: $n_ok output file(s)"
  manifest   # again: the llm stage may only now have assembled pairs/P5r
  consistent || { log "measure: an output read other inputs than the fixed queries -- see above"; return 1; }
  log "measure: every output read the fixed query files"
  [ ${#FAILED[@]} -eq 0 ]
}

collect() {
  local out="qwen_rerun_outputs_$(date +%Y%m%d_%H%M).zip" want=() f
  for f in qwen_manifest.json sweeps_qwen adapters/adapters_P5p.json adapters/adapters_P5q.json adapters/adapters_P5r.json \
           results_xarch results_llm coverage_llm checks "$Q8/q.npy.json" "$Q8/q.npy.txt" "$Q06/q.npy.json" \
           "$Q06/q.npy.txt" results/P5c.json logs/requery_*.log logs/qwen_*.log "$OLD"; do
    [ -e "$f" ] && want+=("$f")
  done
  # the superseded VECTORS are large and stay on the instance; their sidecars travel
  zip -qr "$out" "${want[@]}" -x "*.npy" || { log "zip failed"; return 1; }
  log "wrote $out ($(du -h "$out" | cut -f1)) -- send this file"
}

case "$STAGE" in
  check|requery|verify|measure|consistent|collect) "$STAGE" ;;
  all) check; requery; verify; measure || true; collect ;;
  *) echo "usage: bash scripts/qwen_rerun.sh check|requery|verify|measure|consistent|collect|all"; exit 2 ;;
esac
