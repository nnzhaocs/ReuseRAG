# Rerun of the Qwen3 measurements (instruction sent twice)

## What was wrong, and what it touches

The encoding contracts for Qwen3-Embedding-8B and -0.6B set the query instruction
("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:")
as a literal prefix **and** named the model's own `query` prompt, which is the same string. Every
Qwen3 **query** was therefore encoded with the instruction twice. Fixed in the code: the contracts
now take the instruction from the model prompt only; `apply_contract` refuses any contract that
would send an instruction twice; `encode_check` fails vectors made under an earlier contract; and
`encode.py` records the exact text the model saw.

| Affected, redone here | Not affected |
|---|---|
| Qwen3-8B and Qwen3-0.6B **query** vectors | every document vector (their prefix and prompt were empty) |
| P5p, P5q (MiniLM / bge-base → Qwen3-8B): static sweeps, adapter families, full replays | all ten text pairs, P0, P7, P8 |
| P5r (Qwen3-0.6B → 8B): measure-only run and adapter families | the NQ test (N0-N14), seeds, Zipf, query sweep |
| | the latency kit (it times documents) |

P5c (NQ, MiniLM → Qwen3-8B, measure-only) also used Qwen3-8B queries but is not in the paper;
`WITH_NQ=1` redoes it as well.

## When

**After everything now running or queued has finished**: the 45186ed submission stages, the
extension, and the latency kit. Two reasons: this step installs new code into `/root/sub13run`
(replacing a driver script while bash is running it breaks that run), and it rewrites the Qwen3
query files that the running P5p/P5q/P5r stages read. Collect and send those runs' zips first, as
planned; they stay valid for everything except P5p, P5q and P5r.

## Steps (~10 GPU-min, then ~1.5-2 h CPU)

**0. Install the final code** (same as RUN_NOW.md Step 0, with this zip):

```bash
cd /root/sub13run
git status --short                     # must be clean in reusemap/ scripts/ configs/ tests/ compat/
unzip -o /path/to/reusemap_changes_<commit>.zip -d /tmp/rsc
R=/tmp/rsc/reusemap_submission_changes
cp $R/*.md . && rm -f APPLY.md
cp -r $R/{compat,configs,reusemap,scripts,tests} .
sha256sum -c $R/CHANGES_SHA256.txt --quiet && echo "files intact"
chmod +x scripts/*.sh
git add -A *.md compat configs reusemap scripts tests
git commit -m "Final code: Qwen3 contract fix"
python tests/test_invariants.py | tail -1   # 81 tests passed
```

**1. Check** (seconds):

```bash
bash scripts/qwen_rerun.sh check
```

Pass: `contract: Qwen3 queries take the instruction from the model prompt only`, `pairs read the
query files this script rewrites`, and `check ok: N stale query encoding(s) to redo`. Each Qwen3
query file is listed as `STALE` (made with the instruction twice) or `current`. If it says the pairs
read different files, set `Q8=` (and `Q06=`) to the directory the pairs link to, as the message
says, and check again. If `pairs/P5p` is missing, link it from `/root/reusemap_v3n/pairs`.

**2. Re-encode the queries** (GPU, ~10 min; documents are not touched):

```bash
bash scripts/qwen_rerun.sh requery > logs/qwen_requery.log 2>&1
tail -3 logs/qwen_requery.log          # "requery done"
bash scripts/qwen_rerun.sh verify      # each file: "contract current", "instruction once"
```

Each query file is re-encoded from the same text file, token limit, precision and batch as its
original (read from the old sidecar); the old file is kept in `superseded_double_instruction/`, and
`encode_check` runs on the result. `verify` prints the exact model input for the first query: it
must start with one `Instruct: ...\nQuery:` and then the query text. If the 8B model runs out of
GPU memory, put `FORCE_FP16=1` in front of `requery` (it is then recorded in the sidecar).

**3. Redo the measurements** (CPU, ~1.5-2 h, all in parallel):

```bash
nohup bash scripts/qwen_rerun.sh measure > logs/qwen_measure.log 2>&1 &
tail -f logs/qwen_measure.log          # "measure: 8 output file(s)", then
                                       # "measure: every output read the fixed query files"
```

It first moves the P5p/P5q/P5r results made from the old queries to
`superseded_double_instruction/`, then runs: the P5p/P5q static sweeps (`sweeps_qwen/`), the
adapter families on P5p and P5q (`adapters/`), the P5p/P5q full replays (`results_xarch/`), and
P5r (`results_llm/P5r.json`, `adapters/adapters_P5r.json`). Re-running `measure` is safe: it keeps
what an earlier `measure` produced and skips registered replays.

Before measuring it writes `qwen_manifest.json`: the fingerprint of each pair's input files after
the fix. Every sweep, adapter file and replay records the fingerprint of the files it read, and
the last step checks each one against the manifest (`bash scripts/qwen_rerun.sh consistent` runs
that check alone). The paper's number script applies the same test, so a P5p/P5q/P5r result made
from the old query files is never used, even if an older zip is unpacked over a newer one.

**4. Send back**:

```bash
bash scripts/qwen_rerun.sh collect     # "wrote qwen_rerun_outputs_<time>.zip"
```

Send `qwen_rerun_outputs_<time>.zip`. It holds `qwen_manifest.json`, the new results, the new query sidecars (with the
recorded model input), the superseded sidecars, and the logs; no vectors.

`bash scripts/qwen_rerun.sh all` runs steps 1-4 in one go once the code is installed.

## If something fails

Read the log it names (`logs/requery_*.log`, `logs/qwen_*.log`), fix the cause and re-run the
same stage. `requery` restores the old query file if an encoding fails, and skips files that are
already current. In `measure`, a single failed job does not stop the others; re-run `measure`
after fixing it.
