# Run guide: what is left after the 2026-11-09 submission run

> **2026-11-10: everything below has been run and is in the paper. The last two experiments (seeds on every windowed pair; the planner at 8.8M documents) are in `RUN_FINAL.md`.**

Deadline: **Wed 11 Nov 2026, 17:00 PT (Thu 12 Nov, 02:00 Paris).** The submission run
(`submission_outputs_20261109_1822.zip`) is in the paper. Every remaining measurement is marked
`[pending: …]` in it and fills in from the zips this guide produces. Run on the same instance,
in the folder that holds that run (`/root/sub13run`), so `enc/`, `pairs/` and the committed plan
are where the scripts expect them.

**Why another round (2026-11-09; that round is done).** That run used the code from 8 Nov (`e19eb37`, 76 tests), not the final
code. Two things follow:

1. **Its affine and low-rank adapter fits are not usable.** The ridge penalty was fixed at
   1e-2 × n (λ = 100 at 10,000 pairs). That is larger than nearly every eigenvalue of the
   centred embeddings, so the map collapsed onto the mean vector and every query went to the
   same documents (0–19% relative recall). The code now picks the penalty on held-out pairs.
   The MLP starts from that fit, so all families are refitted. Procrustes and MLP from the
   first run stay in the paper until then.
2. **That version did not run** P5p/P5q (adapters or replays) or any of the extension.

**Qwen3 queries (found 2026-11-09, after this guide's first runs started).** The Qwen3 contracts
sent the query instruction twice, so every Qwen3 query vector, and everything measured on P5p,
P5q and P5r, must be redone. `RUN_QWEN.md` is that rerun: once everything now running has
finished, install the final code and run `scripts/qwen_rerun.sh`. Nothing else is affected.

## What is left, in priority order

| # | Measurement | Command | Hardware | Time | Paper |
|---|---|---|---|---|---|
| 1 | NQ: four encodings, plan **committed**, replays at the planned depths | `extension_runs.sh` encode, then `core` | GPU 20-40 min + CPU ~1 h | | §7.5 |
| 1b | Inline-repair latency (ms for 1, 25, 37, 50 documents) and throughput | `extension_runs.sh throughput` | GPU | ~10 min | §7.3 |
| 2 | Every adapter family refitted, all 14 pairs incl. P5p/P5q | `submission_runs.sh adapters` | CPU | 1.5-2 h at 4 at a time | Table 5 |
| 3 | P5p/P5q full replays | `submission_runs.sh xarch` | CPU | ~1 h | Table 4 |
| 4 | P5r (Qwen3-0.6B → 8B), P11 at seeds 1-4 | `extension_runs.sh` encode_llm, then `all` | GPU <1 h + CPU | | §9 |
| 5 | Zipf(0.9), judged-query sweep | (in `extension_runs.sh all`) | CPU | | §6, §4, §9 |

Rows 2 and 3 run on the CPU while rows 1 and 1b run on the GPU. Their P5p/P5q outputs, and row 4's
P5r, are made from the old Qwen3 queries and are replaced afterwards by `RUN_QWEN.md` (order: these
rows → collect → install the final code → `qwen_rerun.sh`); the paper uses a P5p/P5q/P5r result only
if it matches that rerun's `qwen_manifest.json`, so nothing needs deleting by hand. Row 1b is ten minutes and answers
the paper's largest practical question (what a cold query waits), so do not skip it. Every `[pending]` note in the
paper maps to one of these rows:

| Paper pending note | Row | Stage |
|---|---|---|
| affine and low-rank columns; P5p/P5q adapters (Table 5) | 2 | `adapters` |
| full replays of P5p and P5q (Table 4) | 3 | `xarch` |
| NQ: committed plan against replay (§7.5) | 1 | `encode`, `bm25`, `pairs`, `measure`, `plan`, `replay`, `check` |
| P5r static measurements and adapters (§9) | 4 | `encode_llm`, `p5r` |
| P11 at four further seeds (§9) | 4 | `seeds` |
| Zipf(0.9) replays, bound on their curves (§6, §9) | 5 | `zipf` |
| judged queries D1 and D3 need (§4) | 5 | `qsweep` |
| inline-repair latency of a cold window; cov into accelerator-hours (§7.3, §8) | 1b | `throughput` |

The other pending notes are for the authors: names, the repository URL.

## Step 0 — install the final code (5 min)

```bash
cd /root/sub13run
git status --short                     # must be clean in reusemap/ scripts/ configs/ tests/
unzip -o /path/to/reusemap_submission_changes.zip -d /tmp/rsc
R=/tmp/rsc/reusemap_submission_changes
cp $R/*.md . && rm -f APPLY.md
cp -r $R/{compat,configs,reusemap,scripts,tests} .
sha256sum -c $R/CHANGES_SHA256.txt --quiet && echo "files intact"
chmod +x scripts/submission_runs.sh scripts/extension_runs.sh
git add -A *.md compat configs reusemap scripts tests
git commit -m "Final code: held-out adapter ridge, extension drivers"   # REQUIRED: drivers refuse uncommitted code
python tests/test_invariants.py | tail -1   # 81 tests passed   (~25 s)
python tests/test_runbook.py | tail -1      # runbook ok (...)
bash scripts/submission_runs.sh preflight   # must end with "preflight ok"
```

This does not touch the committed `planner/plan.csv` or any result. Do **not** run
`submission_runs.sh all` again: it would re-run stages that are done.

If preflight stops with `missing pairs/P5p`, the encoder→LLM pairs are still in the earlier
folder. Check every input the remaining stages read:

```bash
for d in pairs/P5p pairs/P5q enc/Qwen3-Embedding-8B sub1m/bm25.npy subnq/corpus.tsv enc_nq; do
  [ -e "$d" ] && echo "ok       $d" || echo "MISSING  $d"; done
```

Link anything missing from `/root/reusemap_v3n` (the 2026-11-07 folder), for example
`ln -s /root/reusemap_v3n/pairs/P5p pairs/P5p`, then run preflight again. `enc_nq` may be absent:
Step 1 creates it (it holds the NQ encodings).

## Step 1 — GPU: the NQ encodings first (they gate the core NQ test)

```bash
bash scripts/extension_runs.sh encode > logs/nq_encode.log 2>&1        # 20-40 min
tail -3 logs/nq_encode.log                                             # must end with "encode done"
bash scripts/extension_runs.sh encode_llm > logs/encode_llm.log 2>&1   # Qwen3-0.6B for P5r, <1 h
bash scripts/extension_runs.sh throughput > logs/throughput.log 2>&1   # ~10 min
```

## Step 2 — CPU, at the same time: adapters, then the P5p/P5q replays

```bash
ADAPTER_JOBS=4 nohup bash scripts/submission_runs.sh adapters > logs/adapters.log 2>&1 &
```

`ADAPTER_JOBS=4` fits four pairs at a time; the first run was killed for memory at twelve. On a
1 TB machine `ADAPTER_JOBS=7` halves the time. The stage first moves the old fits to
`adapters_superseded/<time>/`. Pass: `logs/adapters.log` ends with `13.1 done`, and there are
14 files in `adapters/`, each containing `"affine_ridge": "held-out"`.

When it has finished:

```bash
nohup bash scripts/submission_runs.sh xarch > logs/xarch.log 2>&1 &   # P5p, P5q replays, ~1 h
```

Pass: `results_xarch/` has P5p.json and P5q.json. If the 1M Qwen3-8B vectors are not in
`enc/Qwen3-Embedding-8B`, find them (`ls -d enc*/Qwen3*`) and put `LLM_TGT=<that dir>` in front.
Then:

```bash
bash scripts/submission_runs.sh check; bash scripts/submission_runs.sh collect   # collect even if a check failed
```

Pass: a new `submission_outputs_<date>.zip`.

## Step 3 — the extension (CPU, ~1.5-2 h, after Step 1's NQ encodings)

```bash
bash scripts/extension_runs.sh preflight                  # "NQ encodings: all five present", "preflight ok"
nohup bash scripts/extension_runs.sh all > logs/extension.log 2>&1 &
```

This step and Step 2 can run together on 224 cores. If memory is tight, wait for the
adapters to finish first. Short on time: `bash scripts/extension_runs.sh core` runs only the
NQ prediction (row 1). Whatever the NQ plan says is the prediction; do not change it.

## Step 4 — send back

- the new `submission_outputs_<date>.zip` (Step 2)
- `extension_outputs_<date>.zip` (Step 3)
- if anything failed: `logs/` (zip the folder)

## What passing looks like

| Output | Pass |
|---|---|
| `adapters/adapters_P*.json` | 14 files, each with `"affine_ridge": "held-out"` and a `ridge` on every affine/low-rank entry |
| `results_xarch/` | P5p.json and P5q.json |
| `planner_check_nq.md` | `11 pairs scorable, 11 pre-committed` |
| `results_seeds/`, `seeds_summary.md` | 4 seed files; the range of the margin across the five runs |
| `results_llm/P5r.json`, `adapters/adapters_P5r.json` | both present |
| `results_zipf/`, `curve_check_zipf.md` | 4 files; `0 points above the independent bound` |
| `query_sweep/` | 10 files, each with `smallest_q_all_seeds_agree` |
| `throughput.json` | one entry per model with `docs_per_second` and `repair` timings for 1, 25, 37 and 50 documents |

## If something fails

Read the stage's log in `logs/`, fix the cause, and re-run only that stage, then the checks and
the zip:

```bash
bash scripts/submission_runs.sh <stage>      # adapters, xarch, llm
bash scripts/submission_runs.sh check; bash scripts/submission_runs.sh collect   # collect even if a check failed
bash scripts/extension_runs.sh <stage>       # encode, encode_llm, throughput, bm25, pairs, measure, replay, seeds, zipf, qsweep, p5r
bash scripts/extension_runs.sh check; bash scripts/extension_runs.sh collect
```

Full detail per stage: RUNBOOK.md, Phase 13 (main run) and Phase 14 (extension).

## Not measurements, but needed before submission (authors)

1. Author names and affiliations (single-blind review).
2. The public GitHub repository for the artifacts (EAB requires it).
3. The AI-generated-content declaration the CfP requires.
