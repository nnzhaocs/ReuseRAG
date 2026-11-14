# RUNBOOK — finishing the experiments (replay v3; sweep and band tie-break 2026-11-02)

Follow the phases in order. Every step has a **command**, a **pass check**, and what to do
if it fails. Commands run from the repo root. Timings marked *measured* come from this repo
or from the team's first-run logs; the rest are estimates.

`tests/test_runbook.py` checks that every `python scripts/...` command below uses scripts and
flags that exist and that every config named here exists. Run it after editing this file.

---------------------------------------------------------------------------------------------
## Status 2026-11-09 (after the Phase 13 run) — read this first

- **Phase 13 ran on 2026-11-09** (instance `/root/sub13run`, commit `1725621`; its preflight counts, 76 tests and 82
  runbook commands, match `e19eb37`, whose drivers expected 76 (the final code's drivers expect 81; Phase 13.0), and what that version lacks changes only drivers and
  checks, not a measurement it made). Done: 13.2 (plan committed at
  15:16, then the eight K'=25 replays: price within 4e-5 of N on every scorable pair, quality
  within tau of G(25) on 7 of 8, P5 missing by 6e-5; S4 wins P2 and P9 at K'=25), 13.3 (P0 at
  window 50), 13.5 (window masses; Theorem 1 within 0.81% of every text replay point), 13.6
  (sample sweep), 13.7 (TREC-DL G at 25/50 on P11 and P14). Outputs:
  `submission_outputs_20261109_1822.zip`, merged into the 2026-11-09 data archive.
- **13.1 must be re-run.** Its affine and low-rank fits used a fixed ridge (1e-2 * n = 100 at
  10,000 pairs) larger than nearly every eigenvalue of the centred embeddings, which collapsed
  the map onto the mean vector (relative recall 0-19% on text). `compat.fit_affine` now picks
  the ridge on held-out pairs; the MLP starts from that fit, so all families are refitted. The
  stage moves the earlier files to `adapters_superseded/`. Procrustes and MLP from the first
  run are reported until then. P5p/P5q were not in that version: the re-run adds them.
- **13.9 (xarch, P5p/P5q replays) did not run** (not in that version's `all`).
- **What remains:** 13.1 re-run, 13.9, then Phase 14. `RUN_NOW.md` has the order.
- **Qwen3 query vectors were encoded with the instruction twice** (contract prefix plus the
  model's own `query` prompt, the same string). Fixed in the contracts, guarded in
  `apply_contract` and `encode_check`. P5p, P5q and P5r are redone by `scripts/qwen_rerun.sh`
  after the queued runs; `RUN_QWEN.md` is the guide. Documents and all other pairs are unaffected.

## Status 2026-11-08

- **Done since the 2026-11-05 table below.** Phase 11.0 ran (commit `fbb23c9`: negative
  primary G is judgment sparsity; on TREC-DL G >= 0; every result carries `secondary_frontier`).
  Phase 11.1 ran (`sweeps_fine/`): K'=50 is the
  smallest depth admissible on every pair (P11: G(25) = 0.01368 > eps = 0.01), recorded in
  `configs/msmarco_1m_v3.json` `_window_decided`. All ten text pairs were re-run at K'=50
  (P2-P5, P9 at `fbb23c9`; P10-P14 at `57de2da`). P8 was run after the codesage contract fix.
  Qwen3-Embedding-8B is `unverified` (contract filled 2026-11-06) and appears only in the
  encoder->LLM pairs P5c/P5p/P5q.
- **Phase 11.2 is superseded, not done.** No predictions were committed before the K'=50
  replays, so the paper claims no predictive test from `predictions/`. Its pre-committed test
  is 13.2: `planner/plan.csv` is committed before the K'=25 replays and `check_planner.py`
  scores only plans that precede them. **Phase 11.3 is done:** the tau curve is in the paper;
  the provenance gap is closed because `fbb23c9..57de2da` changes no code and the registered
  config differs only in `_` notes (config hash `5b84535a5f611ed1` at both).
- **What remains is Phases 13 and 14** (`RUN_NOW.md` runs them in priority order).
- **Known, deliberately not changed before submission** (the measurement path stays identical
  to the code that produced the reported results; none of these can change a number on valid
  input): `runner.run_pair` reads `primary_setting.repeat_rate` for the v2 heat-gate weights
  before its friendly v2 check (v2 only; KeyError instead of ValueError);
  `measure._subset_weights` fails on an empty query list with a numpy error (no caller passes
  one); `probe.estimate_miss` at a window below K fails with IndexError (the probe only runs at
  the replay window); `heat_gate`'s `signal_scores` is not JSON-safe (the runner pops it).
  `run_pair.py` now refuses `arrival_target_frac` outside (0, 1).
- **Where the paper's numbers come from:** the 2026-11-07 data archive, through the paper's
  `build/make_numbers.py`. `PAPER_NUMBERS.md`, `UNVERIFIED.md` and `RERUN_K50_STATUS.md` are
  historical records of earlier states and are not updated.
- **Field meanings worth knowing before reading a result file.** `Q` counts every judged query
  including the 97 TREC-DL ones; the replayed (servable) set is `n_eval + n_traffic_pool` = 6,980.
  `features.rho` is rho at the replay window only if that depth is on the result's own `rho`
  ladder; the K'=50 text results' ladder starts at 500, so their stored `features.rho` is rho(500)
  and rho(50) must be read from `sweeps_fine/` (the paper does this). `eps_sensitivity` rows
  recompute the target as M_T - eps, so at eps = 0.02 S0 "reaches" it on any pair whose quality
  gap is below 0.02 (P3, P13): that is the definition, not a cost bug.

## Where things stood (2026-11-05)

**The v3 1M run happened.** Twelve pairs (ten text + the identity control + the code pair), 73
registered runs all `ok`, no dirty trees, no `_dev_overrides` on any paper pair, zero invariant
failures. Claim A's pre-registered falsification condition fired, so the framing is settled as
B (Phase 5.9). What of it is publishable and what is archive-only is itemised in
`PAPER_NUMBERS.md` — read that before putting any number in a table. The short version: the
measurements are sound and reusable; the K′ choice and the prediction check are not yet done,
and those two are what still block the table (Phase 11).

| Item | State | Action |
|---|---|---|
| v3 1M run (2026-11-04): 12 pairs, 828 completing combinations | **measurements usable**; headline claims verified (coverage↔cost to 0.017%, G never binding, S1/S3 never winning) | `PAPER_NUMBERS.md`, then Phase 12 |
| `window_frac = 5e-05` (K′ = 50) as used by that run | **not justified from a frontier**: 50 is the smallest *measured* depth and G is admissible at every depth, so the frontier is censored from below | Phase 11.1 — sweep K′ = 10, 25 (**done 2026-11-07**: 50 confirmed) |
| `predictions/` for the ten paper pairs | **missing**: without them Phase 10.3 cannot run, and the protocol revision stays a re-tuning rather than a predictive test | Phase 11.2 |
| `CROSS_FAMILY.md` in that run's output | **mislabelled**: titled BERT→Qwen3-8B, but its P5 row is bge-m3→bge-small (ρ@50 = 0.2412, same as the main map); P5p/P5q are `_dev_overrides` runs | retitle or delete; there is no cross-family measurement |
| `quality_5models.txt` in that run's output | **stale** (2026-10-30, pre-dates the Phase 4.2 sidecar rename): four of five models still print `no sidecar` | re-run after 4.2 |
| First 1M results (P0–P5, P7–P9, drift arms, P5p/P5q) | **not usable as data**: v1 replay leaked eval queries into traffic; bge-m3, codesage-base and Qwen3-8B encodings suspect | archive them (Phase 0.3) |
| v2 CLS / dev results | **not usable as data**: v2's replay could not separate the strategies at all (see DESIGN.md "Replay protocol v3"), and they came from uncommitted code | archive them |
| `sub1m/` (MS MARCO 1M), `subcsn/` (CodeSearchNet py + CoSQA), BM25 files | reusable if the coverage checks in Phase 3 pass | keep |
| Text encodings MiniLM / bge-small / bge-base / bge-m3 / e5-base | made without contracts or sidecars; bge-small was mean-pooled, e5-base's query rows are misaligned, codesage-base's weights were never downloaded, bge-m3 was never re-encoded | re-encode and check (Phase 4; ~70 GPU-min for the four text models *measured*) |
| `configs/msmarco_1m_v3.json`, `csn_py_v3.json` | drafts | pre-register (Phase 1) |
| `configs/nq_100k_v2.json` | draft, measure-only so v3 does not touch it | pre-register (Phase 1) |
| Qwen3-Embedding, codesage-*, jina-code contracts | `blocked` | Phase 2 |
| Coverage spread across the candidate source models | **measured** on the v3 run: at K′ = 50, 0.2551 (bge-m3) / 0.2622 (bge-small) / 0.2643 (bge-base) / 0.2649 (MiniLM) / 0.2663 (e5-base) — a spread of 0.0112 over all five text models, far too narrow to be a map feature | settled; no action |

---------------------------------------------------------------------------------------------
## Before you start — what this RUNBOOK assumes you already have

Nothing below is a code problem; it is what the repo cannot carry.

**Data you supply** (not in the repo — too large; Phase 3 builds the subsamples from it):
MS MARCO passage `collection.tsv`, `queries.dev.tsv` and the dev + TREC-DL 2019/2020 qrels;
CodeSearchNet Python parquet plus CoSQA dev; ORCAS click counts for RQ4. `sub1m/`, `subcsn/`,
`enc/`, `pairs/` are all generated.

**Machines**: a GPU box for Phase 4 (~2 h for the five text models at 1M documents, *measured*)
and, if you do it, Phase 0.0 (Qwen3-8B over the same 10^6 documents, longer). **Those two
contend for the same card**: "start 0.0 and work on CPU" holds from Phase 5.5 on, but Phases
1-4 come first and Phase 4 wants the GPU. On one card, run 0.0's download and `golden_check.py`
now and queue its encoding behind Phase 4; on two cards, or one with room for an 8B and a 768-d
model at once, they overlap. Everything else is CPU. Phase 5.5 and Phase 7 want **>= 32 GB RAM** — a pair's two
document matrices are memory-mapped, but a pair whose matrices exceed RAM becomes disk-bound
(1M docs: 384-d 1.5 GB, 768-d 3 GB, 1024-d 4 GB, 2560-d 10 GB).

**Three phases are not commands:**
- **Phase 1** and the second commit in **Phase 5.5e** are pre-registration decisions.
- **Phase 5.9** is the framing decision, and it blocks Phase 7. Running the commands in order
  will walk straight past it.
- **Phase 5** records the pair choice in `DECISIONS.md` before Phase 6.

**Two phases are optional:**
- **Phase 0.0 / cross-family** needs Qwen3 unblocked in `configs/models.json`; `encode.py`
  refuses a `blocked` contract, and `--allow-blocked` output is diagnostic only, never paper
  data. Skip it and the main table is unaffected — you lose the only lever on rho.
- **Phase 9 (NQ / EmbedFlow)** needs the same unblock. It is the design's only external
  validation point, so skipping it weakens external validity but blocks nothing.

**Check the checkout first**, which needs no data at all:
```bash
python tests/test_invariants.py          # pass: "81 tests passed"
python tests/test_runbook.py             # pass: "runbook ok"
python scripts/make_synthetic.py --out /tmp/syn && python scripts/sweep.py /tmp/syn/syn_mid --kprime 10,50,100
```
The last line exercises the sweep end to end on generated data, so a failure there is the repo,
not your data.

---------------------------------------------------------------------------------------------
## Phase 0 — environment (once per machine)

### 0.0 Start the cross-family track NOW (GPU, hours; everything else is CPU)
Cross-family (BERT to LLM) is the only thing that can move rho, and it has the longest lead time:
unblock Qwen3 in `configs/models.json`, fill the contract from the card, `golden_check.py`, then
encode 10^6 documents with an 8B model. That is hours of GPU before any measurement exists, so
**start it first and run Phases 5.5-7 on CPU while it proceeds** — with the caveat from "Before
you start": if you have one GPU, its encoding queues behind Phase 4 rather than running
alongside it, so what starts now is the unblock, the contract and `golden_check.py`. The code pair (`sweeps_csn`) is
the only cross-domain evidence available until it lands, and its encoding is unverified: run
`encode_check.py` on codesage-small before quoting its ladder anywhere.

It is numbered 0.0 because it is the only item with a multi-hour lead time: start it, then do Phases 1-7 on CPU while it proceeds. It gates nothing before Phase 7, and it is the only thing that can move rho if the sweep finds it flat.

### 0.1 Install and test
```bash
pip install -r requirements.txt
python tests/test_invariants.py          # pass: "81 tests passed"
python tests/test_runbook.py             # pass: "runbook ok"
```
The count does not reveal one skip: the check that the source parses on Python 3.9-3.11 runs
only if such an interpreter is on PATH (the test prints "(no python3.9-3.11 ...)" otherwise).
On a 3.12-only machine install one (e.g. `python3.11`) if that check matters to you.
Use an environment **pinned to the library versions the model cards specify**. Do not
import `reusemap.tfcompat` for paper encodings: `encode.py` records loaded shims in every
sidecar (`"shims"`), and an encoding listing `reusemap.tfcompat` is not usable. **One exception,
recorded rather than hidden:** codesage's remote code (P7/P8) does not load under
transformers 5.x / torch 2.x without `compat/sitecustomize.py` (on `PYTHONPATH=compat`), which
only restores three moved or removed APIs (no weights, pooling, prefixes or numerics; see its
docstring). Those encodings list `"compat/sitecustomize"` in `"shims"`; that entry is expected
for codesage and for nothing else.

### 0.2 Hardware
- GPU only for Phase 4 (encoding). Everything else is CPU replay over cached vectors.
- RAM: N × dim × 4 bytes per document matrix (1M docs: 384-d 1.5 GB, 768-d 3 GB, 1024-d 4 GB,
  2560-d 10 GB). Vectors are memory-mapped, but a pair whose two matrices exceed RAM becomes
  disk-bound: use ≥ 32 GB for pairs with large encoders.

### 0.3 Archive any earlier results (do not delete them)
```bash
mkdir -p archive_v1
for D in results results_dev results_measure pairs; do
  [ -d "$D" ] && mv "$D" "archive_v1/$D" || echo "no $D to archive"
done
```
Written to be safe on a fresh checkout: a bare `mv results ...` fails with "cannot stat" when
there is nothing to archive yet. `results/` then starts empty, so `results/RUNS.tsv` lists only
v3 runs. Keep the archive: the v1 and v2 numbers are the evidence for several `DECISIONS.md`
rows, and `summarize.py` refuses them anyway.

---------------------------------------------------------------------------------------------
## Phase 1 — decisions and pre-registration (before any v3 pair runs)

For each of `configs/msmarco_1m_v3.json`, `configs/csn_py_v3.json`, `configs/nq_100k_v2.json`:
1. Decide every value the `_status` / `_DECIDE` / `_probe` notes point at: `eval_frac`, the
   `traffic_ratio` grid, `primary_setting`, `delta`, `eps`, `eps_sensitivity`, the `tau` bands,
   `band_tiebreak`, `probe_queries` and `strat_per_query`. The NQ config's pass criterion is
   already stated and it is measure-only, so v3 does not change it.

   **`window_frac` is the exception: it is a TWO-STAGE decision and cannot be settled here.**
   Leave the placeholder, run Phase 5.5, choose the depth from the cost/quality frontier, then
   come back and make a *second* pre-registration commit for it. The sweep needs a window only
   to size its retrieval, so nothing downstream of Phase 1 depends on the placeholder — but
   everything downstream of the *choice* does, which is why it gets its own commit and its own
   `DECISIONS.md` row.
2. Delete the `_status` and `_DECIDE` keys (other `_` keys are documentation and may stay;
   `_`-keys never enter the run key).
3. Commit: `git add configs && git commit -m "pre-register v3: ..."`.

Pass check: `git status` clean and `git log -1 -- configs/` shows the commit.
Changing a pre-registered value later = a new pre-registration commit, and every earlier
result is re-run under it (run keys change automatically).

**Two things to be honest about while doing this.**

- `primary_setting` must come from the measured ORCAS recurrence/arrival rate, not from which
  `traffic_ratio` produces a non-zero result. `scripts/predict.py` prints the predicted saving
  per ratio; it is 0 at ratios 1, 4 and 16 and ~10–12% at 64–256 for MS MARCO 1M at K' = 500.
  Picking the setting after reading that line is exactly the error this phase exists to prevent.
- Nothing was pre-registered before the development runs, and those runs have been seen. That
  cannot be undone, so what makes this revision defensible is its **grounds**: every change in
  v3 is justified by a mechanism (`f**K'` window completion, the coupon-collector arrival rate,
  the probe's cost arithmetic, `in_m` flipping each document once) and written down as a
  prediction in Phase 6.5 before the replay runs. Add a threats-to-validity line to the paper
  saying the protocol was revised after development runs on dev labels, and why.

---------------------------------------------------------------------------------------------
## Phase 2 — encoder contracts (before encoding anything)

Every model needs an entry in `configs/models.json`. `encode.py` refuses unknown models and
`blocked` ones.

### 2.1 Fill and unblock each contract you will use
From the model card: prefixes or prompt name, pooling, padding side, EOS, `revision`
(commit hash), `max_seq_len_cap`. Change `status` from `blocked` to `unverified` only when
every field is filled from the card. Current blockers and what they need:
- **Qwen3-Embedding-***: query prompt (the model's `query` prompt vs a task instruction),
  EOS, padding side (the 1M run's tokenizer reported `right`; contract expects `left`).
- **codesage-***: EOS requirement and pooling. codesage-base queries scored 0.79 while mapped
  target queries scored 0.96 on its documents: check the padding check output first.
- **jina-code**: needs a transformers 4.x environment (remote code).

### 2.2 Golden check (per model)
Copy the card's worked example into `configs/golden/<org>--<model>.json` (template:
`configs/golden/TEMPLATE.json`; texts and printed scores verbatim, card URL, date).
```bash
python scripts/golden_check.py configs/golden/Qwen--Qwen3-Embedding-8B.json
```
Pass: `PASS`. Then set the contract's `status` to `verified` and add
`"verified": {"date": ..., "golden": <file>, "versions": <printed versions>}`.
Fail: fix the contract or environment, never the expected scores. If a card prints no scores,
the model stays `unverified` and relies on the padding and ordering checks.

### 2.3 Commit the contracts
`git add configs/models.json configs/golden && git commit -m "contracts: ..."`

---------------------------------------------------------------------------------------------
## Phase 3 — data

### 3.1 MS MARCO 1M (reuse `sub1m/` if it passes; otherwise rebuild)
```bash
python scripts/build_subsample.py --collection collection.tsv \
  --qrels qrels.dev.tsv:dev 2019qrels-pass.txt:trecdl 2020qrels-pass.txt:trecdl \
  --queries queries.dev.tsv:dev msmarco-test2019-queries.tsv:trecdl msmarco-test2020-queries.tsv:trecdl \
  --N 1000000 --out sub1m
python scripts/bm25_run.py --corpus sub1m/corpus.tsv --queries sub1m/queries.tsv --k 1000 --out sub1m/bm25.npy
```
Pass: last line `judged coverage: 100.00%`; `sub1m/query_sets/trecdl.txt` exists (the v2
config refuses to run without it).

### 3.2 Code (CodeSearchNet Python test+valid, CoSQA dev)
```bash
python scripts/build_code_corpus.py --parquet csn/python-test.parquet csn/python-validation.parquet \
  --cosqa cosqa-dev.json --out subcsn
python scripts/bm25_run.py --corpus subcsn/corpus.tsv --queries subcsn/queries.tsv --k 1000 --out subcsn/bm25.npy
```
Pass: `subcsn/query_sets/cosqa.txt` and `csn.txt` exist; qrels rows ≥ judged queries.

### 3.3 NQ (only for the EmbedFlow comparison, Phase 9)
```bash
python scripts/build_nq_prefix.py --out subnq --prefix 100000 --n-queries 1000
```
None of the constructions tried so far matches EmbedFlow's "canonical prefix" (MiniLM 0.5875
prefix / 0.8426 random vs registry 0.7458). Get the construction from EmbedFlow's builder or
its author first; until then Phase 9 is a comparison, not a calibration.

---------------------------------------------------------------------------------------------
## Phase 4 — encode under contracts (GPU)

Rules: exact model id from `configs/models.json`; **the same `--max-seq-len` for both models
of a pair** (the smaller cap); `--fp16` only if both models of a pair use it.
Measured on the team GPU (first run, 1M MS MARCO docs): bge-m3 ~42 min, bge-base ~15 min,
bge-small ~7.5 min, MiniLM ~5 min.

### 4.1 Text (MS MARCO; all text pairs share max length 256)
```bash
for M in sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5 BAAI/bge-m3 intfloat/e5-base-v2; do
  python scripts/encode.py --model $M --kind doc   --tsv sub1m/corpus.tsv  --out enc/$(basename $M)/doc.npy --max-seq-len 256
  python scripts/encode.py --model $M --kind query --tsv sub1m/queries.tsv --out enc/$(basename $M)/q.npy  --max-seq-len 256
done
```
Pass for every encoding: exits 0; `padding check: ... -> ok` (a `fail` aborts the encode);
`self-retrieval check: top1 ... -> ok` on every `--kind doc` run (a `fail` aborts it);
the `.npy.json` sidecar shows the intended prefix/prompt and `"shims": []`.
`warn` (min cos 0.99–0.999) is acceptable for left-padded rotary models; note it.

### 4.2 Put the query sidecars where the tools read them (seconds)
`encode.py` writes `<out>.json` beside the vectors, so an encoding written to `query.npy` has its
sidecar at `query.npy.json` while `assemble_pair.py` links `q.npy` and `model_quality.py` /
`encode_check.py` look for `q.npy.json`. On the 2026-11-01 run that mismatch made four of five
models print "no sidecar" in the quality table, and e5's `q.npy.json` was a 0-byte file.
```bash
for M in all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5 bge-m3 e5-base-v2; do
  [ -f enc/$M/query.npy.json ] && cp enc/$M/query.npy.json enc/$M/q.npy.json
done
```
The **vectors** are fine - rename only the sidecar. Pass check: every row of the Phase 5 quality
table shows a contract status instead of `no sidecar`.

### 4.3 Check every text encoding before it is used (CPU, seconds)
```bash
for M in all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5 bge-m3 e5-base-v2; do
  python scripts/encode_check.py --enc enc/$M --data sub1m --json checks/$M.json
done
```
Pass: every line `ok` or `warn`, exit 0. These are the four failures of the first run, and only
the first of them had a check:

| Failure | Caught by |
|---|---|
| bge-small mean-pooled because the library ignored the pooling flags | the contract check in `encode.py` (it already did) |
| e5-base's query rows misaligned with the qrels (0.000 on 6,980 of 7,077) | `encode_check.py` "qrels: every query id has a vector", and `model_quality.py` now refuses instead of silently scoring the rest |
| codesage-base's download folder 3.3 MB — the weights were never there | `encode.py --self-check` (document text through the query path) and `encode_check.py`'s collapse check |
| bge-m3 simply never re-encoded | `encode_check.py` row counts, plus the missing sidecar |

Also check the download itself: compare the model's cache directory size against the parameter
count on its card. 3.3 MB is not a 356M-parameter model.

### 4.4 Code (two lengths, because MiniLM's cap is 256)
```bash
for M in sentence-transformers/all-MiniLM-L6-v2 codesage/codesage-small codesage/codesage-base; do
  python scripts/encode.py --model $M --kind doc   --tsv subcsn/corpus.tsv  --out enc_csn256/$(basename $M)/doc.npy --max-seq-len 256
  python scripts/encode.py --model $M --kind query --tsv subcsn/queries.tsv --out enc_csn256/$(basename $M)/q.npy  --max-seq-len 256
done
for M in codesage/codesage-small codesage/codesage-base; do
  python scripts/encode.py --model $M --kind doc   --tsv subcsn/corpus.tsv  --out enc_csn512/$(basename $M)/doc.npy --max-seq-len 512
  python scripts/encode.py --model $M --kind query --tsv subcsn/queries.tsv --out enc_csn512/$(basename $M)/q.npy  --max-seq-len 512
done
# the same check 4.3 runs on the text encodings, on the code ones. Phase 5 gates P8 on it,
# and Phase 0.0 refuses to quote codesage-small's ladder until it passes.
for D in enc_csn256/all-MiniLM-L6-v2 enc_csn256/codesage-small enc_csn256/codesage-base \
         enc_csn512/codesage-small enc_csn512/codesage-base; do
  python scripts/encode_check.py --enc $D --data subcsn --json checks/$(echo $D | tr / _).json
done
```
Code encodings write straight to `q.npy`, so 4.2's sidecar rename does not apply to them.

---------------------------------------------------------------------------------------------
## Phase 5 — quality check and pair selection (decide, then commit)

```bash
python scripts/model_quality.py --qrels sub1m/qrels.tsv --floor MiniLM \
  --model MiniLM=enc/all-MiniLM-L6-v2 --model bge-small=enc/bge-small-en-v1.5 \
  --model bge-base=enc/bge-base-en-v1.5 --model bge-m3=enc/bge-m3 --model e5-base=enc/e5-base-v2
python scripts/model_quality.py --qrels subcsn/qrels.tsv --floor MiniLM \
  --model MiniLM=enc_csn256/all-MiniLM-L6-v2 --model codesage-small=enc_csn256/codesage-small \
  --model codesage-base=enc_csn256/codesage-base
python scripts/model_quality.py --qrels subcsn/qrels.tsv \
  --model codesage-small=enc_csn512/codesage-small --model codesage-base=enc_csn512/codesage-base
```
Pass: no `<-- below floor model` flag; every row shows a contract status and padding `ok`.
A model below MiniLM-L6 is an encoding problem until proven otherwise: go back to Phase 2.
`model_quality.py` now refuses a model whose `q.npy` has no vector for some judged query
(`--allow-partial` only to look at it): scoring the remainder is how e5-base reported 0.000.
The floor is a **flag**, not a block — record the decision to drop or keep a pair in
`DECISIONS.md`. codesage-base at 0.785 against MiniLM's 0.957 means P8 (base → small) is an
"upgrade" only because base is broken: drop P8 until base passes Phase 4.3. *(Resolved
2026-11-06: with `add_eos_token=True` base self-retrieves at top-1 = 1.0; P8 was run after the fix.)*

Then choose the pairs: every pair goes from a worse to a better model **on its corpus**.
Text candidates: P2 MiniLM→bge-base, P3 bge-small→bge-base, P9 MiniLM→bge-small, plus pairs
with bge-m3 and e5-base in whichever direction the table supports (6–10 real pairs in total,
covering same-family and cross-family). Code: P7 MiniLM→best code model (256), P8
code→better code (512). P0 = bge-base identity (control). Record the quality tables and the
final pair list in `DECISIONS.md` and commit **before** Phase 6.

---------------------------------------------------------------------------------------------
## Phase 5.5 — the pre-migration sweep (THE GATE; CPU, ~1 hour for ten pairs)

Everything after this is chosen from its output. One measurement, four questions, no replay and
no re-encoding: `scripts/sweep.py` computes **rho(K'), G(K') and coverage(K') on one K' ladder**
from the vectors already on disk. Depths are nested prefixes of a single deep retrieval, so the
cost is the one retrieval per pair, not the ladder.

### 5.5a Pilot first, to measure the time
```bash
python scripts/sweep.py pairs/P2 --config configs/msmarco_1m_v3.json \
  --kprime 25,50,100,200,350,500,1000 --pilot 200 --out-dir sweeps_pilot
```
It subsamples 200 evaluation queries, marks the output `pilot: true`, prints per-stage timings
and extrapolates to a full pair and to ten. Expect ~4-6 min per pair (precompute dominates;
the measured v3 replays were 5.4-7.2 min each *including* precompute). Schedule the full run
from the number it prints, not from this estimate.

### 5.5b The ten pairs
Among the five encoded text models the valid upgrade pairs are the **population**, not a sample
(MiniLM 0.8245 < bge-m3 0.8468 < bge-small 0.8570 < bge-base 0.8668 < e5 0.8681), so run all of
them and no selection is needed - which is also what keeps LOPO free of a selection effect.
Five already exist; five need assembling, with **no new encoding**:
```bash
python scripts/assemble_pair.py --src enc/all-MiniLM-L6-v2 --tgt enc/bge-m3 --data sub1m --bm25 sub1m/bm25.npy --out pairs/P10 --meta '{"source":"sentence-transformers/all-MiniLM-L6-v2","target":"BAAI/bge-m3"}'
python scripts/assemble_pair.py --src enc/all-MiniLM-L6-v2 --tgt enc/e5-base-v2 --data sub1m --bm25 sub1m/bm25.npy --out pairs/P11 --meta '{"source":"sentence-transformers/all-MiniLM-L6-v2","target":"intfloat/e5-base-v2"}'
python scripts/assemble_pair.py --src enc/bge-m3 --tgt enc/e5-base-v2 --data sub1m --bm25 sub1m/bm25.npy --out pairs/P12 --meta '{"source":"BAAI/bge-m3","target":"intfloat/e5-base-v2"}'
python scripts/assemble_pair.py --src enc/bge-small-en-v1.5 --tgt enc/e5-base-v2 --data sub1m --bm25 sub1m/bm25.npy --out pairs/P13 --meta '{"source":"BAAI/bge-small-en-v1.5","target":"intfloat/e5-base-v2"}'
python scripts/assemble_pair.py --src enc/bge-base-en-v1.5 --tgt enc/e5-base-v2 --data sub1m --bm25 sub1m/bm25.npy --out pairs/P14 --meta '{"source":"BAAI/bge-base-en-v1.5","target":"intfloat/e5-base-v2"}'
```
`DESIGN.md` pre-registers **6-10 pairs** for Claim A; the 2026-11-01 run had five, so the
minimum was not met. Ten meets it. P14 (bge-base to e5) is a 0.0013 quality gap - keep it for
completeness but expect it to behave like the P0 control.

### 5.5c The full sweep
```bash
python scripts/sweep.py pairs/P2 pairs/P3 pairs/P4 pairs/P5 pairs/P9 pairs/P10 pairs/P11 pairs/P12 pairs/P13 pairs/P14 \
  --config configs/msmarco_1m_v3.json --kprime 25,50,100,200,350,500,1000 --mapped --out-dir sweeps
python scripts/sweep.py pairs/P7 --config configs/csn_py_v3.json --kprime 25,50,100,200,500,1000 --out-dir sweeps_csn
```

`sweep.py` and `coverage.py` are both needed and do different jobs. `sweep.py` works on an
assembled **pair** and gives rho, G and coverage together, which is what chooses the window.
`coverage.py` works on a single **encoding** with no pair at all, and writes the per-source
`coverage/<model>.json` that `predict.py` consumes in Phase 6.5 - so run it for every source
model as well:
```bash
for M in all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5 bge-m3 e5-base-v2; do
  python scripts/coverage.py --docs enc/$M/doc.npy --queries enc/$M/q.npy \
    --exclude-qids sub1m/query_sets/trecdl.txt --kprime 25,50,100,200,350,500,1000,2000 \
    --out coverage/$M.json --name $M
done
```
Its coverage at a given depth equals `sweep.py`'s for a pair with that source **when both are
given the same query set**, and they are computed independently, so a disagreement then is a bug
in one of them. The definitions differ in one way that matters: `sweep.py` takes the servable set
as `{q : workload weight > 0}`, which drops the secondary set *and* any zero-weight query, while
this script drops only what `--exclude-qids` lists. Under a uniform workload those coincide;
under a weighted one (ORCAS, the Zipf code sweep) **pass `--weights` here too**, or the two are
measuring different query sets and the disagreement is a definition gap rather than a bug.

### 5.5d What to read out of it

| question | where to look | what the 2026-11-01 run already knows |
|---|---|---|
| How large is the saving? | `ceiling_frac` per depth | 12-16% at K' = 500; coverage falls to 0.645 at 200 and 0.437 at 100, so 36% and 56% are *available* if quality survives |
| Does quality survive? | `G` and `admissible` | G <= 0 at K' = 500 on every pair. Below 500, **unmeasured** |
| Does the map's feature vary? | the cross-pair `rho spread` table | spread was 0.055 at K' = 500 and LOPO had nothing to predict (majority: accuracy 1.00, regret 0.0000) |
| Is the synthetic suite realistic? | `rho` at matched K'/N | synthetic runs at K'/N = 1.0% with rho 0.83-0.96; real pairs at 0.05% with rho 0.030-0.085, i.e. ~80x apart once normalised |

**G, not rho, is the quality price.** They decouple: the code pair had rho = 0.372 at K' = 50
(37% of the target's top-10 outside the window) for G = 0.011. rho is a loose upper bound on the
damage; G is the damage.

### 5.5e Choosing the window - and what it costs
Pick **the smallest K' whose G(K') <= eps on every pair**, from the cost/quality frontier only.

Report the chosen depth's coverage against `coverage_if_independent` from the same
`cov_*.json`, which is the closed form `1 - (1 - K'/N)^|Q|` and needs no retrieval at all. It
is the baseline for any cost claim, and it is not a weak one: on the 2026-11-04 run it
overstates cost by 10.6-15.5% at K' = 50 and 16.6-24.3% at K' = 200, so a planner using it
under-provisions the saving by that much. The measurement's whole contribution over the closed
form is the `concentration` column, so quote both or the bound has no baseline.

Do **not** pick the depth that maximises the cross-pair spread of rho. `features.rho` is read at
the window depth, so choosing the depth to give the map variance is feature-hacking - the same
error as choosing `primary_setting` because it yields a non-zero saving. Report the spread at the
chosen depth as a *result*: if it is flat, the finding is that the selector has nothing to select
at the deployment-optimal depth.

Changing `window_frac` changes `features.rho`, G, coverage, the ceiling, the per-query repair
costs and the meaning of `tau`, so it implies **re-running every pair**. Affordable (Phase 7.3)
but it is the whole experiment again. Record the depth, the frontier it came from, and the
re-run consequence in `DECISIONS.md` before Phase 6.5.

### 5.5f What the sweep decides
- **G admissible at a smaller K' and rho spreads there** - the map has structure and the saving
  is larger; set the window, re-run, Claim A is testable.
- **G admissible at a smaller K' but rho stays flat** - two separable results: a strong economics
  result and a *null* selector result. Both publishable; the contribution list has to be
  written for whichever it is and LOPO becomes an appendix. (This is what happened: see
  `paper/OUTLINE.md`.)
- **rho flat at every depth** - the finding is scoped: *"among the BERT-lineage models currently
  encoded, rho stays low down to K' = 25"*, **not** "real migrations never have high rho".
  Cross-family is the case that would move it, and it needs Qwen3 (Phase 0.0).

---------------------------------------------------------------------------------------------
## Phase 5.9 — SETTLE THE FRAMING (no compute; must be done before any re-run)

> **SETTLED 2026-11-04: B.** Not by judgement — Claim A's pre-registered falsification condition
> fired. `DESIGN.md` says A is falsified if the (kappa, rho) rule's mean regret is at least the
> majority baseline's, and on the twelve pairs it is: majority 9/12 at regret 0.00189, 1nn[both]
> 8/12 at 0.00201, tree2[both] 8/12 at 0.00287. The rules that match the baseline's accuracy
> reproduce its regret to every digit; rho alone is 6/12. See the 2026-11-04 `DECISIONS.md` row.
> The table below is kept as the record of what the choice was and what each side needed.
>
> So: LOPO is an appendix null result, the main table is the coverage/G frontier, and `paper/`
> needs writing for B -- `paper/OUTLINE.md` is that skeleton. Run `tau_curve.py`
> (Phase 10.2) before freezing the table — two of the twelve winners move with tau.

**Which paper was this?**

| | A — "the map predicts the winner" | B — "we derived the economics; selection is moot" |
|---|---|---|
| headline | kappa/rho predict which strategy wins | window reuse can defer at most 1 − coverage, at no quality cost **within the measured depth range** (G binds at K' = 20 on the NQ pair, so the range has a floor), and only realises it if the background is deferred — which is out of scope |
| main table | per-pair winners + LOPO regret | the coverage/G frontier + the deferral result |
| LOPO | Table 2 | appendix, reported as a null result |
| needs | rho to vary across pairs | nothing further; it is already measured |
| current evidence | five pairs all won S2/rand; majority baseline scored accuracy 1.00, regret 0.0000 | `S2/demand_then_rand` collected 0.8632/0.8790/0.8511/0.8345 of N against coverage 0.8632/0.8790/0.8511/0.8346 |

The sweep informs it but does not settle it: even with a 56% ceiling, if all ten pairs win the
same combination then A has nothing to predict. **B is already supported by measured data; A is
not yet.** Both are publishable. No draft existed either way -- `paper/OUTLINE.md`, written
2026-11-05, is the framing-B skeleton.

Decide, write it in `DECISIONS.md`, and only then run Phase 7 — because the choice determines
which pairs matter, whether the prediction files are a headline check or an appendix, and what
the re-runs are for.

**Out of scope this round, recorded so it is not mistaken for an oversight:** new-corpus inflow
(documents arriving already in the target space). It does not reduce the migration set — the old
corpus still costs N_old — and its value would be as a second *workload* axis, so it waits until
the sweep shows whether the map needs one. See the 2026-11-02 `DECISIONS.md` rows.

---------------------------------------------------------------------------------------------
## Phase 6 — smoke tests

### 6.1 Synthetic (no GPU, no data; ~1 min *measured*)
```bash
python scripts/make_synthetic.py --out /tmp/syn
python scripts/run_pair.py /tmp/syn/syn_identity --out /tmp/res/syn_identity.json --identity --allow-dirty
python scripts/run_pair.py /tmp/syn/syn_mid --out /tmp/res/syn_mid.json --allow-dirty
```
Pass: both exit 0 with only `ok` invariant lines. They register in `/tmp/res/RUNS.tsv`,
not in the repo's registry.

### 6.2 Assemble the text pairs
```bash
python scripts/assemble_pair.py --src enc/bge-base-en-v1.5 --tgt enc/bge-base-en-v1.5 --data sub1m --bm25 sub1m/bm25.npy --out pairs/P0 --meta '{"source":"BAAI/bge-base-en-v1.5","target":"BAAI/bge-base-en-v1.5","control":"identity"}'
python scripts/assemble_pair.py --src enc/all-MiniLM-L6-v2 --tgt enc/bge-base-en-v1.5 --data sub1m --bm25 sub1m/bm25.npy --out pairs/P2 --meta '{"source":"sentence-transformers/all-MiniLM-L6-v2","target":"BAAI/bge-base-en-v1.5"}'
python scripts/assemble_pair.py --src enc/bge-small-en-v1.5 --tgt enc/bge-base-en-v1.5 --data sub1m --bm25 sub1m/bm25.npy --out pairs/P3 --meta '{"source":"BAAI/bge-small-en-v1.5","target":"BAAI/bge-base-en-v1.5"}'
```
(Repeat for every pair chosen in Phase 5.)

### 6.3 Development run first (cheap; never paper numbers)
```bash
python scripts/run_pair.py pairs/P2 --config configs/msmarco_1m_v3.json --out results_dev/P2.json \
  --set n_eval_queries=500 --set 'traffic_ratio=[64.0]' --set 'primary_setting={"traffic_ratio":64.0}'
```
Pass: exit 0, `DEVELOPMENT RUN` banner, no `WARNING`. Note the wall time: it calibrates the
estimate in 7.3. `--set` runs go to `results_dev/`; `summarize.py` refuses them.
Use this run to measure the bootstrap spread of `debt` and set `tau.debt` from it (it is
currently a judgement; see `UNVERIFIED.md`).

### 6.4 Real smoke test: P0, P3, P2 (full pre-registered config)
Three pairs, not the ten from 5.5b: this exercises the full pipeline on real data, it is not the
map. P0 is the identity control (it is the only pair whose answer is known in advance, so a
surprise here is a bug, not a finding), P3 is same-family (bge-small → bge-base) and P2 is
cross-family (MiniLM → bge-base) — one of each, because they stress different code paths
(a shared tokenizer and dimension change vs neither). The other seven run in Phase 7.
```bash
python scripts/run_pair.py pairs/P0 --config configs/msmarco_1m_v3.json --out results/P0.json --identity
python scripts/run_pair.py pairs/P3 --config configs/msmarco_1m_v3.json --out results/P3.json
python scripts/run_pair.py pairs/P2 --config configs/msmarco_1m_v3.json --out results/P2.json
python scripts/ann_noise.py --doc enc/bge-base-en-v1.5/doc.npy --q enc/bge-base-en-v1.5/q.npy --kprime 500 1000
```
Pass:
- all exit 0. Exit 1 = an invariant failed = a pipeline bug: stop and report. Exit 2 =
  refused (uncommitted code): commit first.
- P0's ρ(K′) is exactly 0 (exact search). The ANN noise floor is `ann_noise.py`'s output,
  not P0's ρ.
- no `WARNING` line. `LEAKY` means `eval_frac` is 0. `SOURCE QUERY ENCODING SUSPECT` means the
  source encoding needs the Phase 2 checks before the pair is used.
- κ, ρ, winners: whatever they are, they are findings, not bugs.

Configuration errors are refused at start-up (fix the input, not the code):

| Message | Cause |
|---|---|
| `secondary_eval_qids=... does not exist` | `sub1m/query_sets/trecdl.txt` missing (Phase 3.1) |
| `... query indices outside [0, Q)` | secondary file from another corpus |
| `weights.npy exists but workload=...` | pair has workload weights; config must say `"workload": "file"` |
| `every evaluation query has weight 0` / `every traffic-pool query has weight 0` | weights file does not cover these queries |
| `tgt_depth=... < K` | config error |
| `no traffic queries left` | `eval_frac` too large for this query set |
| `primary_setting ... not in the grid` | `primary_setting` must be one of the grid points |
| `has no encoding contract` / `is blocked` | Phase 2 |
| `eval_arrival='first_service' requires mode='sync'` | an arriving query must have its window repaired inline to be servable at target |
| `no evaluation query can be served under this workload` | the weights give every evaluation query probability 0 |
| `refusing to assemble a pair with mismatched inputs` | `assemble_pair.py`: a `.npy` row count does not match the corpus/queries files (Phase 4.3) |

Two v3 warnings are about the **setting**, not the code, and both are predicted in Phase 6.5:

| Warning | Meaning |
|---|---|
| `ARRIVAL SHORTFALL` | this `traffic_ratio` cannot serve its evaluation population inside `max_rounds`. Draws are with replacement, so the requirement is a coupon-collector quantity, not `n_eval / stream_len`: raise `traffic_ratio` or `max_rounds_factor` |
| `NO WINNER at ratio=...` | less than `min_eval_served` of the evaluation **workload weight** was served, so the setting reports no winner instead of a misleading one. The gate is on weight, not query count, because every quality number is workload-weighted: under skew 99% of the weight can arrive by round ~100 while 99% of the count needs ~30,000 rounds, and a count gate would reject every ORCAS/Zipf setting. Both are reported |
| `N of M evaluation queries have draw probability 0` | a weighted workload (ORCAS, Zipf) gives some held-out queries weight 0, so they can never arrive. They are excluded from the measurement and the served fraction is reported against `n_expected`, not `n_eval`. Fix the weights file or accept the smaller evaluation set deliberately |

---------------------------------------------------------------------------------------------
## Phase 6.5 — mechanism predictions (commit them BEFORE the replay)

Most of what the replay reports follows from two static measurements — `G(K')` from
`measure.py` and `coverage(K')` from Phase 5.5. Write those consequences down, commit them,
then run the replay and check it against them.

Not all of them are predictions, and the difference matters when you report the check.
`predicted_C_serve_frac` is a **bound-attainment check**: a window strategy encodes the union
of the windows it serves by construction, so a match confirms the implementation and could
hardly come out otherwise. `predicted_debt_window ≈ G(K')` and `predicted_debt_S0` are genuine
predictions — they can be wrong, and the replay is the only thing that decides them. Say which
is which in the paper, or a reviewer will read the easy agreement as evidence for the hard
claim (see the 2026-11-06 `DECISIONS.md` row, which is exactly that mistake made once).

```bash
for P in P0 P2 P3 P4 P5 P9 P10 P11 P12 P13 P14; do
  python scripts/run_pair.py pairs/$P --config configs/msmarco_1m_v3.json --measure-only \
    --out results_measure/$P.json --name $P
done
# one per pair, each against ITS OWN source model's coverage file
for PS in "P2 all-MiniLM-L6-v2" "P3 bge-small-en-v1.5" "P4 bge-m3" "P5 bge-m3" "P9 all-MiniLM-L6-v2" \
          "P10 all-MiniLM-L6-v2" "P11 all-MiniLM-L6-v2" "P12 bge-m3" "P13 bge-small-en-v1.5" "P14 bge-base-en-v1.5"; do
  set -- $PS
  python scripts/predict.py --result results_measure/$1.json --coverage coverage/$2.json \
    --config configs/msmarco_1m_v3.json --out predictions/$1.json
done
```
`--measure-only` runs the measurements and skips the replay **under the pair's own config**. Use
it rather than borrowing a config that happens to set `replay: false` — `nq_100k_v2.json` would
have run an MS MARCO pair at the NQ window ladder (so `predict.py`'s `set(G) & set(coverage)`
would keep only {500, 1000} of the depths), with nDCG as the primary metric instead of Recall@10,
and with `eval_frac: 1.0` — which means the recorded `arrival` block would describe **every**
judged query while the replay holds out half, so the predictions would be for the wrong query
population. Same config, `replay` off, is the only way those three line up.

`predict.py` takes the per-evaluation-query draw probabilities from the result's `arrival`
block, which `run_pair.py` records. It does **not** reconstruct them, because it cannot: the
held-out set is a seeded permutation of the judged queries, not the first `n_eval` rows, and the
stream's distribution also zeroes the secondary set and renormalises. Under
`workload: "uniform"` every choice happens to give the same answer; under a weighted workload —
**ORCAS for RQ4, and the Zipf(α) code sweep** — it does not, so without that block `predict.py`
refuses rather than predict from the wrong queries' weights. For a result produced before this
version, pass `--weights` together with `--eval-qids`, or just re-run the measure-only pass.

Every pair in the main table needs a prediction file, or the Phase 10 check covers only part of
it. The two loops differ by one on purpose: the measure-only loop runs eleven pairs, the
prediction loop the ten that go in the main table. **P0 gets no prediction file** — it is the
identity control, its answer is known (ρ = 0, G = 0, no strategy to choose), and Phase 10 checks
predictions against the map, which P0 is not part of. Its measure-only result is still needed,
as the control row and as the ANN-floor reference. Commit `predictions/` with the
pre-registration commit. Each file states, per `traffic_ratio`
and depth:
- a window strategy is admissible iff `G(K') <= eps` → the optimal depth and which pairs have
  **no** admissible window strategy at all (dev `G` was 0.013 for P4, above `eps` = 0.01, so
  there S0 wins by definition — which is why `eps_sensitivity` exists);
- `ceiling_frac = 1 - coverage(K')`: the largest possible saving;
- `predicted_C_serve_frac` with a band on `f_bg` only (the union term has no randomness) —
  a bound-attainment check, not a prediction: see above;
- `predicted_debt_S0` and `predicted_debt_window ≈ G(K')`;
- `predicted_per_query_cost_at_f ≈ (1 - f) K'`.

Then after Phase 7, check each one. Agreement validates the implementation. Disagreement inside
the band is the arrival schedule's randomness; **outside the band it is a bug, not a finding** —
stop and fix the code rather than writing up the discrepancy.

---------------------------------------------------------------------------------------------
## Phase 7 — full text map

### 7.1 Run every chosen text pair
```bash
for P in P2 P3 P4 P5 P9 P10 P11 P12 P13 P14; do
  python scripts/run_pair.py pairs/$P --config configs/msmarco_1m_v3.json --out results/$P.json
done
```
Re-running the loop is safe: pairs already in `results/RUNS.tsv` are skipped.

This is **all ten** pairs chosen in Phase 5.5b, and the whole grid in one invocation per pair -
not one ratio per invocation. The 2026-11-01 run sharded by ratio with `--set`, which marked
every result a development run and made `summarize.py` refuse all 21 of them; a full-grid
invocation is also *cheaper*, because `precompute` runs once instead of once per ratio. P0 is
run separately in Phase 6.4 as the identity control, and there is no pair called P1.

### 7.2 What one pair computes
Strategy/order combinations per setting: S0 and S1 run once (random order); S2–S5 run with
`rand`, `size`, `demand`, `demand_then_rand`, plus `signal` only if the pair's heat gate opens.
S4/S5 need `bm25.npy`. Counting them:

| | gate shut | gate open |
|---|---|---|
| S0 + S1 (1 order each) | 2 | 2 |
| S2 + S3 (4 orders, +`signal`) | 8 | 10 |
| S4 + S5 (same, need BM25) | 8 | 10 |
| **total with BM25** | **18** | **22** |
| **total without BM25** | **10** | **12** |

Every combination runs at every `traffic_ratio`: 4 in the v3 draft. `repeat_rate` is not
a grid axis in v3 — the mix of evaluation and other queries is the workload's own.

Each combination ranks each evaluation query **once**, at its arrival, instead of re-ranking
every evaluation query at every round, and the arrival sequence is identical across all
combinations (`smoke.py` checks it). The result also carries, per combination: `debt` (the
objective), `C_serve` (diagnostic), `repairs_per_query_mean/_p95`, a read-only per-query cost
curve (~200 points), and `eps_sensitivity` for the cost diagnostic.

### 7.3 Time
v2's dominant term was eval queries × checkpoints (~32 µs per query per checkpoint, measured at
N = 200k): at 1M that is ~0.56 s per checkpoint, ~5 hours per pair over 200 rounds × 16
combinations × 10 settings. v3 removes the `× checkpoints` factor from the ranking work — one
ranking per evaluation query per combination — so the per-round window set operations and the
exact search dominate instead. Re-time it on the 6.3 development run before planning the full
sweep; `UNVERIFIED.md` records that the old estimate no longer applies. For debugging, shrink
the grid with `--set` or use `n_eval_queries`; that changes the run key and is not paper data.

---------------------------------------------------------------------------------------------
## Phase 8 — code map

```bash
python scripts/assemble_pair.py --src enc_csn256/all-MiniLM-L6-v2 --tgt enc_csn256/codesage-small --data subcsn --bm25 subcsn/bm25.npy --out pairs/P7 --meta '{"source":"sentence-transformers/all-MiniLM-L6-v2","target":"codesage/codesage-small","max_seq_len":256}'
python scripts/assemble_pair.py --src enc_csn512/codesage-small --tgt enc_csn512/codesage-base --data subcsn --bm25 subcsn/bm25.npy --out pairs/P8 --meta '{"source":"codesage/codesage-small","target":"codesage/codesage-base","max_seq_len":512}'
python scripts/run_pair.py pairs/P7 --config configs/csn_py_v3.json --out results/P7.json
python scripts/run_pair.py pairs/P8 --config configs/csn_py_v3.json --out results/P8.json
```
(Use the targets Phase 5 actually chose; the lines above assume codesage-small → codesage-base.)
Pass checks as in 6.4. CoSQA is the evaluation-only secondary set.

**What these two pairs can and cannot support.** `configs/csn_py_v3.json` records that coverage
on this corpus is 1.0000 at every depth from 23 to 1000 — 43,730 queries over 45,283 documents,
so the union of the windows is the corpus and the ceiling `1 - coverage` is exactly 0. No window
strategy saves anything here, whatever the map says. So P7 and P8 are **not** cross-domain
evidence for Claim A: they cannot show different pairs taking different strategies, because on
this corpus only one strategy is ever economic. They are a finding in their own right, and a
sharp one — the window family needs `queries x K'` to be small against `N`, and a corpus this
small against its query set puts it outside that regime. Write them up that way; the
cross-domain evidence for Claim A has to come from Phase 0.0 or Phase 9.

---------------------------------------------------------------------------------------------
## Phase 9 — NQ comparison against EmbedFlow (measure-only)

Qwen3-8B stays `blocked` until Phase 2 resolves it. MiniLM needs length 512 here to match the
registry record, which is above its contract cap; pass the override explicitly so the sidecar
records it:
```bash
python scripts/encode.py --model sentence-transformers/all-MiniLM-L6-v2 --kind doc --tsv subnq/corpus.tsv --out enc_nq/all-MiniLM-L6-v2/doc.npy --max-seq-len 512 --override '{"max_seq_len_cap": 512}'
python scripts/encode.py --model sentence-transformers/all-MiniLM-L6-v2 --kind query --tsv subnq/queries.tsv --out enc_nq/all-MiniLM-L6-v2/q.npy --max-seq-len 512 --override '{"max_seq_len_cap": 512}'
python scripts/encode.py --model Qwen/Qwen3-Embedding-8B --kind doc --tsv subnq/corpus.tsv --out enc_nq/Qwen3-Embedding-8B/doc.npy --max-seq-len 8192 --batch 16
python scripts/encode.py --model Qwen/Qwen3-Embedding-8B --kind query --tsv subnq/queries.tsv --out enc_nq/Qwen3-Embedding-8B/q.npy --max-seq-len 8192 --batch 16
python scripts/assemble_pair.py --src enc_nq/all-MiniLM-L6-v2 --tgt enc_nq/Qwen3-Embedding-8B --data subnq --out pairs/P5c --meta '{"source":"sentence-transformers/all-MiniLM-L6-v2","target":"Qwen/Qwen3-Embedding-8B","registry":"embedflow nq_minilm_100000_qwen3_8b"}'
python scripts/run_pair.py pairs/P5c --config configs/nq_100k_v2.json --out results/P5c.json --name P5c
```
For a diagnostic before Qwen3-8B is unblocked, add `--allow-blocked` to its two encode
commands (recorded in the sidecar; not paper data).
Pass criterion (pre-registered in the config): M_T near the registry's 0.8800 and M_S near
0.7458 first. If either is far off, stop: the corpus or the encoding does not match, and G
is not comparable. Only then compare G(50) with 0.027784 ± 0.01.

---------------------------------------------------------------------------------------------
## Phase 10 — results

```bash
python scripts/run_lopo.py results/*.json --exclude P0 P6 --rho-sweeps sweeps_fine sweeps_csn --out results/lopo.json
python scripts/summarize.py results/*.json --exclude P0 P6 --csv results/tables
```
- `run_lopo.py`: RQ2. It uses the winner at `primary_setting`; measure-only runs are skipped.
  `--rho-sweeps` is required for the K'=50 text results: their own rho ladder starts at 500, so
  their stored `features.rho` is rho(500); the flag substitutes rho at the replay window from
  `sweeps_fine/` and refuses if a sweep lacks it. Without it the script warns per pair.
  For v3 results pass the objective so regret is measured in the objective's units, and add
  `coverage` to the feature sets only if Phase 5.5 showed a spread.
- `summarize.py`: tables `map` (Table 1 / Figure 1), `grid` (winner sensitivity), `probe`
  (RQ3; judge H4 on `total_frac`, not `sample_frac`), `heat` (RQ4), `calib` (Phase 9).
  It refuses leaky, development and uncommitted-code results.
Pass: no `REFUSED` lines for the paper's pairs.

Read the `map` and `grid` tables with the three cost columns together, never `C_serve_frac`
alone. `C_serve_basis` says which of three situations produced that number — work the strategy
did (`at_last_arrival`), how far its background happened to get (`no_repair_needed`, which
shrinks as `traffic_ratio` rises while the strategy is unchanged), or N charged by convention
(`never_reached_target`) — and `C_total_frac` is 1.0000 for every admissible row, because
everything that completes encodes exactly N. `C_serve` is not a saving. A
`COST TIE-BREAK ACROSS BASES` warning means the band was ordered across two of those bases;
the ordering is pre-registered and stays, so report the winner with its basis beside it.
`tied_exact` is how many combinations share the winner's objective *exactly*, as against
`band`, which is how many are within tau: at the primary setting (ratio 64) eight pairs
reported one of 3–4 identically-scoring choices and P0 one of 14 spanning all six families.
Print `tied_exact`, or the winner column overstates how determined it was. Note the ties are a
property of that setting and not of the method — at ratio 4 and 16 the background order does
move `debt`, because window strategies also search the target-side index over migrated
documents and at low traffic intensity the background has migrated a lot by service time.

### 10.2 The tau curve (no compute; reads the result JSONs)
```bash
python scripts/tau_curve.py results/P*.json --csv results/tables
```
`tau` is a pre-registered judgement and still an `UNVERIFIED.md` row, and it is load-bearing for
exactly the rows that make the map look non-trivial: on the 2026-11-04 run P14 moves to the
window family at `tau <= 0.004` and P11 moves to it at `tau >= 0.006`, so *which* pair is the
exception is a function of tau rather than of the pair. Pass: the winner at the config's own tau
equals the winner in each result (the script runs the same `pick_winner_v3`, swept). Then either
set `tau.debt` from the bootstrap spread in Phase 6.3 and say so, or publish this curve beside
the main table. Do **not** retune tau to change a winner — see the 2026-11-04 `DECISIONS.md` row.

### 10.3 Predictions
Check Phase 6.5's predictions against the results, pair by pair, and record each check.
This is the step that makes the protocol revision a predictive test rather than a re-tuning.

---------------------------------------------------------------------------------------------
## Phase 11 — the last tests before the table freezes (framing B)

Read this with `PAPER_NUMBERS.md` open: the v3 1M run is done and most of it is quotable, so
what is left is not "run the study again" but closing the three things that block a table. In
order, because 11.0 can change what the paper claims and 11.1 can change 11.2's inputs.

### 11.0 Is G < 0 real, or is it judgment sparsity? (**done 2026-11-07**, commit `fbb23c9`)
```bash
for P in P2 P3 P4 P5 P9 P10 P11 P12 P13 P14; do
  python scripts/run_pair.py pairs/$P --config configs/msmarco_1m_v3.json \
      --out results_measure/$P.json --measure-only --force
done
```
This is the one experiment that can invalidate the paper's most interesting number, and it
needs no re-encoding: the corpus, the encodings, the TREC-DL qrels and
`sub1m/query_sets/trecdl.txt` all already exist. The run now emits a `secondary_frontier`
block — rho(K') and G(K') on the deeply-judged secondary set, uniformly weighted, against an
M_T measured on those same queries.

**Why it decides something.** The primary metric is Recall@10 against MS MARCO dev qrels,
about one judged passage per query, pooled from older systems. A window strategy is confined
to the source index's top-K', so it returns documents closer to that pool; exhaustive target
retrieval surfaces unjudged documents that count as irrelevant by construction. That mechanism
alone predicts G < 0 — and the 1M run reports G(50) < 0 on **9 of 10** text pairs, on P5 at
-0.0099 against an upgrade gap of +0.0100, i.e. the window recovering 99% of the entire
benefit of the migration. Until this is run, "the window beats exhaustive target retrieval"
and "judgment sparsity flatters the window" are indistinguishable, and a reviewer will say so.

Read it as:
- **G < 0 survives on TREC-DL** → the cascade effect is real, it is the paper's most
  interesting finding, and it deserves a section of its own rather than a line in a table.
- **G >= 0 on TREC-DL while negative on dev** → the primary G is an artefact. The runner
  prints `NOTE G < 0 ... consistent with judgment sparsity` when it sees exactly that. Then
  Section 5's cost model is costing an artefact, Section 7's ordering is moot, and the paper
  has to be rebuilt around the mapping result. Better to find that here than in review.
- **Both negative but much smaller on TREC-DL** → report both and let the reader weigh them.

Pass: every pair has a non-empty `secondary_frontier` with the same depths as its primary
ladder. Then record the verdict in `DECISIONS.md` before touching anything else — 11.1's
window choice and the paper's framing both depend on it.

### 11.1 Finish the window frontier — K′ = 10 and 25 (**done 2026-11-07**; outputs in `sweeps_fine/`)
```bash
for P in P2 P3 P4 P5 P9 P10 P11 P12 P13 P14; do
  python scripts/sweep.py pairs/$P --config configs/msmarco_1m_v3.json \
      --kprime 10,25,50,100,200,500 --out-dir sweeps_fine
done
python scripts/coverage.py --docs enc/all-MiniLM-L6-v2/doc.npy --queries enc/all-MiniLM-L6-v2/q.npy \
    --kprime 10,25,50 --out coverage/MiniLM_low.json
```
No re-encoding: these are nested prefixes of retrieval already done. No `--weights`: nothing in this repo generates `sub1m/weights.npy` (it is an ORCAS artifact you supply, and `assemble_pair.py --weights` links it in), and the reported runs are uniform. If you do have one, pass it here AND in the pair config, or the coverage number will not match the replay's. This is the **only
outstanding item that moves a headline number.** G has not bound anywhere on the measured
ladder (worst +0.0035 at K′ = 50, ≤ +0.0000 above it), so the paper currently cannot say it
chose the smallest admissible depth — 50 is merely the smallest it looked at, and the 73.5%
ceiling is a lower bound set by where the ladder stopped.

Read it off the frontier only, as Phase 5.5e says — never off a replay you have seen.
- **G admissible at 10 or 25** → that depth is the window. Coverage drops to roughly 0.10–0.15,
  the ceiling rises to 85–90%, and the main table's number changes. Re-run Phase 7 at the new
  `window_frac`; the pairs' measurements stand but the replays do not.
- **G binds first at 50** → 50 was the smallest admissible depth after all. Record that in
  `DECISIONS.md` with the measured G at 10 and 25, and the existing replays stand unchanged.

Pass: `kstar` reported for all ten pairs, and the chosen depth justified by a frontier row
rather than by having been the ladder's floor.

### 11.2 Commit predictions, then check them (**superseded by 13.2**; see Status 2026-11-08)
`predictions/` was never committed for the ten pairs, so Phase 10.3 has nothing to check
against and the protocol revision is a re-tuning rather than a predictive test. Do this *after*
11.1 settles the depth, and **commit before looking at any replay**:
```bash
# the pair -> source-model mapping is Phase 6.5's; predict.py needs each pair's OWN source
# coverage file, and both --result and --coverage are required
for PS in "P2 all-MiniLM-L6-v2" "P3 bge-small-en-v1.5" "P4 bge-m3" "P5 bge-m3" \
          "P9 all-MiniLM-L6-v2" "P10 all-MiniLM-L6-v2" "P11 all-MiniLM-L6-v2" "P12 bge-m3" \
          "P13 bge-small-en-v1.5" "P14 bge-base-en-v1.5"; do
  set -- $PS
  python scripts/run_pair.py pairs/$1 --config configs/msmarco_1m_v3.json \
      --out results_measure/$1.json --measure-only
  python scripts/predict.py --result results_measure/$1.json --coverage coverage/$2.json \
      --config configs/msmarco_1m_v3.json --out predictions/$1.json
done
git add predictions && git commit -m "Pre-registered predictions for the v3 table"
```
If 11.1 changed `window_frac`, the `--measure-only` results above **must** be re-run rather
than reused from Phase 6.5: they carry the window depth and the K' ladder the replay will use,
and `predict.py` joins the two by depth.
Then Phase 10.3. Pass: every pair has a prediction file and a recorded check. A prediction
**outside** the band is a bug to find, not a finding to report (`predict.py`'s C_serve formula
is exact at traffic_ratio 4/16/64 and ~0.010 off at 256 — see the 2026-11-03 `DECISIONS.md`
row — so a miss larger than that is the pipeline, not the model).

### 11.3 The τ curve and the provenance gap (**done**; see Status 2026-11-08)
```bash
python scripts/tau_curve.py results/P*.json --csv results/tables
```
Phase 10.2 has the detail. τ is load-bearing for exactly the rows that make the map look
non-trivial, so publish the curve beside the main table or set `tau.debt` from Phase 6.3's
bootstrap and say so. Then close Phase 12 item 2: the twelve pairs span four commits and no
bundle was kept, so either show that the later commits touch no measured code path, or re-run
the odd pairs on one commit.

### 11.4 Housekeeping that would otherwise reach print
- Re-run Phase 4.3's `encode_check.py` and the Phase 5 quality table: the shipped
  `quality_5models.txt` pre-dates the 4.2 sidecar rename and prints `no sidecar` for four of
  five models, so the verification column is unknown.
- Retitle or delete `CROSS_FAMILY.md`. It is titled BERT→Qwen3-8B but its P5 row is
  bge-m3→bge-small; there is no cross-family measurement in this run.
- Footnote P11's κ = −0.0928 (residual > 1) wherever κ appears as "compatibility".

### Not blocking, and deliberately so
- **Phase 0.0 cross-family (Qwen3).** Under B it is external validity, not the headline, and
  `DECISIONS.md` records why it will not revive A: the map needs M_A ≥ M_T − τ ≈ 0.870, the best
  measured is 0.757 at κ = 0.441, and cross-family κ runs *lower* than BERT→BERT. Pre-register
  that prediction before running it and it becomes a confirmed prediction rather than a null.
- **Phase 9 NQ comparison.** DESIGN.md calls it the design's only external validation point;
  worth having, as a comparison and labelled as one, not a calibration.
- **More LOPO feature variants.** With 9/12 on one label and the window variants tied in
  `debt` at the primary setting, 9/12 is the achievable ceiling. Adding features cannot beat a constant that is
  already at the ceiling, so a null here is arithmetic, not evidence.

---------------------------------------------------------------------------------------------
## Phase 12 — freeze for the paper

1. `results/RUNS.tsv`: every paper run has status exactly `ok` — **not** `ok_dirty`, which is
   what a run from uncommitted code now registers as. The dirty marker is also part of the run
   key (it carries a hash of the diff), so a scratch run no longer shares a key with the clean
   run of the same commit and no longer makes the clean rerun look "already run".
2. All of them come from the same commit as the pre-registration, or a later one that changes no
   measured code path.
3. `git tag paper-v3-table1` on the commit recorded in the results' `provenance.git`.
4. Every number in the paper comes from `results/tables/*.csv` or has a `verified` row in
   `UNVERIFIED.md`. The effect **ceiling** (`1 - coverage`) and the predicted values come from
   `coverage/` and `predictions/`; release both.
5. Artifact: release code, configs, contracts, golden files and result JSONs. Do not release
   vectors derived from MS MARCO or ORCAS (non-commercial terms; see DESIGN.md).
6. Automatic checks (both must pass; the driver's `check` stage runs the second):
```bash
python scripts/freeze_check.py results/P2.json results/P3.json results/P4.json results/P5.json results/P9.json results/P10.json results/P11.json results/P12.json results/P13.json results/P14.json sweeps_fine/sweep_P*.json --out FREEZE_CHECK.md
python scripts/check_tables.py results/tables results/P*.json
```
   Pass: `K* = 50, bound by P11` (the 11.1 sweeps carry the 10/25/50 rungs; without them every
   shallow cell reads "not measured" and K* is undecided), and `check_tables.py` exits 0 with
   "all 6 tables describe these results" and no `STALE` or `unreadable` line (since 2026-11-10 an
   unreadable result file also makes it exit 1).

## Phase 13 — submission experiments (ICDE 2027 Round 2, EAB category, due 2026-11-11 17:00 PT)

Run on the instance that holds `enc/` and `pairs/` for the 2026-11-07 round (224 cores, 1 TB
RAM). Every step is CPU only; the encodings exist and the GPU is not used, except by 13.10's
encoding, which Phase 14 runs. The normal route is one command (13.11). Sections 13.1-13.10 say what each
stage does, the manual command for it, and what passing looks like; use them to read the
output or to re-run one stage.

**Order.** 13.3 (P0) runs first and alone, so the plan is computed from the final P0. Then
13.2a computes the plan and **commits it**. Then 13.1, 13.2b, 13.5, 13.6, 13.7, 13.9 and (if
its encodings exist) 13.10 run in parallel; 13.2b's replays must start after the plan's commit. Then 13.8 checks, then the zip.

**Time** (from the instance's measured runs: K'=50 replays 34-73 min each, P0 9 min, one
coverage pass ~157 s per encoder): about 1.5-2.5 h wall-clock in all, bound by the eight
13.2b replays and the two 13.9 replays (4096-dimensional targets). CPU time at most ~250
core-hours; GPU time zero here (13.10's encoding, under an hour, runs in Phase 14).

### 13.0 Install the code and check the machine (~1 min)

Unzip `reusemap_submission_changes.zip` over the checkout at `57de2da` exactly as its
`APPLY.md` says, and **commit** (results from uncommitted code register as `ok_dirty`, and the
driver refuses to start). Then:

```bash
python tests/test_invariants.py          # pass: "81 tests passed"
python tests/test_runbook.py
bash scripts/submission_runs.sh preflight
```

Pass: `preflight ok`. Preflight refuses, naming the cause, if: the branch is not
`submission-eab`; any tracked file differs from HEAD (`run_pair.py` would refuse every
replay); the test count is not 81; an input is missing (`enc/<model>/{doc,q}.npy` for the four
text sources, `enc_csn256/all-MiniLM-L6-v2`, `enc_csn512/codesage-small`, the two exclusion
lists, `pairs/`, `results/`, `coverage/`, `sweeps_fine/`, `sweeps_csn/`); or
`configs/msmarco_1m_v3_d3.json` differs from the registered config in anything but the K'=25
window.

### 13.1 Every Drift-Adapter family, every pair (~20-30 min per text pair; pairs run in parallel) — re-run needed

The replayed bridge is semi-orthogonal Procrustes on 10,000 document pairs. Drift-Adapter
evaluates Procrustes, low-rank affine and a residual MLP and reports 95-99% of re-embedding
recall. Measure all of them (plus ridge affine; low-rank at rank 64 and 128), each fitted on
document, query and combined pairs, before claiming anything about bridging. Each fitted
adapter is scored by one full-corpus retrieval, so a text pair is ~15 retrievals; the driver
runs one process per pair.

```bash
python scripts/adapter_families.py pairs/P2 --config configs/msmarco_1m_v3.json --out-dir adapters
python scripts/adapter_families.py pairs/P7 --config configs/csn_py_v3.json --out-dir adapters
```
(the first line for each text pair P2 P3 P4 P5 P9 P10 P11 P12 P13 P14 and for the encoder->LLM
pairs P5p P5q, the second for P7 P8.)

Affine and low-rank use a ridge chosen on held-out pairs (`compat.fit_affine`, ridge=None);
each file records `"affine_ridge": "held-out"` and the `ridge` of every affine/low-rank fit. A
file without that marker is from the 2026-11-09 run, whose fixed ridge collapsed the map; the
paper does not read its affine/low-rank entries. The driver moves earlier files to
`adapters_superseded/` before fitting; `ADAPTER_JOBS=<n>` fits n pairs at a time (each holds two
1M-vector sets; twelve at once was killed for memory in a 90 GB container).

Pass: 14 files `adapters/adapters_P*.json`, each with `"affine_ridge": "held-out"`. Read
`any_beats_source` in each. If any family
beats the source model on a pair, D2 admits that bridge on that pair and the paper reports
where and by how much; if none does, the result holds across all of Drift-Adapter's families
and three training sets. Report it either way.

### 13.2 Validate D3's priced decision at K' = 25 — commit first (~45-90 min) — done 2026-11-09

13.2a, the plan (seconds), after 13.3:
```bash
python scripts/merge_frontier.py sweeps_fine/sweep_P*.json sweeps_csn/sweep_P*.json --out planner/frontier_all.json
mkdir -p planner/cov && for f in coverage/*.json; do b=$(basename $f); cp $f planner/cov/cov_${b#cov_}; done
python scripts/plan.py results/P*.json --coverage-dir planner/cov --frontier planner/frontier_all.json --csv planner
git add -f planner/plan.csv planner/frontier_all.json && git commit -m "D3 plan, committed before the K'=25 replays"
```
Pass before committing: `planner/plan.csv` has P2 P3 P4 P5 P9 P10 P12 P13 at K'=25, P11 at
50, P0 P8 P14 rebuild, and every window plan priced (the driver checks this and refuses to
commit anything else). Do NOT pass `sweeps_fine/frontier.json` to plan.py: every sweep run
overwrites it, so it holds only the last pair; `merge_frontier.py` also refuses a pair in two
files, or one depth twice in a file. **If the plan differs, stop and report it**; do not edit
the plan to make it match.

13.2b, the eight replays (in parallel; ~45-90 min):
```bash
python scripts/run_pair.py pairs/P2 --config configs/msmarco_1m_v3_d3.json --name P2 --out results_d3/P2.json
```
(for each of P2 P3 P4 P5 P9 P10 P12 P13). Then score them against the committed plan:
```bash
PLAN_AT=$(git log -1 --format=%cI -- planner/plan.csv)   # the committed plan's time
python scripts/check_planner.py --plan planner/plan.csv --results results_d3 --committed-at "$PLAN_AT" --runs results_d3/RUNS.tsv --runs-tz "$(date +%:z)" --md planner_check.md
```
Pass: `planner_check.md` lists the eight pairs as `pre-committed` and the summary reads
`8 pairs scorable, 8 pre-committed`. Whatever the verdicts (PASS: the replay attains cov(25)
and debt within tolerance of G(25)), report them as they are. A FAIL is the planner's failure
mode and goes in the paper; **never re-plan after seeing the replays** (`check_planner.py`
would mark a re-plan post-hoc).

### 13.3 P0 under the final configuration (~5-10 min; first, alone) — done 2026-11-09

```bash
python scripts/run_pair.py pairs/P0 --config configs/msmarco_1m_v3.json --name P0 --identity --out results/P0.json --force
```
Keep a copy of the old file first (`superseded/P0_window500.json`; the driver does this and
restores it if the re-run fails). Pass: `results/P0.json` has `"window": 50`.

### 13.4 What comes back

The driver writes `submission_outputs_<date>.zip` with everything below; send that file. By
hand, collect: `planner/`, `planner_check.md`, `results_d3/`, `adapters/`, every
`results/P*.json`, `results/RUNS.tsv`, `results/tables/`, `superseded/`, `window_mass/`,
`window_mass_csn/`, `curve_check.md`, `curve_check_csn.md`, `results_measure_d3/`,
`sample_sweep/`, `results_xarch/`, `FREEZE_CHECK.md` and `logs/` (and `results_llm/`,
`coverage_llm/`, `adapters/adapters_P5r.json` if 13.10 ran). The paper is then rebuilt from the data
archive with `python -I paper/build/make_numbers.py --data PATH/reusemap_paper_data_2026-11-07
--repo . --out paper` (the build script is in the paper source, not this repo).

### 13.5 The cost model's statistic: window multiplicities (~5-15 min) — done 2026-11-09

`scripts/window_mass.py` makes the same retrieval pass as `coverage.py`, but keeps how many
servable queries touch each document. From that histogram, `reusemap.coverage_model` gives the
exact expected cost-at-service after any number of i.i.d. arrivals, the bound it can never
exceed, and the same curve under Zipf popularity. `coverage_curve_check.py` then scores the
model against every round of every replay. Nothing is fitted.

```bash
for M in all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5 bge-m3; do
  python scripts/window_mass.py --docs enc/$M/doc.npy --queries enc/$M/q.npy --exclude-qids sub1m/query_sets/trecdl.txt --kprime 25,50,100,200,500 --zipf 0.6,0.9,1.2 --sample-fracs 0.1,0.25,0.5 --name $M --out window_mass/$M.json
done
# code sources at the encodings their pairs used (pairs/P7/meta.json: 256 tokens; pairs/P8: 512)
python scripts/window_mass.py --docs enc_csn256/all-MiniLM-L6-v2/doc.npy --queries enc_csn256/all-MiniLM-L6-v2/q.npy --exclude-qids subcsn/query_sets/cosqa.txt --kprime 25,50,100,200 --name all-MiniLM-L6-v2 --out window_mass_csn/all-MiniLM-L6-v2.json
python scripts/window_mass.py --docs enc_csn512/codesage-small/doc.npy --queries enc_csn512/codesage-small/q.npy --exclude-qids subcsn/query_sets/cosqa.txt --kprime 25,50,100,200 --name codesage-small --out window_mass_csn/codesage-small.json
python scripts/coverage_curve_check.py --window-mass window_mass --results results --md curve_check.md
python scripts/coverage_curve_check.py --window-mass window_mass_csn --results results --md curve_check_csn.md
```

Pass: each text `window_mass.py` run should print `excluded 97 of 7077 queries ...; 6980 remain`
(7,077 judged queries, 97 of them TREC-DL; 6,980 is the replayed set). Any other count means
the exclusion list does not match `q.npy`: stop and check before reading the curve check. `curve_check.md` reports a worst relative error of a few percent at most,
and zero points above the independent bound. Points reported separately as "at full coverage"
(realised coverage 1.0 where the bound is within 1e-9 of 1; on the 2026-11-07 data, 92 points
of the code pair P8, at most 2.3e-10 over) are saturation, not violations: the bound is on
EXPECTED coverage. Series marked "r0 ambiguous" (more than one start round fits the thinned
trajectory) are scored at their worst case. Rows that say "window mass is for N=..." are a text
encoder's file meeting a code pair (or the reverse) and are expected.

### 13.6 How small can the planner's sample be? (~25-40 min) — done 2026-11-09

D1 decides from M_T - M_S. In deployment only a sample of the corpus is target-encoded, so
recompute the gap on judged documents plus 1%, 5% and 25% of the rest, three seeds each, and
report the smallest fraction at which D1 decides as on the full corpus. D3's input G(K') is not
estimable from such a sample (a K'-deep window in N has no counterpart in 1% of N); the paper
says so.

```bash
python scripts/sample_sweep.py pairs/P2 pairs/P3 pairs/P4 pairs/P5 pairs/P9 pairs/P10 pairs/P11 pairs/P12 pairs/P13 pairs/P14 --config configs/msmarco_1m_v3.json --results results --fracs 0.01,0.05,0.25 --seeds 3 --out-dir sample_sweep
python scripts/sample_sweep.py pairs/P7 pairs/P8 --config configs/csn_py_v3.json --results results --fracs 0.01,0.05,0.25 --seeds 3 --out-dir sample_sweep
```

Pass: 12 files `sample_sweep/P*.json`, each with `smallest_frac_all_seeds_agree`.

### 13.7 TREC-DL G at K' = 25 and 50 for P11 and P14 (~20-40 min) — done 2026-11-09

The eight 13.2b replays carry TREC-DL G at K'=25/50 in their `secondary_frontier`; P11 and P14
are not replayed at 25, so measure them (measure-only: no replay):

```bash
python scripts/run_pair.py pairs/P11 --config configs/msmarco_1m_v3_d3.json --name P11 --measure-only --out results_measure_d3/P11.json
python scripts/run_pair.py pairs/P14 --config configs/msmarco_1m_v3_d3.json --name P14 --measure-only --out results_measure_d3/P14.json
```

Pass: both files exist and carry `secondary_frontier` with K' 25 and 50.

### 13.8 Checks (~2-5 min, after everything)

```bash
python scripts/summarize.py results/P*.json --exclude P0 P6 --csv results/tables
python scripts/tau_curve.py results/P*.json --csv results/tables
python scripts/check_tables.py results/tables results/P*.json
python scripts/freeze_check.py results/P2.json results/P3.json results/P4.json results/P5.json results/P9.json results/P10.json results/P11.json results/P12.json results/P13.json results/P14.json sweeps_fine/sweep_P*.json --out FREEZE_CHECK.md
```
plus `check_planner.py` (13.2) and `coverage_curve_check.py` (13.5). Pass: "all 6 tables
describe these results"; `K* = 50`, bound by P11.

### 13.9 Encoder->LLM replays (`xarch`, on by default; ~1-2 h, alongside d3)

P5p (MiniLM -> Qwen3-Embedding-8B) and P5q (bge-base -> Qwen3-8B) have static measurements
only; the paper lists their replays as outstanding. The target encodings exist, so this is CPU
only:

```bash
python scripts/run_pair.py pairs/P5p --config configs/msmarco_1m_v3.json --name P5p --out results_xarch/P5p.json
python scripts/run_pair.py pairs/P5q --config configs/msmarco_1m_v3.json --name P5q --out results_xarch/P5q.json
```

Pass: both files exist with no invariant failures. `XARCH=0` leaves them out of `all`.

### 13.10 One LLM->LLM pair, P5r = Qwen3-Embedding-0.6B -> Qwen3-Embedding-8B

Every replayed pair has a BERT-lineage source. A referee can argue that two LLM embedders share
enough pre-training to align, so a bridge might clear the old model there. P5r tests it with
the smallest change: the 8B target vectors are the ones P5p/P5q use, so only the 0.6B source
needs encoding (GPU; the contract in `configs/models.json` follows the model card and matches
the 8B's). Encode at the 512 tokens the 8B encodings used:

```bash
python scripts/encode.py --model Qwen/Qwen3-Embedding-0.6B --kind doc --tsv sub1m/corpus.tsv --out enc/Qwen3-Embedding-0.6B/doc.npy --max-seq-len 512 --batch 128
python scripts/encode.py --model Qwen/Qwen3-Embedding-0.6B --kind query --tsv sub1m/queries.tsv --out enc/Qwen3-Embedding-0.6B/q.npy --max-seq-len 512 --batch 128
python scripts/encode_check.py --enc enc/Qwen3-Embedding-0.6B --data sub1m --json checks/Qwen3-Embedding-0.6B.json
```

Pass: `encode.py`'s self-check passes and `encode_check.py` reports no failure. If either
fails, stop: do not run P5r on an encoding that failed its contract. `extension_runs.sh all`
(Phase 14) does all of this itself: `encode_llm` encodes and checks on the GPU, then `p5r` runs the
stage below in parallel with the NQ replays. By hand (CPU, ~30-60 min; `submission_runs.sh all`
also runs it if the encodings exist when its parallel stages start):

```bash
bash scripts/submission_runs.sh llm
bash scripts/submission_runs.sh collect
```

It assembles `pairs/P5r`, writes the 0.6B window coverage to `coverage_llm/`, a measure-only
result (M_S, M_T, the Procrustes M_A, G and rho at K' = 25, 50, 500, ...) to
`results_llm/P5r.json`, and every adapter family to `adapters/adapters_P5r.json`. If the 8B
encodings are not at `enc/Qwen3-Embedding-8B`, set `LLM_TGT=<dir>`. The paper reports P5r as a
static pair, like P5p/P5q before their replays.

### 13.11 All of Phase 13 in one command

```bash
mkdir -p logs && nohup bash scripts/submission_runs.sh all > logs/submission.log 2>&1 &
tail -f logs/submission.log            # progress; one log per stage in logs/<stage>.log
```

`all` runs preflight, 13.3 alone, 13.2a (plan and commit), then 13.1, 13.2b, 13.5, 13.6, 13.7,
13.9 (unless `XARCH=0`) and 13.10 (only if its encodings exist) in parallel, then 13.8, then
writes `submission_outputs_<date>.zip`. The parallel stages split `nproc` cores by fixed shares
(d3 13/32, adapters 9/32 with one process per pair, mass, TREC-DL, the sample sweep, xarch and
llm 2/32 each), so the machine is never oversubscribed;
`THREADS=n` overrides this. Any stage that exits 0 without writing its outputs fails. The
script exits 0 only if every stage succeeded; on a failure it still runs the checks and writes
the zip from what finished, and names the failed stages in `logs/submission.log`.

If a stage fails, read its log, fix the cause, and re-run that stage alone
(`bash scripts/submission_runs.sh adapters`; a stage run alone gets every core), then
`bash scripts/submission_runs.sh check` and `bash scripts/submission_runs.sh collect`.
Re-running is safe: the plan is re-committed only if it changed (it cannot, unless the inputs
did), and replays already registered in `RUNS.tsv` are skipped as "already run". Do not re-run
`plan` after any d3 replay has started with different inputs: that would make the replays
post-hoc.

---------------------------------------------------------------------------------------------
## Phase 14 — extension: a second corpus with a pre-committed planner test, and P11 seeds

Run after Phase 13 finishes, on the same instance and committed branch (both drivers commit a
plan; they do not share files). The guide is `scripts/extension_runs.sh`; one command runs it:

```bash
bash scripts/extension_runs.sh preflight     # must end with "preflight ok"
nohup bash scripts/extension_runs.sh all > logs/extension.log 2>&1 &
```

**Why NQ.** Every MS MARCO result was replayed before the planner existed in its final form, so
on MS MARCO the planner can only be shown consistent after the fact. On the BEIR NQ 100k subset
(`subnq/`, built for P5c: Wikipedia passages, 1,000 judged queries) it has never been run. Its
plan is computed from static inputs and **committed before any NQ replay**, and each pair is
then replayed at the depth that plan names: a real prediction, scored by the same
`check_planner.py`. With 1,000 queries and 10^5 documents the windows cover a moderate share of
the corpus, so the planner has a decision to make (small BEIR corpora would be saturated, the
ceiling regime of the code corpus).

**Priority.** The NQ prediction is core: if time runs short, `bash scripts/extension_runs.sh
core` runs it alone (encode, bm25, pairs, measure, plan, replay, check, collect). `all` adds, in
this order of importance: the LLM->LLM pair P5r (`encode_llm` on the GPU, then `p5r`, which runs
13.10's `llm` stage), P11 at seeds 1-4 with `scripts/seed_summary.py` reporting the range of S4's
margin across the five runs, then the Zipf replays (14.7), the judged-query sweep (14.5) and the
throughput timing (14.6). GPU work runs in that priority order; a failed `encode_llm` or
`throughput` never blocks the NQ test.

```bash
python scripts/seed_summary.py results/P11.json results_seeds/P11_seed1.json results_seeds/P11_seed2.json --md seeds_summary.md
```

**Time.** `encode` needs the GPU: four encoders (bge-small, bge-base, bge-m3, e5-base; MiniLM
exists from P5c) on 10^5 passages and 1,000 queries, ~20-40 GPU-minutes. It can run while
Phase 13 is still running (`bash scripts/extension_runs.sh encode`), then `encode_llm` (P5r,
<1 h) and `throughput` (~10 min). The rest is CPU, ~1.5-2 h, bound by the MS MARCO replays (P11
seeds, Zipf; 34-73 min each).

### 14.1 Encode (GPU) and BM25

```bash
python scripts/encode.py --model BAAI/bge-base-en-v1.5 --kind doc --tsv subnq/corpus.tsv --out enc_nq/bge-base-en-v1.5/doc.npy --max-seq-len 512
python scripts/encode.py --model BAAI/bge-base-en-v1.5 --kind query --tsv subnq/queries.tsv --out enc_nq/bge-base-en-v1.5/q.npy --max-seq-len 512
python scripts/encode_check.py --enc enc_nq/bge-base-en-v1.5 --data subnq --json checks_nq/bge-base-en-v1.5.json
python scripts/bm25_run.py --corpus subnq/corpus.tsv --queries subnq/queries.tsv --k 1000 --out subnq/bm25.npy
```
(the first three lines for each of the four encoders; 512 tokens, as P5c's MiniLM.) Pass: every
`encode_check.py` reports no failure. **If one fails, stop**: do not build pairs on it.

### 14.2 Pairs and static inputs

`pairs` builds N0 (bge-base identity) and N2, N3, N4, N5, N9, N10, N11, N12, N13, N14: the ten
MS MARCO text pairs, same models, on NQ. `measure` writes coverage per source encoder
(`coverage_nq/`), the K' frontier per pair (`sweeps_nq/`, depths 10-1000) and a measure-only run
per pair (`results_nq_static/`: M_S, M_T, kappa, rho, G). Nothing is replayed.

```bash
python scripts/sweep.py pairs/N2 --config configs/nq_100k_v3.json --kprime 10,25,50,100,200,500,1000 --out-dir sweeps_nq/N2
python scripts/run_pair.py pairs/N2 --config configs/nq_100k_v3.json --name N2 --measure-only --out results_nq_static/N2.json
```

### 14.3 Plan, committed before any replay

```bash
python scripts/plan.py results_nq_static/N*.json --coverage-dir planner_nq/cov --frontier planner_nq/frontier_all.json --config configs/nq_100k_v3.json --csv planner_nq
```
The stage merges the sweeps, copies coverage, runs D1-D3, refuses an incomplete plan (a pair
without a decision, an S2 plan unpriced or at a depth with no pre-registered config) and
commits `planner_nq/plan.csv`. It refuses outright if any NQ replay is already registered.
Whatever the plan says is the prediction; **nothing is re-planned after a replay**.

### 14.4 Replays at the planned depths, adapters, P11 seeds

Each pair is replayed under `configs/nq_100k_v3_k<K'>.json` for the K' its committed plan names
(S0/S1 plans at the default 50; the six configs are identical but for the window, all
pre-registered 2026-11-09: `configs/nq_100k_v3.json` (K'=50) and `configs/nq_100k_v3_k25.json`,
`_k100`, `_k200`, `_k500`, `_k1000`; preflight and `tests/test_runbook.py` check all six). Every adapter family runs on each pair (`adapters_nq/`). In
parallel, `seeds` replays P11, the closest MS MARCO call (S4 clears tau by 0.000086), at seeds
1-4; `--set seed=...` marks them development runs, which the paper tables never include.

```bash
python scripts/run_pair.py pairs/P11 --config configs/msmarco_1m_v3.json --name P11_seed1 --set seed=1 --out results_seeds/P11_seed1.json
```

### 14.5 How many judged queries do D1 and D3 need? (~20-40 min)

G(K') needs target vectors for the documents in the judged queries' source windows (at most q*K';
a document sample cannot stand in). `query_sweep.py` recomputes D1 and D3 from q = 50, 100, 250,
500, 1000, 2000 evaluation queries, five seeds each, on vectors that exist, and reports the smallest
q at which every seed decides as the full set does, with the window encodings that q costs. Its
full-set rule is cross-checked against `planner/plan.csv`.

```bash
python scripts/query_sweep.py pairs/P2 --config configs/msmarco_1m_v3.json --q 50,100,250,500,1000,2000 --seeds 5 --plan planner/plan.csv --out-dir query_sweep
```

### 14.6 Encoder throughput and inline-repair latency (GPU, ~10 min; the paper's latency numbers)

```bash
python scripts/encode_throughput.py --tsv sub1m/corpus.tsv --max-seq-len 256 --model BAAI/bge-base-en-v1.5 --model intfloat/e5-base-v2 --out throughput.json
```
Documents per second for bulk re-encoding (cov(K') in accelerator-hours), and the median and
p95 time of one batched inline repair of 1, 25, 37 and 50 documents (`--repair`): 50 is a cold
query's whole window at K'=50, 37 the replays' mean there, 25 the window at K'=25, and 1 the one
extra forward pass S1 adds; each with median, p95, p99 and maximum. It also times S1's adapter
product and one background batch, the most a repair waits for the accelerator when the
background holds it and repairs go first at batch boundaries. The driver runs it on all four
MS MARCO target encoders. These are
the paper's latency numbers (Section 7.3); run it even if time is short.

### 14.7 A skewed workload (~1-1.5 h, in parallel)

`configs/msmarco_1m_v3_zipf.json` (pre-registered 2026-11-09) differs from the registered config
only in `workload: zipf:0.9`. P2, P5, P11 and P13 are replayed under it; winners, debts and costs
compare with their uniform replays, and Theorem 2's bound, which holds for any workload, is
checked on every curve:

```bash
python scripts/run_pair.py pairs/P2 --config configs/msmarco_1m_v3_zipf.json --name P2 --out results_zipf/P2.json
python scripts/coverage_curve_check.py --bound-only --results results_zipf --md curve_check_zipf.md
```
Pass: `0 points above the independent bound`. (The uniform multiplicity model is not compared
here: under Zipf arrivals it describes a different workload.)

### 14.8 Checks and what comes back

```bash
python scripts/check_planner.py --plan planner_nq/plan.csv --results results_nq --committed-at "$(cat logs/nq_plan_committed_at.txt)" --runs results_nq/RUNS.tsv --runs-tz "$(date +%:z)" --md planner_check_nq.md
```
Pass: `11 pairs scorable, 11 pre-committed`. Report the verdicts as they are; a FAIL is a finding.
`check` also writes NQ tables (`results_nq/tables/`), the tau curve and `FREEZE_CHECK_NQ.md`.
`collect` writes `extension_outputs_<date>.zip`; send it back with Phase 13's zip.
