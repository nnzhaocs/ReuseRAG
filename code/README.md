# reusemap — artifact for "When the Bill Falls: Pricing Embedding-Model Migration from the Serving Index"

(Working title of the repository was "Re-embed, Map, or Window?".)

Offline replay of embedding-model migration strategies over cached encodings.
For each source→target model pair it measures compatibility, window residual miss
ρ(K′) and window gap G(K′), replays strategies S0–S5 × background orders, applies the
pre-registered winner rule, evaluates the bounded-cost probe (RQ3) and the heat gate
(RQ4), and checks pipeline invariants. `run_lopo.py` then does leave-one-pair-out winner
prediction (RQ2).

**Start with `scripts/sweep.py`** (RUNBOOK Phase 5.5). It measures ρ(K′), G(K′) and
coverage(K′) on one K′ ladder from vectors already on disk — no replay, no re-encoding — and
every later choice is made from its output: the window depth, the pair set, how large the saving
can be, and whether the map's own feature has any variance.

**Replay v3 (2026-10-29)** — each evaluation query is measured once, when the workload
first serves it; window repairs are encoded inline and charged; completion is an
admissibility constraint rather than part of the cost, and the objective is the quality
lost during the migration (`debt`). `DESIGN.md` has the arithmetic for why v2 could not
separate the strategies at all. Two things are knowable before any replay and should be
measured first: `scripts/coverage.py` (the effect ceiling, `1 − |∪ windows| / N`) and
`scripts/predict.py` (the predicted trajectory, to commit before running).

Start with `DESIGN.md` (purpose, claims, scope fence), then `DECISIONS.md`,
`UNVERIFIED.md`, `PREFLIGHT.md`; to run things, follow `RUNBOOK.md`.

## Status

| Part | State |
|---|---|
| `reusemap/` (search, metrics, compat, measure, simulate, winner, probe, heat, smoke) | tested on synthetic pairs |
| `tests/test_invariants.py` (invariants, estimator, metrics, Procrustes, winner/LOPO, heat gate, replay semantics, no-leak split, repeat rate, traffic grid, fixed κ, mapped-query flag, encoding contracts, padding check, input guards, batched top-k, measure-only mode, plus v3: arrival semantics, strategy-independent arrival sequence, no unpaid scoring, τ band, common-range area, coupon-collector arrival estimate, provenance dirty key, self-retrieval check, misaligned-encoding refusals, coverage against a direct union) | 81 pass (2026-11-10) |
| `tests/test_runbook.py` (every RUNBOOK command names an existing script, flag and config; stated test count matches) | ok |
| `reusemap/models.py`, `configs/models.json`, `scripts/golden_check.py` | contract logic tested with fake models; real encoding untested here (no models in the sandbox) |
| `scripts/make_synthetic.py`, `run_pair.py`, `run_lopo.py`, `summarize.py`, `coverage.py`, `predict.py`, `encode_check.py`, `sweep.py` | tested (v3 flow rehearsed end to end on five synthetic pairs) |
| `scripts/build_subsample.py` | tested on a tiny fake collection only |
| `scripts/encode.py`, `bm25_run.py`, `ann_noise.py` | **untested** (no models / faiss / bm25s in the dev sandbox); `encode.py --self-check` needs a real model |
| `scripts/assemble_pair.py` | its input-shape refusals are tested; the symlinking is not |
| Adapter families (S1/S3) | the replays use semi-orthogonal Procrustes; `scripts/adapter_families.py` (RUNBOOK 13.1) fits Drift-Adapter's three families (orthogonal, low-rank affine, residual MLP) plus ridge affine (low-rank at rank 64 and 128), each fitted on three training sets: documents, queries, both. Call S1 "a Procrustes bridge", not "Drift-Adapter". |

**Status 2026-11-08:** the 1M runs of 2026-11-04 and 2026-11-07 are real-data results, and the
paper uses the 2026-11-07 ones (see RUNBOOK "Status 2026-11-08"). This paragraph was written
before the first real run. Synthetic outputs are
for testing only and must not appear in the paper.

## Workflow

Python: the reported runs used 3.12, the version the code is tested on. The source must also
parse on 3.9-3.11; `tests/test_invariants.py` checks that only when such an interpreter is on
PATH, and says so when it skips. The current run order is in `RUN_NOW.md`. **The "81 tests passed" gate does not show that skip**: on a machine with only 3.12 that one check is silently not done. Any CI or release check must have a python3.9-3.11 on PATH; the authors' release zips are built and parsed under 3.11.

See `RUNBOOK.md` (environment → pre-registration → synthetic smoke test → data →
real smoke test P0/P3/P2 → full map → tagging). Test without data:
`python tests/test_invariants.py` (self-contained: each test builds its own synthetic pair; `scripts/make_synthetic.py --out /tmp/syn` is only for the sweep smoke test in RUNBOOK 0.1).

## Versioning and the do-not-rerun registry

- The repo is a git repository. Commit before any run whose numbers may reach the paper;
  `run_pair.py` **refuses a dirty tree** (exit 2) unless `--allow-dirty` (scratch runs only).
- Each result JSON carries `provenance`: git commit, config hash, data fingerprint
  (input file names, sizes, first MB), and the run key derived from them.
- `results/RUNS.tsv` lists every completed run. `run_pair.py` **skips a run whose key is
  already registered** with status ok (exit 0, prints where the result is); `--force` to repeat.
  This replaces a hand-maintained "LOST/DO_NOT_RERUN" file: the registry is written by the
  code, so it cannot drift.
- Tag the commit used for a paper table: `git tag paper-table1-v1`.

## Pair directory format

See `reusemap/pair.py`: `doc_src.npy, doc_tgt.npy, q_src.npy, q_tgt.npy, qrels.tsv`
(required); `bm25.npy, doc_len.npy, weights.npy, meta.json` (optional). Row indices are
document / query ids.

## Modelling choices you should know about

- **Exact search everywhere** for ρ, G, and quality, so measurements carry no ANN error.
  Windows of different depths are prefixes of one deep retrieval (nested), which makes
  ρ(K′) non-increasing by construction. ANN noise is P0's job.
- **Cost axis** C = number of documents with a target vector (each encoding counted once,
  from background or on-demand repair).
- **State-based checkpoints.** A query is *warm* if its whole window is encoded: it is
  ranked by target scores over the window ∪ target index over migrated docs. A *cold*
  query is served its nomination-order answer (old system's ranking for S2/S4/S5, mapped
  ranking for S3). Scoring not-yet-encoded window docs at a checkpoint would give window
  strategies target quality at C = 0 for free; mixing target-scored and unscored docs
  compares scores across spaces. `"checkpoint": "served"` gives the user-visible sync
  ranking for diagnostics only.
- **sync vs async** differ in the stream: sync encodes a served query's window inline
  (reported as `sync_enc_per_query_mean`, a latency proxy); async queues it ahead of
  background work.
- **O-demand** has no background work and need not complete; it is exempt from invariant I4.
- **Winner rule** (pre-registered): admissible if floor ≥ M_S − δ; winner = smallest
  C_ε, then larger AUC; exact ties broken by a fixed simplicity order (S0 < S1 < … ;
  rand < size < signal < demand) and all tied combos are recorded in `winner_ties`.
- **Probe cost** is dominated by the probe windows (n_queries × K′). At N = 1M and
  K′ = 500, 20 probe queries already cost 1% of N. Budget n_queries, not just m.
- **Document length** for O-size is in characters (`assemble_pair.py`); state this, or
  replace `doc_len.npy` with target-tokenizer lengths if GPU time per document matters.
- **`sync_enc_per_query_mean`** averages over *served* queries, so repeats of a query count
  (as 0 once its window is encoded); it is a per-request latency proxy, not a per-distinct-query cost.
- **Heat gate** uses fractions (dense-only share of exposure vs relative exposure
  change) to avoid a magnitude confound; O-signal runs only if the AUC's bootstrap 95%
  CI excludes 0.5.

## Cost at scale (rough)

See RUNBOOK §0 (RAM) and §5 (measured replay cost). Document vectors are memory-mapped
when already unit-norm (encode.py writes them normalised). Source retrieval depth is
max(largest K′, probe_c × strategy window), not probe_c × largest K′.
