# DESIGN — what this repo is for, and how the code maps to the paper

Read in this order: **DESIGN.md → DECISIONS.md → UNVERIFIED.md → PREFLIGHT.md**; to run, follow **RUNBOOK.md**.
`results/RUNS.tsv` (written by `run_pair.py`) is the list of runs already done — do not repeat them.

## Purpose (one sentence)
Measure what reusing the serving index actually saves when an embedding-model migration must
finish (it defers work, never removes it), price that deferral from the serving index before
migrating, and test the decision rules that follow (paper: "When the Bill Falls"). The original
purpose, testing whether κ and ρ(K′) predict the winning strategy, was falsified on
2026-11-04 (claim A below) and is reported as a null result.

## Falsifiable claims (these are NOT Thaw's C1–C3)
| Claim | Test | Falsified if | Draft |
|---|---|---|---|
| **A — the map predicts** — **FALSIFIED 2026-11-04**, see `DECISIONS.md`; the framing is B (RUNBOOK Phase 5.9) | LOPO over 6–10 real pairs, winner by the pre-registered rule | mean regret of the (κ, ρ) rule ≥ majority baseline's (regret is primary; accuracy secondary) — **met**: majority 0.00189 at 9/12, 1nn[κ,ρ] 0.00201 at 8/12 | H2, now an appendix null |
| **B — the probe is deployable** — **FALSIFIED** (paper Table 11, H4) | intruder-count estimator vs exact ρ(K′) | error > tolerance on more than one pair at the stated budget (tolerance and budget pre-registered) | H4 |
| **C — doc-level ordering helps (gated)** — **FALSIFIED** (paper Table 11, H5) | dense-only-share signal vs relative exposure change | AUC bootstrap 95% CI covers 0.5, or O-signal does not beat O-rand at equal budget → negative result | H5 |
| supporting | G(K′) grows as κ falls (H1); the low-κ/low-ρ cell is non-empty (H3, exploratory) | see draft | H1, H3 |

## Scope fence — stop if code starts doing any of these
- a new serving mechanism or online system (window re-rank exists: EmbedFlow; Thaw, under submission)
- exactness guarantees for mixed-state retrieval, economics of incomplete migration, chained upgrades (Thaw's C1–C3)
- runtime strategy switching
- claiming S2 *is* Thaw or EmbedFlow: S2 is this repo's implementation of the window-re-rank family. Its cold/warm semantics (cold query → old answer) are a modelling choice; the paper must describe them, not borrow either system's name for them.

## RQ → code → workload → metric
| RQ | Code | Workload | Primary metric |
|---|---|---|---|
| RQ1 map | `measure.py`, `simulate.py`, `winner.pick_winner_v3` | MS MARCO judged + random fill (1M) | v3 (the paper): `debt` at first arrival, then `C_serve`, each with its band τ (`runner._replay_v3`, `objective="debt"`); Recall@10 on MS MARCO dev. (v2 used C_ε, ε = 0.01, subject to floor ≥ M_S − δ; see "Replay protocol v3") |
| RQ2 prediction | `winner.lopo`, `scripts/run_lopo.py` | same, 6–10 pairs; P0 and P6 excluded | mean regret (×N), accuracy |
| RQ3 probe | `probe.evaluate_probe` | same | abs error of ρ̂ per pair, at total encodings ≤ budget |
| RQ4 ordering | `heat.heat_gate`, O-signal in `simulate` | ORCAS click frequencies (text); Zipf(α) α∈{0,0.6,0.9,1.2} (code) | AUC + bootstrap CI; trajectory gain vs O-rand |
| secondary quality | `metrics.ndcg_at_k` | TREC-DL 2019+2020 (`secondary_eval_qids`) | nDCG@10 |

## Why these workloads
- **MS MARCO dev + TREC-DL**: thousands of sparsely judged queries for ranking strategies, plus ~100 densely judged ones for graded quality. The subsample keeps every judged passage.
- **ORCAS**: real click frequencies, independent of any retrieval model — avoids defining heat by what the old model retrieved (circular).
- **CodeSearchNet**: corpus-scale code, where identifiers carry lexical signal (where S4 should matter).
- **CIPBench repos**: sanity checks only; too small (K′/N is large, ρ trivially near 0).
- **Zipf(α)**: code has no public query logs, so skew is a swept parameter, not an assumption.

## Replay protocol v3 (2026-10-29) — first-arrival measurement

v2 could not separate the strategies, for two independent reasons, both arithmetic.

1. **The warm rule.** A window was usable only when *every* document in it had been migrated.
   Under random background migration at fraction `f` a `K'`-document window is complete with
   probability `f**K'`, which is 50/50 only at `f = 0.5**(1/K')` — 0.9986 at K' = 500, 0.9965 at
   200. So every trajectory sat at the old model's quality until ~95% of N and then jumped, every
   `C_eps/N` was 1.000, and the winner came down to one checkpoint plus the tie order (a 0.13%
   difference in `C_eps` flipped a pair between S0 and S5). Nothing about reuse was being measured.
2. **The grid.** Draws are with replacement, so the rounds needed to serve the evaluation
   population is a weighted coupon-collector quantity, not `n_eval / stream_len`. At N = 1e6,
   `rounds` = 100 and K' = 500 that is ~1585 rounds at `traffic_ratio` 1, 397 at 4, 100 at 16,
   25 at 64. Background finishes at round 100, so at the v2 `primary_setting` (ratio 1) the
   migration was already complete before most evaluation queries were served: the predicted
   saving there is exactly zero, before any code runs.

v3 changes the measurement, not the cost accounting:

- **Each evaluation query is measured once, when the workload first serves it.** Its missing
  window documents are encoded inline, persist in `in_m`, and are counted in C like any other
  encoding. Nothing unpaid is ever scored (`DECISIONS.md` row 13), and no query is scored after
  its own repair, which is what made v1's O-demand win by construction (row 17).
- **The served stream has its own generator**, so the arrival sequence is identical for every
  strategy and order: all combinations are compared on the same queries, in the same order, at
  the same points of the migration. (In v2 one generator was shared with `background_order`,
  which consumes a permutation for `rand`/`size`/`signal` and nothing for `demand` — so the
  orders were being compared on different traffic.)
- **`repeat_rate` is gone from the primary design.** The mix of evaluation and other queries is
  the workload's own, not a knob; `traffic_ratio` remains and is the parameter that decides
  whether reuse can pay at all.
- **Completion is an admissibility constraint, not part of the cost.** At completion every
  strategy has encoded exactly N documents (`in_m` flips each document once), so total cost
  cannot separate them. Without the constraint the cost-minimal policy is "never migrate
  anything a served query did not ask for", which wins by construction — and whether never
  completing is acceptable is Thaw's C2, out of scope (see the scope fence).
- **The objective is `debt`**: the weighted mean, over the arrived queries, of what the fully
  migrated target system would have delivered minus what was actually delivered. It is the
  quality lost per query during the migration. `C_serve` (encodings at the last arrival if the
  population was served within ε) is kept as a diagnostic.
- **Both references are accumulated over the queries that actually arrived** (`target_mean`,
  `source_mean`), never compared against M_T/M_S, which are means over the whole evaluation set.
  The admissibility floor is `served_mean >= source_mean - delta`.
- **`demand_then_rand`** holds background work back until every evaluation query has arrived,
  then completes. Under this objective it is the cost-minimal policy that still completes, so
  leaving it out of the order set would put the best admissible policy outside the search space.
- **One equivalence band τ**, in the objective's units: everything within τ of the best is
  equivalent and the simplest of those wins (the default; the v3 configs set
  `band_tiebreak="cost"`, so the cheapest wins, see `DECISIONS.md` 2026-11-02). **One common range for every area**, so a run that
  stops early is not credited with a shorter integration range while the area still breaks ties.

### What is predictable without the replay
`G(K')` (`measure.py`) and `coverage(K') = |∪_q source top-K'(q)| / N` (`scripts/coverage.py`)
are both static. Together they give, per pair and before any replay:

- a window strategy is admissible iff `G(K') <= eps`;
- the optimal depth is the smallest `K'` with `G(K') <= eps`;
- the largest possible saving against a full rebuild is `1 - coverage(K')` — 12–14% at K' = 500
  on the first 1M run, which is the ceiling the paper has to state;
- the predicted cost is `1 - (1 - coverage)(1 - f_bg)`, `f_bg` being the background share done
  by the last arrival;
- a cold query at migration fraction `f` needs about `(1 - f) K'` encodings.

Depth buys the quality that makes a strategy admissible and costs coverage, which *is* the
saving. That trade-off is this paper's economics, and `scripts/predict.py` writes it out as a
prediction file to commit **before** the replay runs: agreement validates the implementation,
disagreement inside the stated band is the arrival schedule, outside it is a bug. The band goes
on `f_bg` alone — the union term is a fixed set given the source index and K', with no Monte
Carlo component.

**Open framing question, decided by the coverage measurement.** The query set and K' are fixed
across the pairs of one workload, so `coverage` varies between pairs only through the source
model's window concentration. If it spans a usable range, it is a third map feature and Claim A
has something to predict. If every candidate source lands at the same value, it predicts nothing
across pairs and the contribution is the derived economics plus its confirmation, with LOPO
secondary. That is a different paper from the one the draft `.tex` is written for, so run
`scripts/coverage.py` and decide before building the winner rule around either.

## Replay protocol v2 (2026-10-27, superseded by v3; `eval_arrival: "none"` still reproduces it)
- **Query roles.** A seeded `eval_frac` of the primary-judged queries is held out for
  evaluation; TREC-DL is evaluation-only; every other query is past traffic. The cache is
  warmed only by traffic, plus re-asked evaluation queries at an explicit `repeat_rate`.
- **Traffic vs encoding budget.** `traffic_ratio` = (queries per round × window) /
  background encodings per round. The winner is reported per (traffic_ratio, repeat_rate)
  setting; RQ2 predicts the winner at the pre-registered `primary_setting`, and the other
  settings show how sensitive the winner is to the workload.
- **κ is fixed.** Always document Procrustes, independent of the S1/S3 adapter.
- **What the grid means.** Small `traffic_ratio` serves few queries (0.25 ≈ 5 per round at
  1M): low traffic is the scenario, so demand-driven warming covers little of the corpus.
  `repeat_rate > 0` measures re-asked queries, whose windows may already be warm; it is not
  a cold-query evaluation and is reported as such.
- **Heat gate is pair-level.** Its AUC and the O-signal ordering are computed once, on the
  traffic distribution of `primary_setting`, and reused in every setting.
- **S4/S5 BM25 depth = window depth** (`bm25_depth`), so the lexical nominator gets the same
  candidate budget as the dense window.
- **Workload weights are explicit.** A pair with `weights.npy` (e.g. ORCAS clicks) must run
  with `workload: "file"`; any other workload is refused rather than silently ignoring it.
- **Encoders.** Explicit contracts (`configs/models.json`), padding check inside
  `encode.py`, golden-example check, ordering check, and a mapped-query screening flag in
  every result (`warnings`).

## Why reviewers should believe the metrics
- The floor is anchored to M_S (the system being replaced). Anchoring to M_T makes every strategy inadmissible at C = 0.
- The winner rule, δ, ε and window depth are committed in `configs/` before the first real run.
- Checkpoints are state-based: no strategy gets target quality for encodings it hasn't paid for.
- Exact search with nested windows: ρ and G carry no ANN error; P0 reports the ANN noise floor separately.
- LOPO reports regret, AUC reports a bootstrap CI, probe error is per pair (not pooled), and every relative number is shown with absolute M_S and M_T.

## External calibration (NQ, against EmbedFlow's public registry)
- Pair: all-MiniLM-L6-v2 → Qwen3-Embedding-8B on Natural Questions (EmbedFlow registry row; G(50) listed as 0.03255).
- Metric: **nDCG@10 at the same K′ (K′ = 50)**, using the same definition G(K′) = M_T − M_{T|S_K′}. Not Recall@10, and not each pipeline's own best K′.
- The definitions match, so a disagreement can only come from the setup: NQ subset, queries, prompts, truncation, dtype, exact vs approximate source retrieval. **Replicate EmbedFlow's registry record, do not reconstruct a similar subset** (RUNBOOK §3b). The public docs only say "canonical-prefix" subsets and max length 8192 for the Qwen3-4B/8B rows; the rest must come from the packaged registry record or the author. If it can't be matched exactly, P5′ is reported as a comparison, not a calibration. The calibration passes if G(50) agrees within a tolerance fixed before running (e.g. ±0.01) and G(K′) has the same shape over K′. If it fails, report the setup differences, not "their pipeline is wrong".

## Data licensing for the artifact
- MS MARCO and ORCAS: non-commercial research terms, not OSI licenses. **Release scripts, configs and result JSONs, not cached vectors** derived from them. If cached vectors must be shared, label them "non-commercial research use only, subject to the MS MARCO/ORCAS terms".
- CodeSearchNet, CoSQA, BEIR-NQ: check each dataset's own license statement before redistributing anything derived from it.

## Code → paper
| Module | Paper |
|---|---|
| `measure.py` | Table: κ, CKA, ρ(K′), G(K′) per pair; Figure: pairs on (κ, ρ) labelled by winner |
| `simulate.py` | trajectory figures (quality vs encodings) |
| `winner.py` | winner column; LOPO table |
| `probe.py` | probe-error table |
| `heat.py` | AUC/gate table; O-signal head-to-head |
| `smoke.py`, `tests/` | "Pipeline invariants" paragraph in Setup |
| `provenance.py` | artifact appendix: every number traces to a commit, config hash and data fingerprint |
