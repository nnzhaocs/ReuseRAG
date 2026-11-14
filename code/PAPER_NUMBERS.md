# PAPER_NUMBERS — what the 2026-11-04 1M run may be quoted for, and what it may not

> **Every number in this file is from the 2026-11-04 twelve-pair run unless a line says otherwise. It will not match the paper**, whose numbers come from the 2026-11-10 archive (`reusemap_paper_data_2026-11-10/`) through `build/make_numbers.py`. Do not check the paper against this file.

> **Historical record (status as of 2026-11-08).** This file describes the 2026-11-04 K'=500 run and its 2026-11-06 addendum. Since then: Phase 11.1 finished (K'=50 is
> the smallest depth admissible on every pair; P11's G(25) = 0.01368 > eps), all ten text pairs were
> re-run at K'=50, and P7 is the 45,773-document rebuild. The `quality_5models.txt` row predates the
> encoding-contract fixes and the paper does not cite it: model quality in the paper is each pair's
> own measured M_S/M_T on the 6,980 judged dev queries. It is kept
> for provenance and is not updated. The paper's numbers are generated from the canonical
> 2026-11-07 data archive (`reusemap_paper_data_2026-11-07/`, results at K'=50 from commits
> `fbb23c9` and `57de2da`, listed per pair in that archive's README) by the paper's
> `build/make_numbers.py`; nothing is copied from this file. What remains to run is RUNBOOK Phase 13.


Every number here was recomputed from the result JSONs of that run, not copied from its
`SUMMARY.md`. Three tiers:

- **PAPER** — provenance-clean, reproduced from the raw results, safe to put in a table now.
- **ARCHIVE** — keep for the record, never cite: superseded, dev-override, or mislabelled.
- **BLOCKED** — the measurement is sound but a claim built on it needs one more step first.

**2026-11-06 addendum — a second archive arrived (`results1.zip`).** The eleven text results in
it are **byte-identical** to this run, so everything below still stands for them. What is new:
P8 (code→code), P5c (NQ 100k, MiniLM→Qwen3-8B, measure-only), `onnx_check.json`, a re-run
quality table and `tables/tau_curve.csv`. Three cautions, in order of consequence:

1. **The query set diverged.** The map describes Q = 6980 judged queries (`res_P2.json`:
   `n_eval` 3490 + `n_traffic_pool` 3490). The re-run `quality_5models.txt` describes **7077**,
   with every model's Recall@10 lower by 0.003–0.006 and nDCG@10 higher by ~0.007. That is a
   different judged set, not a re-measurement, so the two tables cannot both describe one setup.
   The code corpus demonstrably was rebuilt (P7: 45,283 → 45,773 documents, data fingerprint
   `49cb1bab…` → `6458d525…`, and P7 was re-run against it); the text pairs were **not** re-run.
   Resolve before freezing: run `scripts/run_pair.py` on one text pair without `--force`. If it
   reports "already run", the text corpus is unchanged and only the quality table needs
   re-running on the 6980 set. If it re-runs, the subsample moved under the map and the eleven
   text results are stale.
2. **P7 and P8 are `ok_dirty`** (`d4609e25ffa7-dirty:a57d960f`) and **P5c is `no-git`**. None of
   the three new results is quotable under Phase 12 item 1.
3. **P8 is absent from `tables/tau_curve.csv`** (12 pairs covered, P8 not among them), and the
   two P7 variants are on different corpora — `_carried_P7_20261104.json` at N = 45,283 and
   `res_P7.json` at 45,773. Pick one and say which.

New results worth having once their provenance is clean: `onnx_check.json` shows the stored
vectors agreeing with transformers, sentence-transformers and ONNX to ~1e-12 on all five models,
which is the strongest encoder-contract evidence in the project; P8 confirms the zero-ceiling
regime on a second code pair (C_serve/N = 1.0000); and S1/S3/S5 still win nothing, now across
fourteen pairs.

Provenance of the run itself: 73 registered runs, **all `ok`**, no dirty trees, no
`_dev_overrides` on any of the twelve paper pairs, **zero invariant failures** across all
fourteen result files. That clears Phase 12 item 1. It does *not* clear item 2 — the twelve
pairs span **four commits** (`7501f41` ×6, `60e9c35` ×5, `20f8760` ×1, with `a780b91` for the
superseded generation) and the archive ships no bundle, so "a later commit that changes no
measured code path" is unverified. Resolve that before the freeze, or re-run the odd pairs
on one commit.

---

## PAPER — quote these

| Number | Value | Source | Checked how |
|---|---|---|---|
| Pairs, corpus | 12 pairs (10 text + identity control P0 + code pair P7); N = 1,000,000 MS MARCO / 45,283 CodeSearchNet-py | `res_P*.json` `N`, `meta` | — |
| Held-out eval set | 3,490 queries of 6,980 | `n_eval` | — |
| Per-pair M_S, M_T | 0.8300–0.8752 Recall@10 | `settings[].summaries[].source_mean` / `target_mean` | reproduced |
| κ, CKA, ρ(K′), G(K′) ladders | K′ ∈ {50, 500, 1000, 2000, 5000} | `compat`, `rho`, `G` | reproduced |
| **Realised cost = static coverage** | C_serve/N vs coverage@50: 0.264861/0.264907, 0.255091/0.255134, 0.262135/0.262176 — **0.016–0.017% relative**, 8 pairs, 3 source models | `summaries[].C_serve_frac` vs `cov_*.json` | recomputed both sides |
| **Claim A falsified** | majority 9/12 at regret 0.0018943011779689507; 1nn[κ,ρ] 8/12 at 0.0020117; tree2[κ,ρ] 8/12 at 0.0028713; 1nn[ρ] 6/12. The two rules matching majority's accuracy match its regret to every digit | `LOPO.md` | per-rule means recomputed |
| **Mapping never wins** | S1 0, S3 0, S5 0 wins across 336 (pair × setting × τ) cells; only S0 (39), S2 (276), S4 (21) ever win | `summaries[]`, τ grid | recomputed with completion enforced |
| Adapter quality | M_A 0.0514–0.9497; worst five pairs below 0.35; S1 debt 0.12–0.82 | `summaries["S1/rand"].served_mean` | reproduced |
| **G never binds** | worst-case G = **+0.0035** at K′ = 50 and **≤ +0.0000** at 500/1000/2000/5000, over all 11 text pairs | `G` | recomputed |
| Coverage ladder | K′ = 50/100/200/500/1000 → 0.2649/0.4374/0.6453/0.8787/0.9646 (MiniLM); concentration peaks at K′=200 | `cov_*.json` | — |
| **Coverage beats its closed-form baseline** | the coupon-collector form `1 − (1 − K′/N)^\|Q\|` **overestimates** cost at every depth and encoder, by 10.6–15.5% at K′ = 50, 16.6–24.3% at K′ = 200 (where the gap peaks) and 3.5–6.4% at 1000. In absolute terms 28,299–39,479 documents at K′ = 50 and 107,178–147,203 at K′ = 200, per 10^6. Already computed: `coverage_if_independent` and `concentration` in every `cov_*.json` | `cov_*.json` | recomputed from the formula |
| Coverage spread across source models | 0.2551–0.2663 at K′ = 50 over five models; spread **0.0112** | `cov_*.json` | recomputed |
| **S0's debt is the quality gap — where the background does not finish first** | `debt(S0) == M_T − M_S` to **0.00e+00** on all 10 text pairs and P0. **Both code pairs depart**: P7 by −0.0185, and P8 at every traffic ratio (−0.000739 against a gap of +0.000529 at ratio 64), where the gap itself moves with traffic ratio. Same mechanism as the order-dependence row — the code corpus is small enough (N = 45,773) that the background completes before the population is served, so some queries get target quality and the rebuild's debt falls below the gap. One mechanism, both exceptions | `summaries["S0/rand"]` | recomputed |
| Every completing strategy encodes N | **828 of 828** completing combinations have `C_final == N` (also enforced by invariant I4′) | `summaries[].C_final` | exhaustive |
| Background order is immaterial **at high traffic intensity only** | At the primary setting (ratio 64) `debt` is identical across completing orders on every pair but P11, and 8 pairs reported one of 3–4 exact ties with P0 one of 14 spanning all six families. **Not a general property**: at ratio 4 and 16 `debt` varies across completing orders on every non-degenerate pair, 42 of 48 (pair × setting × strategy) cells varying somewhere. The mechanism is that window strategies also search the target-side index over already-migrated documents, so what the background has reached does affect the served answer — it just has not reached anything yet by the time a dense workload is served (S0's cost-at-service is 0.63 of N at ratio 4 against 0.03 at 64) | `tied_exact` | recomputed |
| τ sensitivity | 5/48 cells move at τ = 0.004, 4/48 at 0.006, 22/48 at τ = 0; P14 → window family at τ ≤ 0.004, P11 → window family at τ ≥ 0.006 | `scripts/tau_curve.py` | the shipped rule, swept |
| Code pair's ceiling is zero | coverage = 1.0000 at every depth 23–1000; 43,730 queries over 45,283 documents | `configs/csn_py_v3.json`, `res_P7.json` | — |
| Model quality order | e5 0.8681 > bge-base 0.8668 > bge-small 0.8570 > bge-m3 0.8468 > MiniLM 0.8245 | `quality_5models.txt` | *numbers* usable; see BLOCKED for the contract column |

## ARCHIVE — keep, never cite

| Thing | Why |
|---|---|
| `_old_k500/` (window 500, no cost tie-break) | superseded generation. Useful as a robustness note — **the 500 → 50 change moved no winner** — but not a table |
| `_k50_ladder_v1/` | window 50 but ρ/G measured only at 500+; the final generation added depth 50. Winners identical to the final run |
| `res_P5p.json`, `res_P5q.json` | carry `_dev_overrides`; `SUMMARY.md` already refuses them, and so does `summarize.py` |
| `CROSS_FAMILY.md` | titled BERT→Qwen3-Embedding-8B, but its P5 row is bge-m3→bge-small (ρ@50 = 0.2412, identical to the main map). **There is no cross-family measurement in this run** — Qwen3 got a contract and a six-document smoke test. Retitle or delete; do not cite as external validity |
| Any first-1M or v2 result | eval-query leak / v2 replay could not separate strategies (RUNBOOK status table) |

## BLOCKED — sound measurement, claim needs one more step

| Claim you might want | What blocks it | Step |
|---|---|---|
| "We chose the smallest admissible window depth" | **Resolved 2026-11-08** (Phase 11.1, `sweeps_fine/`: K′ = 10 and 25 measured; the depth is now D3's per pair, K′ = 25 on eight pairs and 50 on P11). As of this record: **false as it stands.** K′ = 50 is the smallest *measured* depth, and G is admissible at every depth on the ladder, so the frontier is censored from below. The 73.5% ceiling is a **lower bound** set by where the ladder stopped | Phase 11.1: sweep K′ = 10, 25 |
| "The protocol revision was a predictive test" | `predictions/` was never committed for the ten pairs, so there is nothing to check the replays against | Phase 11.2 |
| Any per-model contract status | `quality_5models.txt` is dated 2026-10-30 and pre-dates the Phase 4.2 sidecar rename; four of five models print `no sidecar`. The *quality numbers* are fine; the verification column is unknown | re-run after 4.2 |
| "κ predicts the adapter's quality" | r(κ, M_A) = **+0.742** on the ten text pairs excluding P0 (p ≈ 0.014, n = 10) — suggestive, not established, and it does not transfer across corpora (P7: κ = 0.0607 with the highest M_A of any pair, 0.9497). CKA's apparent signal over all twelve is carried entirely by P0: drop it and r(CKA, M_A) = −0.122 against κ's +0.415 | report as a secondary observation with n stated, or add pairs |
| P11's κ in a "compatibility" column | κ = **−0.0928** (Procrustes residual 1.0928 > 1: the orthogonal fit is worse than predicting zero) | footnote it, or clip with a note — do not print bare |
| Anything about the lazy-migration regime | `S2/demand` pays 0.2649 of N with debt −0.0040 and dominates every admissible row on both axes, but that is the economics of incomplete migration — **Thaw's C2**, outside the scope fence | out of scope (`DECISIONS.md` 2026-11-04). This paper may say only that its cost column is cost-at-service and the remainder is paid later |
| The code pair as cross-domain evidence for the map | its ceiling is exactly 0, so only one strategy is ever economic there; it cannot show different pairs choosing differently | it is a finding of its own (the window family needs queries × K′ small against N), not map evidence |

## How to read the cost columns

Never quote `C_serve_frac` alone. `C_serve_basis` says which of three situations produced it,
and they are not comparable with each other:

- `at_last_arrival` — work the strategy actually did. The only basis on which two rows compare.
- `no_repair_needed` — it reached target quality having repaired nothing, so the number is how
  far the background happened to get: 0.63 of N at traffic_ratio 4 down to 0.00 at 256, with
  nothing about the strategy changing. A small number here is a quality concession inside τ.
- `never_reached_target` — N by convention, so the column reads 1.0000 whatever happened.

`C_total_frac` is 1.0000 for every admissible row, because everything that completes encodes N.
A `COST TIE-BREAK ACROSS BASES` warning means the band was ordered across two of those bases —
the P14 case, where S0's 0.0300 beat the window's 0.2642 at ratio 64 though both encode N. The
ordering is pre-registered and stays; report the winner with its basis beside it.

## The three sentences this run supports

1. The window family dominates re-embedding and mapping during an upgrade: across 336
   (pair × setting × τ) cells, mapping wins none, and S1 costs 0.12–0.82 Recall@10.
2. Its cost admits a tight static bound — the union of served windows is a lower bound by
   construction, it is 10.6–15.5% (K′ = 50) to 16.6–24.3% (K′ = 200) tighter than the closed
   form assuming independent windows, and the replay confirms it is attained to within 0.02%
   over eight pairs and three source models. The last clause validates the implementation; the
   middle one is the result.
3. Which window variant you pick does not matter: no feature-based selector beats a constant
   predictor, the constant's mean regret (0.0019) falls inside the band the study declares
   equivalent, and at the measured primary setting the background order leaves `debt`
   identical across completing orders (though not at lower traffic intensities — see above).

All three survive τ anywhere in [0, 0.01], the 500 → 50 window change, and the choice of pair.
