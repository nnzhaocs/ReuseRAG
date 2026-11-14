# Paper outline — framing B

**This repo has never contained a `paper/` draft** (checked: `git log --diff-filter=A -- paper/*`
is empty), though the RUNBOOK referred to one in three places. So this is not an edit of your
draft — it is the framing-B skeleton the Phase 5.9 decision calls for: title, abstract, claims,
contribution list, section order, and the table-to-source map. Drop your A-framing text beside
it and move what survives, or send the draft and I will rewrite it directly.

Every number cited below is tier **PAPER** in `PAPER_NUMBERS.md`. Nothing here depends on an
unverified row, and the three headline claims survive τ anywhere in [0, 0.01], the 500 → 50
window change, and the choice of pair. **Two numbers are still provisional** and marked
*(pending 11.1)*: anything about the chosen window depth and the size of the ceiling.

---

## Title

> **Re-embed, Map, or Window? The Economics of Embedding Model Upgrades**

The original title asks a three-way question. Keep it — the measured answer is a clean "window,
never map", which is a better result than a title promising a predictor we do not have. Drop
"Characterizing Reuse Strategies": it promises a taxonomy, and the contribution is economics.

## Abstract (draft)

> Upgrading the embedding model behind a vector index forces a choice: re-embed the corpus,
> bridge the old index with a learned map, or re-rank a window of old candidates with the new
> model. We measure all three across twelve model pairs on a 1M-passage MS MARCO corpus and a
> 45K-document code corpus, under a replay that charges each strategy for the work it has done
> at the moment it first serves each query.
>
> Three findings. First, mapping does not work: across 336 pair × setting × tolerance
> configurations, no adapter-bridge strategy ever wins, and bridging costs between 0.12 and 0.82
> Recall@10 against the target model. Second, the cost of windowed reuse can be bounded before
> committing any accelerator time. Encoding the union of the served windows is a lower bound by
> construction; the question is how tight it is, and against what. A closed form assuming
> independent windows overstates that cost by 10–24% across the operating range, the union the
> workload actually touches closes the gap, and the replay confirms the bound is attained rather
> than exceeded by background work. Third, and against our own pre-registered hypothesis, *which* window
> strategy you choose does not matter: no selector built on space-compatibility features beats a
> constant predictor, that constant's mean regret falls inside the equivalence band we declared
> before running anything, and at the traffic intensity we pre-registered the background
> migration order leaves served quality unchanged.
>
> We report the second result as the usable one — a cost model you can evaluate before
> committing to a migration — and the third as a negative result with its falsification
> criterion met as stated.

## Claims, as they will be defended

| | Claim | Evidence | Survives |
|---|---|---|---|
| **C1** | The window family dominates re-embedding and mapping during an upgrade | S1 0 wins, S3 0, S5 0 of 336 cells; only S0 (39), S2 (276), S4 (21) ever win. S1's debt 0.12–0.82; M_A 0.0514–0.9497 | every τ, every pair, every traffic ratio |
| **C2** | Window cost admits a **tight static bound** (not a prediction — encoding the served windows' union is a lower bound by construction) | the closed form assuming independent windows, `1 − (1 − K′/N)^\|Q\|`, overstates cost by 10.6–15.5% at K′ = 50 and 16.6–24.3% at K′ = 200; the measured union closes that gap, and the replay confirms the bound is attained (C_serve/N vs coverage@K′ to 0.016–0.017% over 8 pairs × 3 encoders) | — |
| **C3** | Strategy selection is moot (negative, pre-registered) | majority 9/12 at regret 0.0018943; 1nn[κ,ρ] 8/12 at 0.0020117; 1nn[ρ] 6/12. Falsification condition from DESIGN.md met | every τ in [0, 0.01] |
| **S1** | G(K′) never binds in the measured range | worst +0.0035 at K′ = 50, ≤ +0.0000 at 500–5000, 11 text pairs | *(pending 11.1 — this is why the depth claim is provisional)* |
| **S2** | "Do nothing" is admissible exactly when the upgrade is worth less than the band | debt(S0) = M_T − M_S to 0.00e+00 on 11 of 12 pairs | state as the tautology it is, not a discovery |

**Not claimed**, and say so once in Limitations: anything about incomplete migration (that is
Thaw's C2 and fenced out — see `DECISIONS.md`); cross-family or cross-architecture generality
(no such pair was measured — `CROSS_FAMILY.md` in the run output is mislabelled); and the code
corpus as evidence for strategy choice (its coverage is 1.0000 at every depth, so its ceiling
is exactly 0 and only one strategy is ever economic there).

## Section order

1. **Introduction** — the upgrade decision; the three families; what a practitioner needs to
   decide before committing compute. End on the three findings.
2. **Problem setting** — corpus, workload, the migration period, what "serving" means. Define
   the first-arrival measurement here, because everything downstream depends on each query
   being measured once at first service.
3. **Strategies and measurables** — S0–S5; κ, CKA, ρ(K′), G(K′), coverage(K′). Keep this short:
   two of these turn out not to matter and the reader should not invest in them.
4. **The cost model** — coverage as the ceiling, then the C2 verification table. *This is the
   paper's centre of gravity.* It is also the only section a practitioner needs.
5. **What the strategies actually cost** — C1: the family comparison, mapping's failure, and
   the κ↔M_A observation (n = 10, as a secondary observation with n stated).
6. **Selection is moot** — C3, the LOPO null, with the pre-registered criterion quoted before
   the result. Then the τ curve, as the sensitivity analysis rather than a defence.
7. **The regime where none of this helps** — the code corpus. Window reuse needs
   queries × K′ small against N; a corpus whose query set approaches its document count is
   outside it. Short, and honest that it is one corpus.
8. **Limitations** — the four "not claimed" items; the single commit-provenance caveat if
   Phase 12 item 2 is not closed; τ as a declared judgement.
9. **Related work** — EmbedFlow, Drift-Adapter, Thaw (cite, and state the fence explicitly:
   this paper measures the complete-migration regime).
10. **Conclusion.**

Appendix A: the LOPO table in full. Appendix B: the τ curve. Appendix C: per-pair measurables.

## Table and figure sources

| Artifact | Content | Source |
|---|---|---|
| Table 1 | the twelve pairs: models, dims, M_S, M_T, κ, CKA, ρ@K′, G@K′, coverage@K′ | `results/tables/map.csv` |
| **Table 2** | **C2**: realised C_serve/N against static coverage, per pair | `results/tables/map.csv` + `coverage/*.json` |
| Table 3 | C1: debt by strategy family, per pair | `results/tables/grid.csv` |
| Table 4 | C3: LOPO accuracy and mean regret per rule, majority first | `results/lopo.json` |
| Figure 1 | the coverage/G frontier over the K′ ladder | `coverage/*.json` *(pending 11.1)* |
| Figure 2 | the τ curve: winner by pair against τ | `results/tables/tau_curve.csv` |
| Appendix C | per-pair measurables at every depth | `results/tables/map.csv` |

Every cost column must carry `C_serve_basis` beside it and `C_total_frac` in the same table —
`C_serve` is three different quantities and 1.0000 is the eventual cost of everything
admissible. `PAPER_NUMBERS.md` has the rules; do not quote `C_serve_frac` alone.

Table 1 needs a footnote for P11's κ = −0.0928 (Procrustes residual > 1, so the orthogonal fit
is worse than predicting zero). Table 3's winner column needs `tied_exact` beside it: on this
run at the pre-registered traffic intensity the background order did not move `debt`, so eight
pairs reported one of 3–4 identically-scoring choices and the identity control one of fourteen
spanning all six families. (At lower traffic intensities the order does move `debt`; say so.)

## Venue note

**ICDE 2027 has no experiments-and-analysis track.** Its submission options are Research
(Second Round), DEFT, and the PhD Symposium. That matters more than a line in the header,
because the plan above was built on an EAB framing and does not survive the move intact:

| | what an EAB track rewards | what a Research track rewards |
|---|---|---|
| a pre-registered null | a contribution — the design is the work | a weakness, unless the null refutes a published claim |
| a measurement over 12 pairs | a contribution in itself | evidence for a claim, not the claim |
| a tight bound on a known quantity | useful and publishable | needs to be the thing that is new |

Under a Research framing the order has to change again, and **C1 becomes the paper**: mapping
never wins across 336 configurations, with adapter quality of 0.05–0.76 against targets of
0.83–0.88. That is a refutation of a published method (Drift-Adapter, ref [1]) on a scale its
own evaluation did not reach, and a refutation is a research claim. C2 supports it by
explaining *why* the alternative is affordable — the bound, with its closed-form baseline — and
C3 becomes a scoping result: selection is moot in this regime, so the engineering question is
the regime boundary, not the selector.

The negative results then stop being the contribution and become the method's credibility:
pre-registered, with three of five hypotheses failing on their own stated criteria. Keep them,
report them in a table, but do not lead with them.

Two other routes worth weighing before committing:
- **VLDB's Experiment, Analysis & Benchmark track**, if it still runs in the relevant cycle,
  fits the current framing with no rewrite. Check the call rather than taking this note's word.
- **ICDE's DEFT track** suits a forward-looking piece, which this is not — it is a measurement
  of current practice. Probably the wrong home.

Whichever it is, the decision should be made *before* Phase 11.0 reports, because if G < 0
turns out to be judgment sparsity then C1 is all that is left standing and the Research framing
becomes the only one available.
