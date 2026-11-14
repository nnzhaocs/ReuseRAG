# Pre-registration: the planner's inputs at 8.8M documents (written 2026-11-10)

Written and committed **before any full-corpus vector exists**. The commit that adds this file is
the timestamp; `scripts/final_runs.sh` records it in `scale_outputs/PREREG_COMMIT.txt` before it
encodes anything.

## What is measured

The full MS MARCO passage collection (8,841,823 passages; every judged passage is in it), built by
`build_subsample.py` with the same query and qrels files as `sub1m/`, so the query set is the same
7,077 queries in the same order. Three pairs whose models are cheap to encode, static measurements
only (`sweep.py`, `window_mass.py`, `configs/msmarco_full_v3.json`), K' in {10, 25, 50, 100, 200, 350,
500, 1000}:

| pair | source -> target | 1M: gap | 1M: G(25) | 1M: G(50) | 1M: D3 depth | 1M: covered docs at K'=50 |
|---|---|---|---|---|---|---|
| P2f | MiniLM-L6 -> bge-base | +0.0471 | +0.0094 | +0.0022 | 25 | 264,274 |
| P3f | bge-small -> bge-base | +0.0154 | -0.0032 | -0.0029 | 25 | 261,214 |
| P9f | MiniLM-L6 -> bge-small | +0.0317 | +0.0050 | -0.0010 | 25 | 264,274 |

No replay is run at 8.8M (20 vCPUs cannot replay it in the time left); the cost side rests on the
identity between on-demand cost and served-window coverage, which the 1M replays established.

## Predictions (each is checked per pair; the paper reports every outcome)

**S1, D1 reads a smaller corpus's gap as a lower bound** (Section 7.5, sample sweep): the gap
M_T - M_S at 8.8M is **not smaller** than at 1M, and D1's decision (migrate) is unchanged.

**S2, a fixed window loses quality as the corpus grows** (Section 8): G(K') at 8.8M is **not
smaller** than at 1M for K' = 25 and 50, so D3's depth at 8.8M (smallest K' > K with
G <= eps = 0.01 and gap - G > tau = 0.005) is **at least** its 1M depth, 25.

**S3, the windows overlap less in a larger corpus** (Section 4): the number of distinct documents
in the servable queries' windows at 8.8M is **at least** the 1M number at every K', and at most
m K' (m = 6,980 servable queries); so cov(K') falls, by at most 8.8-fold (the count can grow,
up to m K'), and the most a reuse strategy can defer, 1 - cov(K'), rises. The measured fall is
reported either way.

**S4, the cost model holds at the new size**: the independent-window value 1 - (1 - K'/N)^m
(m = 6,980) bounds the measured cov(K') from above at every K' and both sources.

"Not smaller" allows no tolerance: a single pair where the 8.8M value is lower, by any amount,
falsifies S1 or S2 for that pair, and the paper says so.

## Amendments (before any full-corpus vector exists)

- 2026-11-10 07:30 CET: S3's gloss said cov(K') "falls roughly 8.8-fold"; 8.8 is the bound, not a
  prediction (the distinct-document count also grows). The prediction itself, 1M count <= full
  count <= m K', is unchanged.
