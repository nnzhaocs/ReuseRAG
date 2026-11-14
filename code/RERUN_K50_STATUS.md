# Phase 7 replay window status (2026-11-07 09:45)

> **Historical record (status as of 2026-11-08).** This file describes the state at 09:45 on 2026-11-07. P10-P14 were re-run at K'=50 later that day (commit
> `57de2da`); all ten text pairs in the canonical archive are at K'=50. It is kept
> for provenance and is not updated. The paper's numbers are generated from the canonical
> 2026-11-07 data archive (`reusemap_paper_data_2026-11-07/`, results at K'=50 from commits
> `fbb23c9` and `57de2da`, listed per pair in that archive's README) by the paper's
> `build/make_numbers.py`; nothing is copied from this file. What remains to run is RUNBOOK Phase 13.


K* decision: window_frac = 5e-5 (K'=50), commit 57cb631.

Original Phase 7 replays (01:18-03:09) ran at placeholder K'=500, i.e. before the K* commit.
/root/fixwin.sh re-ran the ten pairs at K'=50 on 2026-11-07 09:03.

Completed at K'=50: P2 P3 P4 P5 P9 (results/*.json 09:37-09:38).
Not re-run (still K'=500): P10 P11 P12 P13 P14 (killed 09:39, no partial output written).
K'=500 originals of all ten pairs kept in results_win500/.
P0/P5c/P7/P8 were run at their own windows (500/500/200/200) and were not part of this re-run.

Winner is unchanged for every pair after the re-run (S2/demand_then_rand, P0 and P8 S0/rand).
Resume: for P in P10 P11 P12 P13 P14; do python scripts/run_pair.py pairs/$P --config configs/msmarco_1m_v3.json --out results/$P.json; done
