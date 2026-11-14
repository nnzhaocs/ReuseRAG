# Final2 code review / fixes (2026-11-11)

## Confirmed experiment intent

Scale replays are intentionally **fixed at K'=50** for P2f/P3f/P9f at 2M, 4M and 8.84M. They compare the same depth across corpus sizes; they are not D3-selected-depth replays. The three replay configs remain unchanged at `window: 50`.

## Fixes made

1. **No more false `ALL DONE`.** `scripts/final2_runs.sh all` now propagates failures from a lane or any sub-job. Partial outputs are still collected, but the command returns non-zero and prints `ALL FINISHED WITH FAILURES` when something failed.
2. **`SIZES=` now really skips 2M/4M.** The driver preserves an explicitly empty environment variable, and `scale_subset.py --sizes ""` is a supported no-op instead of crashing/running the defaults.
3. **Union output semantics corrected.** `union_price.py` now calls the quantity what it actually is: static candidate-union coverage over all servable queries. It is not claimed to be identical to first-arrival `C_serve`. Replay `C_serve` and inline repairs are diagnostic comparisons. Existing `cov` JSON fields are preserved for compatibility, with an explicit `cost_semantics` field and `static_candidate_union_cov` in replay comparisons.
4. **Tests/docs updated** to match the corrected semantics. The Final2 synthetic test now checks that the delayed reference replay's inline repairs do not exceed the static full candidate union, rather than requiring equality with first-arrival `C_serve`.

## Validation performed

- `bash -n scripts/final2_runs.sh`: passed.
- Python compile check for modified Python files: passed.
- `python tests/test_final2.py`: **5/5 passed**.
- `python tests/test_runbook.py`: passed (`101 commands, 30 scripts, 81 tests`).
- Explicit `SIZES=` driver check: returns 0 and skips intermediate sizes.
- Synthetic lane failure check: lane returns non-zero when a sub-job records failure.
- Synthetic `all` check: returns non-zero, collects partial outputs, and does not print a false success.
- The full `tests/test_invariants.py` suite was started in this sandbox but did not finish within the tool's execution limit; no invariant failure was observed before timeout. Run `preflight` on the experiment server before launching the full job, as the runbook already requires.

## Resource estimate (existing vectors present)

This Final2 round is CPU-only: **0 GPU-hours** if the existing full/subset vectors are present.

For an **8-core CPU** machine, plan roughly **14-22 hours wall-clock**, with up to ~24 h on slower storage. The driver auto-splits 8 cores into approximately 4 cores per lane. Disk requirements are unchanged: about 13 GB for 2M and 26 GB for 4M; `SIZES=` skips both intermediate corpora and still runs the 8.84M replay.

If the full 8.84M target vectors do not already exist, their encoding is a separate prerequisite and requires GPU time; it is not part of Final2.
