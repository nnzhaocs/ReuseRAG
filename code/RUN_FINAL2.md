# Final runs, round 2 (2026-11-11): price the unions, more skews, intermediate sizes, replays at 8.8M

CPU only. No new encoding: everything reads vectors that already exist on the instance
(`pairs/`, `pairs_full/`, `$DATA_ROOT/enc_full/`). One script, `scripts/final2_runs.sh`, in the
same folder as before (`/root/sub13run`).

| stage | what | time (estimate) |
|---|---|---|
| union | static candidate-union coverage and quality of S2, S3, S4, S5 at K' = 25, 50, 100 from the serving side, uniform and Zipf(0.3/0.6/0.9/1.2), on the 10 text pairs; replay C_serve/repairs are diagnostic comparisons, not assumed identical | 40-70 min |
| prereg | the predictions for the new Zipf replays, **committed before any of them runs** | 1 min |
| zipf | 36 Zipf replays: alpha 0.3, 0.6, 1.2 on the 10 text pairs, 0.9 on the 6 not yet replayed; scored | 60-90 min |
| p5r | P5r (Qwen3-0.6B -> 8B) replayed like the text pairs | 20-40 min |
| subset | nested 2M and 4M corpora cut from the 8.8M vectors (needs ~13 GB and ~26 GB of disk) | 20-40 min |
| diag | why P3's gap shrank at 8.8M: paired bootstrap on the dev queries, TREC-DL condensed nDCG | 20-40 min |
| sweep2 | static sweeps of P2f, P3f, P9f at 2M and 4M | 40-80 min |
| replay2 | replays at 8.8M (P9f, P2f, P3f), then 2M and 4M; **all scale replays intentionally use fixed K'=50** so the same depth is compared across corpus sizes; each prediction is committed first | 2-4 h |

`all` runs two lanes in parallel (union -> prereg -> zipf -> p5r; subset -> diag -> sweep2 ->
replay2) and then writes `final2_outputs_<time>.zip`. Every stage skips work already done, so after
a failure fix the cause and run `all` (or the stage) again. `all` now returns non-zero if either lane
records a failed stage/sub-job; it may still collect partial outputs, but it will not print a false
`ALL DONE`.

## Step 0 -- install (2 min)

```bash
cd /root/sub13run
unzip -o /path/to/reusemap_changes_<commit>.zip -d /tmp/rsc2
bash /tmp/rsc2/reusemap_submission_changes/scripts/install_changes.sh \
     /tmp/rsc2/reusemap_submission_changes "Final runs 2: union prices, Zipf skews, intermediate sizes, 8.8M replays"
```

Against the instance's tree the zip adds the Final2 files and changes only a docstring in
`scripts/sweep.py`, the installer (it now also runs `tests/test_final2.py`) and `tests/test_runbook.py`
(it now checks this driver); nothing that produced a number in the paper changes. The last line
must read `installed and committed: ...`.

## Step 1 -- check, then start

```bash
df -h /root/autodl-tmp; free -g
bash scripts/final2_runs.sh preflight        # must print "full-scale inputs ok" and then "preflight ok"
nohup bash scripts/final2_runs.sh all > logs/final2_all.log 2>&1 &
tail -f logs/final2_lane1.log                # and logs/final2_lane2.log
```

If the data disk has less than ~40 GB free, cut only 2M: `SIZES=2000000` in front of the command.
If it has less than ~14 GB, `SIZES=` skips the intermediate sizes (the 8.8M replays still run).
An explicitly empty `SIZES` is preserved by the driver and is a supported no-op for `scale_subset.py`.

`preflight` treats the 8.84M replay as required, not optional. Before any Final2 work starts it
checks that `subfull/corpus.tsv` has exactly 8,841,823 passages; the full and 1M query files match;
all three full encodings exist with the expected row counts/dimensions and their `checks_full/*.json`
files have `hard_failures: 0`; `pairs_full/P2f,P3f,P9f` and the three `sweeps_full/sweep_*.json`
files are complete; every sweep contains K'=50; and all three scale-replay configs still set
`window: 50`. Any failure stops the run. `SIZES=` only skips the optional 2M/4M stages; it never
skips the 8.84M preflight or replay.

If `results_full/*.json` (or `results_s2m`, `results_s4m`) already exists, the driver does not trust
the filename: it keeps a result only if its provenance shows the same pair data fingerprint, the same
config (hashed exactly as `run_pair.py` hashes it) and no change under `reusemap/` or
`scripts/run_pair.py` since the commit that produced it; otherwise the replay is rerun. (Comparing
the full run key would force a rerun after every commit, and the other lane commits its predictions
while this one runs.) If results exist but no committed `PREREG_REPLAY_<size>.md` exists, the driver
stops instead of writing a prediction after seeing the replay.

Time: 0 GPU-hours (existing vectors only). On the 20-vCPU instance the lanes get 12 and 8 cores and
the whole run takes roughly 4--7 hours; on an 8-core machine (4 cores per lane) budget 14--22 hours.
Lane 1 finishes first; if time runs short, `collect` at any point (Step 2).

## Step 2 -- send back

`final2_outputs_<time>.zip` (results, prices, predictions, scores, logs; no vectors). If time runs
short, run `bash scripts/final2_runs.sh collect` at any point and send what exists: every stage's
outputs are useful on their own, and the predictions are already committed.

## What passing looks like

| output | check |
|---|---|
| preflight | prints `full-scale inputs ok: N=8,841,823, ... replay K'=50` and then `preflight ok`; any missing/mismatched 8.84M asset is a hard failure |
| `union_price/UNION_PRICE.md` | reports static candidate-union coverage. For the delayed demand-then-rand reference replay, inline repairs must not exceed that union; first-arrival C_serve is shown diagnostically and is **not required to equal it** |
| `PREREG_ZIPF2.md` | committed **before** the first `results_zipf_a*` file (`final2_outputs_commits.txt` has the time) |
| `ZIPF2_SCORE.md` | `identity holds on 36` (a smaller number means the replay's weights are not the scan's: stop and send the log) |
| `SCALE_DIAG.json` | three pairs, each size with a dev gap and a 95% interval for its change |
| `REPLAY_full_SCORE.md` | the replays at 8.8M scored against `PREREG_REPLAY_full.md` |
