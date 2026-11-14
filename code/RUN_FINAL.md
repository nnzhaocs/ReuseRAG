# Final runs (2026-11-10): seeds on every windowed pair, and the planner at 8.8M documents

Two experiments, one script (`scripts/final_runs.sh`), on the same instance and folder
(`/root/sub13run`). Everything earlier is done and in the paper; nothing here re-runs it.

| | what | hardware | time (estimate) | disk |
|---|---|---|---|---|
| A | P2, P3, P4, P5, P9, P10, P12, P13 at seeds 1-4: 32 replays, same config as the paper | CPU | 6-8 h, 4 at a time | none new |
| B | the full MS MARCO collection (8.8M passages): MiniLM, bge-small, bge-base documents; P2, P3, P9 static measurements; window histograms | GPU ~4.5 h, then CPU ~3 h | | **~63 GB** on the data disk |

A starts now. B needs the data disk expanded first (it has 1.9 GB free). They run together.

## Step 0 — install the code (5 min)

**Do not install while `final_runs.sh` is running** (replacing a script bash is executing can
break it, and the brief uncommitted state makes new replays refuse to start). If the seeds or
`all` are already running from `7af0950`, leave them: nothing in this zip changes what they
measure. Instead unzip the **full** code zip outside the run folder and run Step 1b from it:
`unzip -o reusemap_code_full_<commit>.zip -d /tmp/rfull`, then in `/root/sub13run` use
`python /tmp/rfull/reusemap_submission_code/scripts/zipf_scan.py ...` (it imports the code next
to it, reads `pairs/` and `results_zipf/` here, writes `zipf_scan/`, and touches no code here).
Install the changes zip after the runs have finished.

```bash
cd /root/sub13run
unzip -o /path/to/reusemap_changes_<commit>.zip -d /tmp/rsc
bash /tmp/rsc/reusemap_submission_changes/scripts/install_changes.sh \
     /tmp/rsc/reusemap_submission_changes "Final runs: seeds, full-corpus scale study, Zipf scan"
```

It checks every file against `CHANGES_SHA256.txt` before and after copying, refuses a folder
with uncommitted code, runs the tests, and commits only if all of that passed (last line:
`installed and committed: ...`). The commit timestamps `PREREG_SCALE.md`: commit before any
full-corpus vector exists. Run keys include the commit, so they change with it; the drivers
skip finished work by its output files, not by run key, so nothing finished is redone.

## Step 1 — start the seeds now (CPU)

```bash
SKIP_SCALE=1 bash scripts/final_runs.sh preflight     # "cores: 20 (nproc says 224)", "preflight ok"
nohup bash scripts/final_runs.sh seeds > logs/seeds_all.log 2>&1 &
```

Check the core line: it must say 20 (the instance's quota), not 224. If it says 224, put
`NCORES=20` in front of every command. A finished replay is skipped if the stage is run again.

## Step 1b — the Zipf scan (CPU, 30-60 min, alongside the seeds)

The Zipf(0.9) replays changed the winner on P2 and P5. This scans the cause statically on all ten
text pairs: the planner's quality inputs (gap, G ladder, D1/D3's decision) under Zipf(alpha) for
alpha in {0, 0.3, 0.6, 0.9, 1.2} and five popularity orders each, with the effective number of
judged queries. It checks itself against the Zipf replays (S0's replay debt must equal the
weighted gap: `diff +0.00e+00`). No replay, no new vectors.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nohup python scripts/zipf_scan.py \
  pairs/P2 pairs/P3 pairs/P4 pairs/P5 pairs/P9 pairs/P10 pairs/P11 pairs/P12 pairs/P13 pairs/P14 \
  --config configs/msmarco_1m_v3.json --zipf-results results_zipf --out-dir zipf_scan > logs/zipf_scan.log 2>&1 &
# when done:  echo $?  is not available after nohup; check the log instead:
#   grep -c "VALIDATION" logs/zipf_scan.log   -> 0   (FAILED or MISSING would mean a non-zero exit)
#   grep -c "Zipf(0.9) replay S0 debt" logs/zipf_scan.log   -> 4  (P2, P5, P11, P13 were checked)
zip -qr zipf_scan_$(date +%Y%m%d_%H%M).zip zipf_scan logs/zipf_scan.log
```

## Step 2 — expand the data disk, then start the scale study

In the AutoDL console, expand the data disk (数据盘 → 扩容) by **at least 70 GB**; then check:

```bash
df -h /root/autodl-tmp                         # Avail must be >= 63G
bash scripts/final_runs.sh preflight           # "disk: N GB free ... ~63 GB needed", "GPU: ...", "preflight ok"
```

If preflight says the MS MARCO files are not found, point it at the folder `sub1m/` was built
from (it needs `collection.tsv`, `qrels.dev.tsv`, `queries.dev.tsv`, the two TREC-DL query files
and their qrels): `export RAW=/that/folder`.

Then:

```bash
nohup bash scripts/final_runs.sh all > logs/final_all.log 2>&1 &
tail -f logs/final_all.log
```

`all` records the pre-registration commit, builds `subfull/` (all 8,841,823 passages, the same
7,077 queries as `sub1m/`; it stops if the queries differ), encodes the three models in chunks
(peak RAM one chunk, files on the data disk), checks each encoding, assembles P2f, P3f, P9f,
runs the sweeps and the window histograms, waits for the seeds (it does not start them a second
time), and writes the zip. Every stage skips work already done, so after a failure fix the cause
and run `all` again.

## Step 3 — send back

```bash
ls final_outputs_*.zip                         # the newest one
```

Send `final_outputs_<time>.zip` (no vectors: results, sweeps, histograms, checks, logs) and
`zipf_scan_<time>.zip`.

## What passing looks like

| output | pass |
|---|---|
| `logs/final_all.log` | ends with `ALL DONE` |
| `results_seeds/` | 32 new files `<P>_seed<s>.json` (plus P11's four) |
| `seeds_all.md` | one row per pair; read it, nothing is required of it |
| `checks_full/*.json` | three files, `hard_failures: 0` |
| `sweeps_full/` | `sweep_P2f.json`, `sweep_P3f.json`, `sweep_P9f.json` |
| `window_mass_full/` | two files |
| `SCALE_CHECK.md` | written if the 1M sweeps are in `sweeps/` and `sweeps_fine/` here; otherwise I compute it from the zip. Its pass/FAIL lines are results either way. |

## If something fails

- **Out of disk during encode**: expand more, delete `/root/autodl-tmp/enc_full/<model>/doc.npy.partial.npy`, run `all` again.
- **Out of memory during encode**: `CHUNK=200000` in front of `all`.
- **A seed replay fails**: read `logs/seed_<P>_<s>.log`; run `bash scripts/final_runs.sh seeds` again.
- **A sweep is slow**: expected; each 8.8M pair takes 30-60 min on 20 cores.
- **Running short on time**: the seeds alone are worth having; B's results are worth having for
  any one pair (P9f finishes first: it needs only the two 384-dimension models).
