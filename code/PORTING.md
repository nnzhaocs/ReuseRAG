# PORTING — merging the 2026-10-27 and 2026-10-29 changes into the team fork

The team fork (`/root/reusemap`, commits dab329a…5638a50) diverged from this copy. It added
`reusemap/embcache.py`, `reusemap/tfcompat.py`, drift adapters in `compat.py`, NQ/code
builders and run scripts. Those files were not available here, so this copy was changed
in self-contained pieces. Merge in this order.

## 1. Replay (no conflicts expected unless the fork edited these files)
Take these files whole: `reusemap/pair.py`, `reusemap/simulate.py`, `reusemap/runner.py`,
`reusemap/smoke.py`, `reusemap/heat.py`, `reusemap/probe.py`.
- `pair.py` imports `FITTERS, compatibility, relative_residual` from `compat.py`. The fork's
  drift adapters keep working if they are registered in `compat.FITTERS` under the names the
  configs use (`drift_mlp`, `drift_mlp_op`); κ no longer uses them.
- Old configs still run, but note: `eval_frac` now defaults to 0.5 (no leak) and an explicit
  `stream_len` overrides the `traffic_ratio` rule. Use `configs/msmarco_1m_v2.json`.
- Result JSONs gain `settings`, `warnings`, `leaky`, `n_traffic_pool`, `adapter_info`;
  the top-level `summaries/winner/trajectories` are the primary setting, so `run_lopo.py`
  is unchanged.

## 2. Encoding contracts (touches the fork's embcache)
Take `reusemap/models.py`, `configs/models.json`, `configs/golden/`, `scripts/golden_check.py`,
`scripts/model_quality.py`. Then, in the fork's `embcache.py`:
- replace `resolve_prefix(model, kind, override)` with
  `c = models.get_contract(model, allow_blocked=...)`, `prefix = models.prefix_for(c, kind)`,
  `prompt = models.prompt_name_for(c, kind)`;
- after loading the SentenceTransformer: `findings = models.apply_contract(st, c, max_seq_len)`;
- before encoding: `pad = models.padding_check(models.make_encode_fn(st, c, kind), texts)`,
  abort on `fail`;
- write `models.encoding_record(c, kind, max_seq_len, findings, pad)` into the cache sidecar
  and include the contract (not just the prefix) in the cache key, so a contract change
  invalidates the cache.
`env_versions()` lists `reusemap.tfcompat` under `shims` when it was imported; paper
encodings should show an empty list.
Or simply use this copy's `scripts/encode.py`, which does all of the above (without caching).

## 3. Replay v3 and the checks around it (2026-10-29)
Take whole: `reusemap/simulate.py`, `reusemap/runner.py`, `reusemap/winner.py`,
`reusemap/smoke.py`, `reusemap/pair.py`, `reusemap/provenance.py`, and the new scripts
`coverage.py`, `predict.py`, `encode_check.py`. Notes for the merge:
- **Nothing in v2 changes behaviour unless the config asks.** `eval_arrival` defaults to
  `"none"`, which is the v2 path; the v3 configs set `"first_service"`. The one deliberate
  change to the v2 path is that the served stream now has its own generator, so the orders are
  no longer compared on different traffic (see `DECISIONS.md` 2026-10-29).
- `provenance.git_version` now returns `<commit>-dirty:<diff hash>` and that string is part of
  the run key, so **the fork's existing RUNS.tsv keys change for dirty rows only**; clean rows
  keep their keys. Dirty runs register as `ok_dirty`, so a freeze check for "every row ok" now
  catches them. If the fork reads the status column, accept `ok` only.
- `scripts/model_quality.py` and `scripts/assemble_pair.py` now refuse mismatched inputs. If
  the fork's `embcache` writes vectors whose row count can differ from the corpus file, fix that
  rather than passing `--allow-partial`.
- `encode.py` gained `--self-check` (document text through the query path). It needs the model,
  so if the fork encodes through `embcache`, call `models.self_retrieval_check(emb,
  models.make_encode_fn(st, c, "query"), texts, idx, pool)` there and store it in the sidecar.
- `run_lopo.py` takes `--objective` and `--features`; v3 results carry `lopo_penalty`.

## 4. Then
Run `python tests/test_invariants.py` (81 tests as of 2026-11-10) and `python tests/test_runbook.py` in the fork,
re-encode the suspect models under `verified` contracts, check each with `encode_check.py`,
measure coverage (RUNBOOK Phase 5.5), pre-register v3, commit the predictions (Phase 6.5), and
re-run.
