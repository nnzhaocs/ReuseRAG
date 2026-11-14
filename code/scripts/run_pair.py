"""Run all measurements for one pair directory and write <out>.json; then check invariants.

Refuses to repeat a run already registered in <out dir>/RUNS.tsv (same code commit,
config, and input data) unless --force. Refuses a dirty git tree unless --allow-dirty,
because a result from uncommitted code cannot be traced back to a version.
Exit codes: 0 ok / already run, 1 invariant failed (pipeline bug), 2 refused.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from reusemap import provenance, runner, smoke  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("pair_dir")
ap.add_argument("--out", required=True)
ap.add_argument("--config", help="JSON file overriding runner.DEFAULT_CFG")
ap.add_argument("--name")
ap.add_argument("--identity", action="store_true", help="pair is the identity control (checks I1)")
ap.add_argument("--force", action="store_true", help="rerun even if already registered")
ap.add_argument("--allow-dirty", action="store_true", help="allow uncommitted code (not for paper numbers)")
ap.add_argument("--measure-only", action="store_true",
                help="measure M_S/M_T/kappa/rho/G and record the arrival block, then skip the "
                     "replay. Use it WITH the pair's own config (the one the replay will use), so "
                     "the window depths, the primary metric and the held-out split are the same "
                     "in the measurements as in the replay; scripts/predict.py joins the two and "
                     "silently loses depths, or predicts for the wrong query population, if they "
                     "disagree. Unlike --set this is not a development override.")
ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                help="override one config value for a DEVELOPMENT run, e.g. --set n_eval_queries=500 "
                     "--set 'traffic_ratio=[1.0]'. Recorded in the result and the run key; never use "
                     "for paper numbers (write to a separate --out directory).")
a = ap.parse_args()

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
name = a.name or os.path.basename(a.pair_dir.rstrip("/"))
cfg = dict(runner.DEFAULT_CFG)
if a.config:
    try:
        with open(a.config) as fh:
            cfg.update(json.load(fh))
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"refused: cannot read --config {a.config}: {e}")
if a.config and "tau" not in json.load(open(a.config)):
    # DEFAULT_CFG keeps tau = 0 (no band) on purpose: changing a default would change the config
    # hash of every run made from a config that omits tau. The paper's configs (*_v3*) all set it.
    print(f"WARNING: {a.config} sets no tau, so the winner rule runs with band 0, not the paper's "
          f"0.005 -- fine for a measure-only run, not for a replay meant to be compared with the paper",
          file=sys.stderr)
_tf = float(cfg.get("arrival_target_frac", 0.99))
if not 0 < _tf < 1:   # at 1.0 the arrival-rounds search never terminates: no finite T reaches it
    sys.exit(f"refused: arrival_target_frac must be in (0, 1), got {_tf}")
if a.measure_only and any(item.split("=", 1)[0] == "replay" for item in a.set):
    sys.exit("refused: --measure-only and --set replay=... set the same thing, and --set is "
             "applied last, so --set would silently win. Drop one.")
if a.measure_only:
    cfg["replay"] = False
overrides = {}
for item in a.set:
    k, _, v = item.partition("=")
    try:
        overrides[k] = json.loads(v)
    except json.JSONDecodeError:
        overrides[k] = v
if overrides:
    if "primary_setting" not in overrides and ("traffic_ratio" in overrides or "repeat_rate" in overrides):
        tr = overrides.get("traffic_ratio", cfg["traffic_ratio"])
        rr = overrides.get("repeat_rate", cfg["repeat_rate"])
        tr, rr = (tr if isinstance(tr, list) else [tr]), (rr if isinstance(rr, list) else [rr])
        if len(tr) == 1 and len(rr) == 1:
            overrides["primary_setting"] = {"traffic_ratio": tr[0], "repeat_rate": rr[0]}
        else:
            sys.exit(f"refused: --set gave {len(tr)} traffic_ratio and {len(rr)} repeat_rate "
                     f"values, so which one is primary cannot be inferred. Add "
                     f"--set 'primary_setting={{\"traffic_ratio\": {tr[0]}, "
                     f"\"repeat_rate\": {rr[0]}}}' (the runner would otherwise fail later with "
                     f"'primary_setting is not in the grid').")
    cfg.update(overrides)
    cfg["_dev_overrides"] = overrides
    print(f"DEVELOPMENT RUN: config overrides {overrides} (not for paper numbers)")
git = provenance.git_version(ROOT)
chash = provenance.config_hash({**cfg, "dev_overrides": overrides})
dfp = provenance.data_fingerprint(a.pair_dir)
key = provenance.run_key(git, chash, dfp)
reg = provenance.registry_path(a.out)

if provenance.is_dirty(git) and not a.allow_dirty:
    print(f"refused: code tree is dirty ({git}). Commit first, or --allow-dirty for scratch runs.")
    sys.exit(2)
prev = provenance.lookup(reg, key)
if prev and not a.force:
    print(f"already run (key {key}) on {prev['date']}: {prev['out']}  -- use --force to rerun")
    sys.exit(0)

res = runner.run_pair(a.pair_dir, cfg, name=name)
fails, notes = smoke.check(res, identity=a.identity)
res["invariants"] = {"fails": fails, "notes": notes}
res["provenance"] = {"git": git, "config_hash": chash, "data_fingerprint": dfp, "run_key": key,
                     "pair_dir": os.path.abspath(a.pair_dir), "config_file": a.config}
runner.save(res, a.out)
provenance.register(reg, name, key, git, chash, dfp, provenance.run_status(fails, git), a.out)
for n in notes:
    print("  ", n)
for f in fails:
    print("  INVARIANT FAILED:", f)
sys.exit(1 if fails else 0)
