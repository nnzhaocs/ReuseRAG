"""Provenance and the do-not-rerun registry.

Every result JSON records: git commit (+ '-dirty' if the tree has uncommitted changes),
a hash of the effective config, and a fingerprint of the pair's input files. The triple
defines a run key. results/RUNS.tsv lists every completed run; run_pair.py refuses to
repeat a run whose key is already registered with status ok (use --force to override).
"""
import datetime
import hashlib
import json
import os
import subprocess

REG_COLS = ["date", "name", "run_key", "git", "config_hash", "data_fp", "status", "out"]


def git_version(repo_dir):
    """'<commit>', or '<commit>-dirty:<8 hex>' where the hex identifies the uncommitted diff.

    The diff hash matters: it goes into the run key, so two different uncommitted states are
    distinguishable in RUNS.tsv instead of collapsing into one 'dirty' bucket.
    """
    try:
        commit = subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "--short=12", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
        diff = subprocess.run(["git", "-C", repo_dir, "diff", "HEAD", "--"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout or b""
        if not diff:
            return commit
        return f"{commit}-dirty:{hashlib.sha256(diff).hexdigest()[:8]}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def is_dirty(git):
    return "-dirty" in str(git)


def run_status(fails, git):
    """Status written to RUNS.tsv. A result from uncommitted code is never plain 'ok', so the
    freeze check ('every row ok') cannot pass on a scratch run, and lookup() will not treat it
    as the clean run of the same commit."""
    if fails:
        return "invariant_fail"
    return "ok_dirty" if is_dirty(git) else "ok"


def config_hash(cfg):
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:16]


def data_fingerprint(pair_dir, head_bytes=1 << 20):
    """Hash of each input file's resolved path, size, and first MB (cheap; detects swaps)."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(pair_dir)):
        path = os.path.join(pair_dir, name)
        if not os.path.isfile(path):
            continue
        real = os.path.realpath(path)
        h.update(name.encode())
        h.update(str(os.path.getsize(real)).encode())
        with open(real, "rb") as f:
            h.update(f.read(head_bytes))
    return h.hexdigest()[:16]


def run_key(git, cfg_hash, data_fp):
    """Key for the do-not-rerun registry.

    The dirty marker is PART of the key. Stripping it (as this did before 2026-10-29) made a
    scratch run from uncommitted code share a key with the clean run of the same commit, so the
    clean paper run was then refused as 'already run' and the dirty result stood in for it.
    `git_version` returns commit + '-dirty:<8 hex of the diff>', so two different uncommitted
    states are different keys too.
    """
    return hashlib.sha256(f"{git}|{cfg_hash}|{data_fp}".encode()).hexdigest()[:16]


def table_inputs(results):
    """{pair name: run_key} for the results a derived table was built from."""
    out = {}
    for r in results:
        name = r.get("name") or "?"
        out[name] = (r.get("provenance") or {}).get("run_key") or "no-run-key"
    return out


def write_table_inputs(csv_path, results, repo_dir=None):
    """Record what a table in results/tables was built from, in `<csv_path>.inputs.json`.

    A table there is a VIEW of the result JSONs, and nothing tied the two together. Two
    scripts write into that one directory -- summarize.py makes map/grid/probe/heat/calib,
    tau_curve.py makes tau_curve -- so a re-run that rebuilds one set and not the other leaves
    the tables side by side looking equally current. That shipped: in the 2026-11-07 archive
    map.csv and grid.csv matched the P*.json beside them row for row, while tau_curve.csv still
    described the previous generation -- 177 of its 416 rows disagreed, across ten pairs, and
    the stale rows were the ones that flattered the paper (no S4 winner anywhere, where the
    current results have S4 winning all four P11 cells at 3.2x the window's cost).

    The sidecar is per CSV, not per directory, because that is the granularity the bug has.
    """
    doc = {"built": datetime.datetime.now().isoformat(timespec="seconds"),
           "git": git_version(repo_dir) if repo_dir else "no-git",
           "inputs": table_inputs(results)}
    with open(csv_path + ".inputs.json", "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    return doc


def check_table_inputs(csv_path, results):
    """[] if the table matches the results now present; one message per problem otherwise.

    A result present but absent from the table is NOT a problem: `summarize.py --exclude P0 P6`
    is the documented way to build the paper tables, and the measure-only pairs appear in
    calib and nowhere else. The problems are a recorded input whose run_key has changed (the
    table describes a different run of that pair), a recorded input no longer present, and a
    table with no sidecar, which cannot be checked at all.
    """
    side = csv_path + ".inputs.json"
    base = os.path.basename(csv_path)
    if not os.path.exists(side):
        return [f"{base}: no {os.path.basename(side)}, so which results it came from is "
                f"unknown -- rebuild it with the current scripts"]
    try:
        with open(side, encoding="utf-8") as fh:
            rec = (json.load(fh) or {}).get("inputs") or {}
    except (OSError, json.JSONDecodeError) as e:
        return [f"{base}: unreadable sidecar ({e})"]
    if not rec:
        return [f"{base}: sidecar records no inputs"]
    now = table_inputs(results)
    msgs = []
    for name in sorted(rec):
        if name not in now:
            msgs.append(f"{base}: built from {name}, which is not among the results given")
        elif rec[name] != now[name]:
            msgs.append(f"{base}: {name} was run_key {rec[name]} when this table was written, "
                        f"now {now[name]} -- the table describes an older run")
    return msgs


def registry_path(out_path):
    return os.path.join(os.path.dirname(os.path.abspath(out_path)), "RUNS.tsv")


def lookup(reg, key):
    if not os.path.exists(reg):
        return None
    with open(reg) as f:
        rows = [dict(zip(REG_COLS, line.rstrip("\n").split("\t"))) for line in f if line.strip()]
    hits = [r for r in rows if r.get("run_key") == key and r.get("status") == "ok"]
    return hits[-1] if hits else None


def register(reg, name, key, git, cfg_hash, data_fp, status, out):
    """Append one row. Under an exclusive lock, and the header is written only if the file is
    still empty once the lock is held: two runs started in parallel on a missing RUNS.tsv used to
    both see 'missing' and both write a header (the drivers create it first; a hand-run did not)."""
    with open(reg, "a") as f:
        try:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass   # no flock (non-POSIX): the drivers' init_registry still writes the header first
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write("\t".join(REG_COLS) + "\n")
        f.write("\t".join([datetime.datetime.now().isoformat(timespec="seconds"), name, key, git,
                           cfg_hash, data_fp, status, os.path.abspath(out)]) + "\n")
