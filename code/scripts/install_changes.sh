#!/usr/bin/env bash
# Install a reusemap_changes_<commit>.zip into the run folder, all or nothing.
#
#   unzip -o reusemap_changes_<commit>.zip -d /tmp/rsc
#   bash /tmp/rsc/reusemap_submission_changes/scripts/install_changes.sh /tmp/rsc/reusemap_submission_changes "<commit message>"
#
# Run it from the run folder (/root/sub13run). It stops at the first problem: a file that does
# not match CHANGES_SHA256.txt (before or after copying), uncommitted code in the folder, or a
# failing test; nothing is committed unless every check passed.
set -euo pipefail
R=${1:?usage: install_changes.sh <unzipped reusemap_submission_changes dir> "<commit message>"}
MSG=${2:-"Install $(basename "$R")"}
[ -f "$R/CHANGES_SHA256.txt" ] || { echo "no $R/CHANGES_SHA256.txt"; exit 1; }
[ -d .git ] && [ -d reusemap ] || { echo "run this from the run folder (it has .git and reusemap/)"; exit 1; }
[ -z "$(git status --porcelain -- reusemap scripts configs tests compat)" ] \
  || { echo "uncommitted changes in reusemap/ scripts/ configs/ tests/ compat/: commit or stash them first"; exit 1; }
( cd "$R" && sha256sum -c CHANGES_SHA256.txt --quiet ) || { echo "STOP: the unzipped files do not match CHANGES_SHA256.txt (corrupted transfer?)"; exit 1; }
cp "$R"/*.md . && rm -f APPLY.md
cp -r "$R"/{compat,configs,reusemap,scripts,tests} .
sha256sum -c "$R/CHANGES_SHA256.txt" --quiet || { echo "STOP: installed files do not match; nothing committed"; exit 1; }
chmod +x scripts/*.sh
t=$(python tests/test_invariants.py | tail -1); echo "$t"
case "$t" in *" tests passed") ;; *) echo "STOP: invariant tests failed; nothing committed"; exit 1 ;; esac
f2=$(python tests/test_final2.py | tail -1); echo "$f2"
case "$f2" in *" final2 tests passed") ;; *) echo "STOP: Final2 tests failed; nothing committed"; exit 1 ;; esac
python tests/test_runbook.py | tail -1
git add -A ./*.md compat configs reusemap scripts tests
git commit -q -m "$MSG"
echo "installed and committed: $(git log --oneline -1)"
