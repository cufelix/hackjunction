#!/bin/bash
# One A/B arm on Modal: classical policy from a snapshot of the integration worktree HEAD + overlay, sort_fixed,
# one episode per container (seed s = the same layout as box A STEP 0 trial s-1000), settle 3, nominal DR.
#   bash modal/ab_arm.sh <overlay> [seeds=1000-1019] [worktree] [rev=HEAD]
# Snapshot = `git archive HEAD` (uncommitted edits in the worktree are NOT included). Results: modal/runs/<overlay>_<sha>/
set -euo pipefail
OV=$1; SEEDS=${2:-1000-1019}
WT=${3:-<HJ_ROOT>/aalto-starter/work/integration}
REV=${4:-HEAD}
HERE="$(cd "$(dirname "$0")" && pwd)"
SHA=$(git -C "$WT" rev-parse --short "$REV")
git -C "$WT" log -1 "$REV" --format='%h %ci %s' | tee "$HERE/logs/ab_${OV}_${SHA}.sha"
git -C "$WT" cat-file -e "$REV:aaltonomous/baselines/classical/overlays/$OV.yaml"
if git -C "$WT" show "$REV:aaltonomous/baselines/classical/overlays/$OV.yaml" | grep -q '^placeholder: *true'; then
    echo "overlay $OV is still a placeholder at $SHA"; exit 3
fi
SNAP="${TMPDIR:-/tmp}/aalto_ab_snap/${OV}_${SHA}"
rm -rf "$SNAP"; mkdir -p "$SNAP"
git -C "$WT" archive "$REV" | tar -x -C "$SNAP"
TAG="${OV}_${SHA}"
CMD="env AALTO_OVERLAY=$OV AALTO_TIME_LIMIT=120 bash -c 'python tools/flags.py && exec python -m aaltonomous.baselines.classical.policy_server'"
cd "$HERE/.."
AALTO_WORKTREE="$SNAP" .venv-modal/bin/python -m modal run modal/aalto_eval.py --seeds "$SEEDS" --trials 1 \
    --settle 3 --tag "$TAG" --policy-cmd "$CMD" \
    --extra-args "--team $TAG --time-limit 120 --gate-max-clips 0 --gate-min-hz 27"
