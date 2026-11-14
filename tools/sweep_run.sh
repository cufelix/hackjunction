#!/usr/bin/env bash
# Sequential, resumable grasp-yaw sweep against ONE already-running sim. Never starts/stops the sim.
#   bash tools/sweep_run.sh [--dry-run] [--run-dir runs/sweep/<ts>] [--only trimmed|rest|all] [--time-limit 45]
# Order: trimmed (per arm: base, phi90, phi105, phi120) first, then rest (phi60, phi135 per arm).
# A cell is done when <run-dir>/<arm>_<cell>.done exists (13 rows written); re-run with the same --run-dir to resume.
# Run inside conda env isaac, from anywhere (cds to the worktree root). Never set AALTO_PROFILE=1.
set -u
DRY=0; RUN_DIR=""; ONLY=all; TL=45; ARMS="left right"; PYBIN="${PYBIN:-python}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1;; --run-dir) RUN_DIR="$2"; shift;; --only) ONLY="$2"; shift;;
    --time-limit) TL="$2"; shift;; --arms) ARMS="${2//,/ }"; shift;; *) echo "unknown arg $1" >&2; exit 2;;
  esac; shift
done
unset AALTO_PROFILE
cd "$(dirname "$0")/.." || exit 2
[ -n "$RUN_DIR" ] || RUN_DIR="runs/sweep/$(date +%Y%m%d-%H%M%S)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

CELLS=()                                  # "arm cell phi baselineflag"
if [ "$ONLY" != rest ]; then
  for arm in $ARMS; do CELLS+=("$arm base '' 1" "$arm phi90 90 0" "$arm phi105 105 0" "$arm phi120 120 0"); done
fi
if [ "$ONLY" != trimmed ]; then
  for arm in $ARMS; do CELLS+=("$arm phi60 60 0" "$arm phi135 135 0"); done
fi

if [ "$DRY" != 1 ]; then
  mkdir -p "$RUN_DIR"
  mkdir "$RUN_DIR/.lock" 2>/dev/null || { echo "locked: $RUN_DIR/.lock (another sweep_run on this dir?)" >&2; exit 3; }
  trap 'rmdir "$RUN_DIR/.lock" 2>/dev/null' EXIT
fi
echo "run dir: $RUN_DIR  cells: ${#CELLS[@]}  time-limit: $TL"
for c in "${CELLS[@]}"; do
  eval "set -- $c"; arm=$1; cell=$2; phi=$3; base=$4
  name="${arm}_${cell}"
  args=(--arms "$arm" --phis "$phi" --time-limit "$TL" --out "$RUN_DIR/$name.json")
  [ "$base" = 1 ] && args+=(--baseline)
  cmd=("$PYBIN" -m aaltonomous.sim.grasp_test_cls "${args[@]}")
  if [ "$DRY" = 1 ]; then
    if [ -e "$RUN_DIR/$name.done" ]; then echo "# skip (done) $name"; else echo "${cmd[*]} > $RUN_DIR/$name.log 2>&1"; fi
    continue
  fi
  if [ -e "$RUN_DIR/$name.done" ]; then echo "skip (done) $name"; continue; fi
  echo "[$(date +%H:%M:%S)] start $name"
  "${cmd[@]}" > "$RUN_DIR/$name.log" 2>&1; rc=$?
  n=$("$PYBIN" -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$RUN_DIR/$name.json" 2>/dev/null || echo 0)
  if [ "$rc" = 0 ] && [ "$n" = 13 ]; then
    date > "$RUN_DIR/$name.done"; echo "[$(date +%H:%M:%S)] done $name ($n rows)"
  else
    echo "[$(date +%H:%M:%S)] FAILED $name rc=$rc rows=$n (see $RUN_DIR/$name.log); stopping. Fix, then re-run with --run-dir $RUN_DIR" >&2
    exit 1
  fi
done
echo "all requested cells done: $RUN_DIR"
