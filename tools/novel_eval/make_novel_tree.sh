#!/bin/bash
# make_novel_tree.sh [stock_tree] [novel_tree]: COPY of the stock sim tree + our novel-object overlay.
# The stock tree and its running sims are never touched. Refuses if the target exists.
set -e
SRC=${1:-/data/aaltonomous-humanoid}; DST=${2:-/data/work/novel/tree}
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ -e $DST ]] && { echo "$DST exists"; exit 1; }
cp -a "$SRC" "$DST"
cp "$HERE/sim_overlay/novel_assets.py" "$DST/aaltonomous/sim/aalto_task/novel_assets.py"
python3 "$HERE/sim_overlay/apply_overlay.py" "$DST"
