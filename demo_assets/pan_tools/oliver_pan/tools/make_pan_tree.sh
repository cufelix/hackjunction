#!/usr/bin/env bash
# Build the pan sim tree: cp -a of the stock tree + pan_sim.patch + object_pan.stl. Stock tree untouched.
#   make_pan_tree.sh [orig=/data/aaltonomous-humanoid] [pan=/data/aaltonomous-humanoid-pan]
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG="${1:-/data/aaltonomous-humanoid}"; PAN="${2:-/data/aaltonomous-humanoid-pan}"
[[ -d "$PAN" ]] && { echo "$PAN exists; remove it first (no sim may run from it)"; exit 1; }
cp -a "$ORIG" "$PAN"
(cd "$PAN" && patch -p1 --forward < "$HERE/pan_sim.patch")
python3 "$HERE/make_pan_stl.py" "$PAN/aaltonomous/scene/assets/object_pan.stl"
echo "pan tree ready: $PAN  (start: bash $HERE/start_pan_sim.sh)"
