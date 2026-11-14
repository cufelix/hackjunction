#!/usr/bin/env bash
# Build the adversarial sim tree on a sim box: copy of the ORIGINAL tree + our patch + profiles. Original untouched.
#   make_adv_tree.sh [orig=/data/aaltonomous-humanoid] [adv=/data/aaltonomous-humanoid-adv]
# Run from a checkout that holds tools/adv/ (or after scp of tools/adv to the box). Idempotent: re-applies from a fresh copy.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG="${1:-/data/aaltonomous-humanoid}"; ADV="${2:-/data/aaltonomous-humanoid-adv}"
[[ -d "$ADV" ]] && { echo "$ADV exists; remove it first (it must not be running)"; exit 1; }
cp -a "$ORIG" "$ADV"
(cd "$ADV" && patch -p1 < "$HERE/adv_sim.patch")
cp "$HERE"/profiles/*.yaml "$ADV/aaltonomous/sim/dr/"
echo "adv tree ready: $ADV  (start: bash $HERE/start_adv_sim.sh)"
