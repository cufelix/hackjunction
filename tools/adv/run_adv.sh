#!/usr/bin/env bash
# Usage: run_adv.sh <branch|sha|checkout_dir> <overlay> <profile> <seed0> <n> [port=8002]
# Runs on a sim box with the adv sim up (tools/adv/start_adv_sim.sh). Policy = checkout code in the integ image
# (like run_ab.sh), overlay via AALTO_OVERLAY, evaluated with `aalto.sh eval` of the ADV tree against the adv sim instance
# after switching its profile. profile: nominal | real_mild | real_hard | out_of_range.  task: ADV_TASK (default sort_fixed).
# Output: $ADV_RUNS/<profile>__<overlay>__<sha>__s<seed0>n<n>/{eval dir with eval.json, episodes.json, run.log, meta.json}
#   $ADV_RUNS default /data/work/oliver/adv/runs.   Branch/sha are resolved in $ADV_REPO (default /data/work/oliver/cand)
#   and checked out as a detached worktree under /data/work/oliver/adv/ck_<sha>.
set -u
[[ $# -ge 5 ]] || { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
SRC=$1; OV=$2; PROF=${3#adv_}; SEED0=$4; N=$5; PORT=${6:-8002}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/adv_env.sh"
ADV_RUNS=${ADV_RUNS:-/data/work/oliver/adv/runs}; ADV_REPO=${ADV_REPO:-/data/work/oliver/cand}
TASK_NAME=${ADV_TASK:-sort_fixed}; IMG=${IMG:-hackjunction/policy:integ-770888c}
if [[ -d "$SRC/aaltonomous" ]]; then CK=$(readlink -f "$SRC"); SHA=$(git -C "$CK" rev-parse --short=8 HEAD 2>/dev/null || basename "$CK" | sed "s/^ck_//")
else
  SHA=$(git -C "$ADV_REPO" rev-parse --short=8 --verify "$SRC^{commit}") || { echo "cannot resolve $SRC in $ADV_REPO"; exit 1; }
  CK=/data/work/oliver/adv/ck_$SHA
  [[ -d "$CK" ]] || git -C "$ADV_REPO" worktree add --detach "$CK" "$SHA" >/dev/null || exit 1
fi
OUT="$ADV_RUNS/${PROF}__${OV}__${SHA}__s${SEED0}n${N}"; mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "[$(date +%T)] ckout=$CK ($SHA) overlay=$OV profile=$PROF seeds=$SEED0..$((SEED0+N-1)) task=$TASK_NAME policy_port=$PORT sim_instance=$AALTO_INSTANCE"
bash "$HERE/set_profile.sh" "$PROF" || { echo "set_profile failed (adv sim up?)"; exit 1; }
NAME=pol-adv-$(basename "$OUT" | sed 's/[^a-zA-Z0-9_.-]/_/g')
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --rm --name "$NAME" -p 127.0.0.1:$PORT:8000 -e AALTO_OVERLAY="$OV" \
  -v "$CK/aaltonomous:/app/aaltonomous:ro" "$IMG" >/dev/null || exit 1
trap 'docker logs $NAME > "$OUT/policy_server.log" 2>&1; docker rm -f $NAME >/dev/null 2>&1' EXIT
sleep 5
docker exec "$NAME" python3 tools/flags.py > "$OUT/flags.txt" 2>&1 || true
printf '{"checkout":"%s","sha":"%s","overlay":"%s","profile":"%s","seed0":%s,"n":%s,"task":"%s","sim_instance":"%s"}\n' \
  "$CK" "$SHA" "$OV" "$PROF" "$SEED0" "$N" "$TASK_NAME" "$AALTO_INSTANCE" > "$OUT/meta.json"
T0=$SECONDS
cd "$ADV_TREE"
bash aaltonomous/cloud/aalto.sh eval --policy ws://localhost:$PORT --team "adv_${PROF}_${OV}" --task "$TASK_NAME" \
  --trials "$N" --seed "$SEED0" --time-limit 120 --settle 3 --gate-max-clips 0 --gate-min-hz 27 --out "$OUT"
echo "[$(date +%T)] eval rc=$? wall=$((SECONDS-T0)) s"
python3 "$HERE/adv_report.py" episodes "$OUT"
