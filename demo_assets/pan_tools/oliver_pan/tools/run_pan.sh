#!/usr/bin/env bash
# Usage: run_pan.sh <checkout_dir> <overlay> [variants=all] [port=8003]
# Policy = <checkout>/aaltonomous mounted into $IMG (default hackjunction/policy:rc1-7a4168e-g), overlay via
# AALTO_OVERLAY; episodes from run_pan.py against the running pan sim. Out: $PAN_RUNS/<overlay>__<sha>__<stamp>/
set -u
[[ $# -ge 2 ]] || { sed -n '2,4p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
CK=$(readlink -f "$1"); OV=$2; VAR=${3:-right_in,left_in,right_diag,left_diag}; PORT=${4:-8003}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/pan_env.sh"
IMG=${IMG:-hackjunction/policy:rc1-7a4168e-g}; PAN_RUNS=${PAN_RUNS:-/data/work/oliver/pan/runs}
SHA=$(git -C "$CK" rev-parse --short=8 HEAD 2>/dev/null || echo nogit)
OUT="$PAN_RUNS/${OV}__${SHA}__$(date +%m%d_%H%M%S)"; mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "[$(date +%T)] ckout=$CK ($SHA) overlay=$OV variants=$VAR image=$IMG port=$PORT"
NAME=pol-pan-$SHA; docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --rm --name "$NAME" --gpus all -p 127.0.0.1:$PORT:8000 -e AALTO_OVERLAY="$OV" ${POLICY_ENV:-} \
  -v "$CK/aaltonomous:/app/aaltonomous:ro" "$IMG" >/dev/null || exit 1
trap 'docker logs $NAME > "$OUT/policy_server.log" 2>&1; docker rm -f $NAME >/dev/null 2>&1' EXIT
sleep 5
cd "$PAN_TREE"
bash aaltonomous/cloud/aalto.sh py "$HERE/run_pan.py" --policy ws://localhost:$PORT --out "$OUT" --variants "$VAR" \
  ${PAN_ARGS:-}
echo "[$(date +%T)] done -> $OUT"
