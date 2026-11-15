#!/bin/bash
# run.sh <slot> <gpu> <cores> <name> <mode> <task> <seed> <trials> [image]
SLOT=$1; export SLOT_GPU=$2 SLOT_CORES=$3
NAME=$4; MODE=$5; TASK_=$6; SEED=$7; TR=$8; IMG=${9:-hackjunction/policy:rc4-890d387-auto}
. /data/work/matrix/bin/slotenv.sh $SLOT
R=/data/work/cine/out/$NAME; rm -rf $R; mkdir -p $R
TREE=/data/work/cine/tree
export OMP_NUM_THREADS=1 TASK=Aalto-Sort-G129-Dex3-Joint CUDA_VISIBLE_DEVICES=$GPU NOVEL_CAM=1 CINE_LIGHT=1 CINE_MODE=$MODE CINE_T=${CINE_T:-40}
cd $TREE
ping() { bash aaltonomous/cloud/aalto.sh py -m aaltonomous.sim.scene_client ping >/dev/null 2>&1; }
simpid() { for p in $(pgrep -f "aalto_sim_main.py"); do tr "\0" "\n" </proc/$p/environ 2>/dev/null | grep -qx "AALTO_INSTANCE=x$SLOT" && echo $p; done | head -1; }
if ! ping; then
  nohup setsid taskset -c $CORES bash aaltonomous/sim/run_sim.sh --camera_include front_camera,left_wrist_camera,right_wrist_camera,demo_camera > $R/sim.log 2>&1 </dev/null &
  for _ in $(seq 300); do ping && break; sleep 4; done
  echo "sim ready $(date -u +%T)" >> $R/state
  sleep ${WARM:-100}
fi
CN=cine-pol-$SLOT
docker rm -f $CN >/dev/null 2>&1
docker run --rm -d --cpuset-cpus=$CORES --name $CN -p 127.0.0.1:$POLPORT:8000 $IMG >/dev/null || { echo RUNFAIL >> $R/state; exit 1; }
sleep 8
cat > /tmp/rec_$SLOT.py <<PY
from aaltonomous.sim.scene_client import SceneClient
import sys
c = SceneClient('127.0.0.1')
print(c.call(cmd=sys.argv[1], dir=sys.argv[2]) if len(sys.argv) > 2 else c.call(cmd=sys.argv[1]))
PY
bash aaltonomous/cloud/aalto.sh py /tmp/rec_$SLOT.py rec_start $R/f >> $R/state 2>&1
echo "eval start $(date -u +%T)" >> $R/state
taskset -c $CORES bash aaltonomous/cloud/aalto.sh eval --policy ws://localhost:$POLPORT --task $TASK_ --seed $SEED --trials $TR --out $R/eval > $R/eval.txt 2>&1
echo "eval rc=$? end $(date -u +%T)" >> $R/state
bash aaltonomous/cloud/aalto.sh py /tmp/rec_$SLOT.py rec_stop >> $R/state 2>&1
docker logs $CN > $R/policy.log 2>&1; docker rm -f $CN >/dev/null 2>&1
echo DONE >> $R/state
[[ -n "$KEEPSIM" ]] && exit 0
p=$(simpid); [[ -n $p ]] && { kill -INT $p; sleep 15; kill -9 $p 2>/dev/null; }
for q in $(pgrep -f image_server_process); do tr "\0" "\n" </proc/$q/environ 2>/dev/null | grep -qx "AALTO_INSTANCE=x$SLOT" && kill -9 $q; done
