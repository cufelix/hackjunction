#!/bin/bash
# start_sim.sh <n> [gpu] [cores] : start (or reuse) our sim instance nv<n> from the novel tree, with the demo camera
. /data/work/novel/tools/slot.sh "$@"
cd $NOVEL_TREE
simpid() { for p in $(pgrep -f "aalto_sim_main.py"); do tr '\0' '\n' </proc/$p/environ 2>/dev/null | grep -qx "AALTO_INSTANCE=nv$NV" && echo $p; done | head -1; }
ping() { bash aaltonomous/cloud/aalto.sh py -m aaltonomous.sim.scene_client ping >/dev/null 2>&1; }
if [[ -n "$(simpid)" ]] && ping; then echo "sim nv$NV already up"; exit 0; fi
export OMP_NUM_THREADS=1 TASK=Aalto-Sort-G129-Dex3-Joint CUDA_VISIBLE_DEVICES=$GPU NOVEL_CAM=1
mkdir -p $NOVEL_ROOT/simlogs
nohup setsid taskset -c $CORES bash aaltonomous/sim/run_sim.sh --camera_include front_camera,left_wrist_camera,right_wrist_camera,demo_camera > $NOVEL_ROOT/simlogs/sim_nv$NV.log 2>&1 </dev/null &
for _ in $(seq 200); do ping && { echo "sim nv$NV ready (gpu $GPU cores $CORES)"; exit 0; }; sleep 4; done
echo "sim nv$NV not ready"; tail -30 $NOVEL_ROOT/simlogs/sim_nv$NV.log; exit 1
