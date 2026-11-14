#!/bin/bash
# run_arm.sh <sim n> <arm tag e.g. A2> <scenes csv|all> <seeds csv> [gpu] [cores]
# Starts the arm's policy container (image hackjunction/policy:<arm>-*, built by the repo's submission.Dockerfile via the
# matrix), runs our scenes against sim instance nv<n>, stops the container. Results: $NOVEL_ROOT/runs/<arm>/
. /data/work/novel/tools/slot.sh $1 $5 $6
ARM=$2; SCENES=$3; SEEDS=$4
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep "^hackjunction/policy:$ARM-" | head -1)
[[ -n $IMG ]] || { echo "no image for $ARM"; exit 3; }
OUT=$NOVEL_ROOT/runs/$ARM; FR=$NOVEL_ROOT/frames; mkdir -p $OUT $FR
NAME=novel-pol-$ARM-nv$NV; docker rm -f $NAME >/dev/null 2>&1
if (exec 3<>/dev/tcp/127.0.0.1/$POLPORT) 2>/dev/null; then echo "port $POLPORT busy"; exit 1; fi
docker run --rm -d --gpus device=$GPU --cpuset-cpus=$CORES --name $NAME -p 127.0.0.1:$POLPORT:8000 $IMG >/dev/null || exit 1
sleep 5; docker logs $NAME 2>&1 | grep "flags on" > $OUT/flags_nv$NV.txt
echo "[$(date -u +%T)] $ARM img=$IMG scenes=$SCENES seeds=$SEEDS"
cd $NOVEL_TREE
taskset -c $CORES bash aaltonomous/cloud/aalto.sh py $NOVEL_ROOT/tools/novel_eval.py --policy ws://localhost:$POLPORT --arm $ARM \
   --scenes $SCENES --seeds $SEEDS --out $OUT --container-name $NAME --frames-dir $FR 2>&1 | tee -a $OUT/run_nv$NV.log
docker logs $NAME > $OUT/policy_full_nv$NV.log 2>&1; docker rm -f $NAME >/dev/null 2>&1
echo "[$(date -u +%T)] $ARM done"
