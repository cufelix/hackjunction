#!/bin/bash
# waitsim.sh n... : wait (<=100 s per call) until all given sims answer ping
for i in $(seq 18); do
  ok=1
  for n in "$@"; do
    . /data/work/novel/tools/slot.sh $n >/dev/null 2>&1
    (cd $NOVEL_TREE && bash aaltonomous/cloud/aalto.sh py -m aaltonomous.sim.scene_client ping >/dev/null 2>&1) || ok=0
  done
  [ $ok = 1 ] && { echo "all up $(date -u +%T)"; exit 0; }
  sleep 4
done
echo "not yet $(date -u +%T)"
