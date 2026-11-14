#!/bin/bash
# stop_sim.sh <n> : stop ONLY our instance nv<n>
. /data/work/novel/tools/slot.sh "$@"
for p in $(pgrep -f "aalto_sim_main.py"); do tr '\0' '\n' </proc/$p/environ 2>/dev/null | grep -qx "AALTO_INSTANCE=nv$NV" && { kill -INT $p; for _ in $(seq 40); do kill -0 $p 2>/dev/null || break; sleep 1; done; kill -9 $p 2>/dev/null; }; done
for q in $(pgrep -f "image_server_process"); do tr '\0' '\n' </proc/$q/environ 2>/dev/null | grep -qx "AALTO_INSTANCE=nv$NV" && kill -9 $q; done
echo stopped nv$NV
