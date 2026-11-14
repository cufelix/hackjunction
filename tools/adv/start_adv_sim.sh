#!/usr/bin/env bash
# Start the adversarial sim (adv tree) as its own instance. Log: /tmp/adv_sim_<inst>.log. Stop: kill -INT <pid>
# printed below (NEVER aalto.sh stop / pkill: they act on the first sim found).
#   AALTO_DR=<profile file name in aaltonomous/sim/dr, e.g. adv_real_hard> picks the START profile; switch later with
#   tools/adv/set_profile.sh <name> (no restart). Default start profile = nominal.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/adv_env.sh"
cd "$ADV_TREE" || exit 1
export OMP_NUM_THREADS=1
LOG=/tmp/adv_sim_${AALTO_INSTANCE}.log
nohup setsid taskset -c "${ADV_CORES:-15-29}" bash aaltonomous/sim/run_sim.sh >"$LOG" 2>&1 </dev/null &
echo "adv sim instance $AALTO_INSTANCE starting from $ADV_TREE (log $LOG); wait for: AALTO_INSTANCE=$AALTO_INSTANCE bash $ADV_TREE/aaltonomous/cloud/aalto.sh status"
for _ in $(seq 150); do
  if bash aaltonomous/cloud/aalto.sh py -m aaltonomous.sim.scene_client ping >/dev/null 2>&1; then
    echo "ready; sim pid: $(pgrep -f "^python .*aalto_sim_main.py.*" | while read p; do tr '\0' '\n' </proc/$p/environ 2>/dev/null | grep -q "^AALTO_INSTANCE=$AALTO_INSTANCE$" && echo $p; done)"; exit 0; fi
  sleep 4
done; echo "not ready after 10 min, see $LOG"; exit 1
