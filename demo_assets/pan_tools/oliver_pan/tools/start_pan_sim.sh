#!/usr/bin/env bash
# Start the pan sim (default instance) from the pan tree. Log /tmp/pan_sim.log. Stop: kill -INT <pid printed>.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/pan_env.sh"
cd "$PAN_TREE" || exit 1
if pgrep -f "^python .*aalto_sim_main.py" >/dev/null; then echo "a sim is already running on this box; stop yours first"; exit 1; fi
export OMP_NUM_THREADS=1
nohup setsid bash aaltonomous/sim/run_sim.sh >/tmp/pan_sim.log 2>&1 </dev/null &
for _ in $(seq 180); do
  if bash aaltonomous/cloud/aalto.sh py -m aaltonomous.sim.scene_client ping >/dev/null 2>&1; then
    echo "pan sim ready; pid $(pgrep -f '^python .*aalto_sim_main.py')"; exit 0; fi
  sleep 4
done; echo "not ready after 12 min, see /tmp/pan_sim.log"; exit 1
