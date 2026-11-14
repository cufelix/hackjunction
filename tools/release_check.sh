#!/usr/bin/env bash
# R24 release check for candidate images. usage: tools/release_check.sh <image> [port]   (run on the GPU box)
#  (2) no-env docker run, synthetic websocket replay, resolved profile log lines, flag-table line
#  (4, brain images) time to first valid action, peak VRAM, llama-server kill -> still acting
# Offline checks: (1) python3 tools/equivalence_check.py in two trees and diff (flags off == c78d0fe);
#                 (3) python3 tools/dump_sim_cfg.py <overlay> in two trees and diff (only intended sim deltas).
set -u
IMG=${1:?image}; PORT=${2:-8199}; N=rc-$$
docker rm -f "$N" >/dev/null 2>&1
t0=$(date +%s.%N)
docker run -d --name "$N" --gpus all -p "$PORT":8000 "$IMG" >/dev/null || exit 1
for i in $(seq 1 200); do (echo > /dev/tcp/127.0.0.1/"$PORT") 2>/dev/null && break; sleep 0.1; done
t1=$(date +%s.%N)
python3 -m aaltonomous.deploy.replay --synthetic 20 --policy ws://localhost:"$PORT" 2>&1 | tail -1 | cut -c1-200
t2=$(date +%s.%N)
echo "port_open_s=$(echo "$t1 - $t0" | bc)  replay_done_s=$(echo "$t2 - $t0" | bc)"
docker logs "$N" 2>&1 | grep -E "\[profile\]|effective config|Traceback|FLAG MISMATCH" | cut -c1-200
if docker exec "$N" sh -c 'pgrep llama-server' >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  docker exec "$N" sh -c 'pkill llama-server'; sleep 1
  python3 -m aaltonomous.deploy.replay --synthetic 10 --policy ws://localhost:"$PORT" 2>&1 | tail -1 | cut -c1-200
fi
docker rm -f "$N" >/dev/null
