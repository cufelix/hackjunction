#!/bin/sh
# Entrypoint of the single finals container (tools/brain/Dockerfile.policy_brain).
# 1. start llama-server (Qwen3-VL-8B Q8 + mmproj) on 127.0.0.1:8080 in the BACKGROUND, never waiting for it;
# 2. exec the policy server right away, so the container is healthy on :8000 whether or not the LLM comes up.
# The policy treats the brain as optional: connection refused / timeout / garbage = classical behaviour.
BRAIN_DIR="${BRAIN_DIR:-/opt/brain}"
LOG="${BRAIN_LOG:-/tmp/llama-server.log}"
MODEL="$BRAIN_DIR/models/Qwen3VL-8B-Instruct-Q8_0.gguf"
MMPROJ="$BRAIN_DIR/models/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"

if [ "${BRAIN_DISABLE:-0}" != "1" ] && [ -x "$BRAIN_DIR/bin/llama-server" ] && [ -f "$MODEL" ] && [ -f "$MMPROJ" ]; then
  export LD_LIBRARY_PATH="$BRAIN_DIR/lib:${LD_LIBRARY_PATH:-}"
  # a crashing / OOM-ing server must never take the container down: it is a detached child of this shell
  ( "$BRAIN_DIR/bin/llama-server" -m "$MODEL" --mmproj "$MMPROJ" --host 127.0.0.1 --port 8080 \
        -ngl 99 -c 8192 -np 1 >"$LOG" 2>&1 || echo "llama-server exited ($?): brain OFF, policy continues" >>"$LOG" ) &
else
  echo "[entrypoint] brain files missing or BRAIN_DISABLE=1: policy runs without the LLM (classical)" >&2
fi
exec python3 -m aaltonomous.baselines.classical.policy_server "$@"
