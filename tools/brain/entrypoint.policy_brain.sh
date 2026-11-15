#!/bin/sh
# Entrypoint of the single finals container (tools/brain/Dockerfile.policy_brain).
# 1. if BRAIN_ENABLED=1 (baked at build time, same value as the overlay's brain.enabled), start llama-server
#    (Qwen3-VL-8B Q8 + mmproj) on 127.0.0.1:8080 in the BACKGROUND, never waiting for it; its output goes to the
#    container's stderr with a "[llama] " prefix, so it shows in `docker logs`;
# 2. exec the command (default: the policy server on :8000) right away, so the container is healthy whether or not
#    the LLM comes up. The policy treats the brain as optional: connection refused / timeout / garbage = classical.
BRAIN_DIR="${BRAIN_DIR:-/opt/brain}"
MODEL="$BRAIN_DIR/models/Qwen3VL-8B-Instruct-Q8_0.gguf"
MMPROJ="$BRAIN_DIR/models/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"

if [ "${BRAIN_ENABLED:-0}" = "1" ] && [ "${BRAIN_DISABLE:-0}" != "1" ] && [ -x "$BRAIN_DIR/bin/llama-server" ] \
   && [ -f "$MODEL" ] && [ -f "$MMPROJ" ]; then
  export LD_LIBRARY_PATH="$BRAIN_DIR/lib:${LD_LIBRARY_PATH:-}"
  # a crashing / OOM-ing server must never take the container down: it is a detached child of this shell
  ( "$BRAIN_DIR/bin/llama-server" -m "$MODEL" --mmproj "$MMPROJ" --host 127.0.0.1 --port 8080 \
        -ngl 99 -c 8192 -np 1 2>&1 | sed -u 's/^/[llama] /' >&2
    echo "[llama] llama-server exited: brain OFF, policy continues" >&2 ) &
else
  echo "[entrypoint] brain OFF (BRAIN_ENABLED=${BRAIN_ENABLED:-0}) or files missing: policy runs without the LLM (classical)" >&2
fi
exec "$@"
