#!/usr/bin/env bash
# Local cold-start check for the x86 submission image. Does not deploy.
#
#   bash tools/wendy/smoke.sh
#
# From the repo root: docker build -f tools/submission.Dockerfile, then
# `docker run --network none`. Waits until 127.0.0.1:8000 inside the container
# accepts a TCP connection, prints the seconds since `docker run`, then stops
# and removes the container.
#
# --network none has no published ports. Loopback still exists, which is the
# same check as Wendy tcpSocket readiness. --platform linux/amd64 keeps an
# Apple Silicon Docker host from building an arm64 image for this check.
# Override with TAG, NAME, or DEADLINE (seconds).
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

tag="${TAG:-hackjunction/policy:wendy-smoke}"
name="${NAME:-vaiski-policy-smoke}"
port=8000
deadline="${DEADLINE:-120}"

export DOCKER_BUILDKIT=1
docker build --platform linux/amd64 -f tools/submission.Dockerfile -t "$tag" .

cleanup() {
  docker stop "$name" >/dev/null 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker rm -f "$name" >/dev/null 2>&1 || true

start="$(date +%s)"
docker run -d --name "$name" --platform linux/amd64 --network none "$tag" >/dev/null

ready=0
while true; do
  now="$(date +%s)"
  elapsed="$((now - start))"
  if [ "$elapsed" -ge "$deadline" ]; then
    break
  fi
  if docker exec "$name" python3 -c "import socket; s=socket.create_connection(('127.0.0.1', ${port}), 1); s.close()" >/dev/null 2>&1; then
    ready=1
    break
  fi
  state="$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo missing)"
  if [ "$state" != "running" ]; then
    echo "cold-start FAILED: container state=${state} after ${elapsed}s" >&2
    docker logs "$name" >&2 || true
    exit 1
  fi
  if [ "$elapsed" -gt 0 ] && [ "$((elapsed % 10))" -eq 0 ]; then
    echo "waiting for port ${port} (${elapsed}s)" >&2
  fi
  sleep 1
done

now="$(date +%s)"
elapsed="$((now - start))"
if [ "$ready" != 1 ]; then
  echo "cold-start FAILED: port ${port} not accepting after ${elapsed}s (limit ${deadline}s)" >&2
  docker logs "$name" >&2 || true
  exit 1
fi
echo "cold-start: ${elapsed}s"
