#!/usr/bin/env bash
# Start (or rebuild) the team Warren hub and sign this machine's CLI in as admin.
set -euo pipefail
cd "$(dirname "$0")"
HUB=https://felix-acer.tailb3b6fa.ts.net
if [ ! -f .env ]; then
  umask 077
  printf 'PUBLIC_URL=%s\nWARREN_SETUP_TOKEN=%s\nWARREN_ADMIN_TOKEN=%s\n' \
    "$HUB" "$(openssl rand -hex 32)" "wr_admin_$(openssl rand -hex 32)" > .env
fi
docker compose up -d --build
for _ in $(seq 30); do curl -sf -o /dev/null http://127.0.0.1:8791/.well-known/agent-card.json && break; sleep 1; done
WARREN_ADMIN_TOKEN=$(grep '^WARREN_ADMIN_TOKEN=' .env | cut -d= -f2-) warren login --hub "$HUB" --admin >/dev/null
echo "hub up: $HUB  (dashboard: $HUB/app)"
