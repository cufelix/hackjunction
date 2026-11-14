#!/usr/bin/env bash
# Sekvenční A/B na sort_language proti JIŽ BĚŽÍCÍMU simulátoru (skript sim nikdy nestartuje/nezastavuje):
#   A = classical server s instruction.resolve=false, B = totéž s instruction.resolve=true.
# Flag se mění přes bind-mount upraveného config.yaml (load_config čte config.yaml z image; policy_server
# nemá env/CLI hook na instruction.resolve). Kód ani nainstalovaný repo se needituje.
# LANG_AB_DRY=1 = jen vypíše příkazy, nic nespouští.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AALTO="$ROOT/aaltonomous/cloud/aalto.sh"
SEED="${LANG_AB_SEED:-1000}"
TRIALS="${LANG_AB_TRIALS:-10}"
SETTLE="${LANG_AB_SETTLE:-3}"
PORT="${LANG_AB_PORT:-8000}"
IMAGE="${LANG_AB_IMAGE:-aalto/resolver-ab:local}"
RUN_ID="${LANG_AB_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${LANG_AB_OUT:-$ROOT/runs/lang_ab/$RUN_ID}"
CFG_SRC="$ROOT/aaltonomous/baselines/classical/config.yaml"
DRY="${LANG_AB_DRY:-0}"
CONTAINER=""

run() { if [[ "$DRY" == 1 ]]; then echo "DRY: $*"; else "$@"; fi; }

grep -qE '^  resolve: false$' "$CFG_SRC" || { echo "config.yaml nemá 'instruction: resolve: false'." >&2; exit 1; }
URDF_REL="third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
[[ -f "$ROOT/$URDF_REL" ]] || { echo "Chybí $URDF_REL (zkopíruj z /data/aaltonomous-humanoid)." >&2; exit 1; }
mkdir -p "$OUT_ROOT"
sed 's/^  resolve: false$/  resolve: true/' "$CFG_SRC" > "$OUT_ROOT/config_resolve_true.yaml"
cp "$CFG_SRC" "$OUT_ROOT/config_resolve_false.yaml"
diff "$OUT_ROOT/config_resolve_false.yaml" "$OUT_ROOT/config_resolve_true.yaml" || true

if [[ "$DRY" != 1 ]]; then
    bash "$AALTO" py -m aaltonomous.sim.scene_client ping >"$OUT_ROOT/sim_ping_before.json" \
        || { echo "Scene service neodpovídá; simulátor musí běžet." >&2; exit 1; }
    python3 - "$PORT" <<'PY' && { echo "Port $PORT je obsazen." >&2; exit 1; } || true
import socket, sys
with socket.socket() as s:
    s.settimeout(0.3)
    raise SystemExit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
fi

run docker build -f "$ROOT/tools/baseline.step0.Dockerfile" -t "$IMAGE" "$ROOT"

cleanup() { [[ -n "$CONTAINER" ]] && docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

run_one() {  # $1 = false|true
    local flag="$1" team="lang_ab_resolve_$1"
    CONTAINER="aalto-langab-$flag-${RUN_ID//[^a-zA-Z0-9_.-]/-}"
    echo "== Run resolve=$flag =="
    run docker run --rm -d --name "$CONTAINER" -p "127.0.0.1:$PORT:8000" \
        -v "$OUT_ROOT/config_resolve_$flag.yaml:/app/aaltonomous/baselines/classical/config.yaml:ro" "$IMAGE"
    run bash "$AALTO" py -m aaltonomous.deploy.replay --synthetic 10 --policy "ws://localhost:$PORT" --min-hz 27
    set +e
    run bash "$AALTO" eval --policy "ws://localhost:$PORT" --team "$team" --task sort_language \
        --trials "$TRIALS" --seed "$SEED" --time-limit 120 --settle "$SETTLE" \
        --gate-max-clips 0 --gate-min-hz 27 --out "$OUT_ROOT"
    local rc=$?
    set -e
    [[ "$DRY" == 1 ]] || { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
    echo "stop $CONTAINER"
    CONTAINER=""
    if [[ $rc -eq 2 ]]; then echo "Gate selhal (rc=2); data platná, pokračuji." >&2
    elif [[ $rc -ne 0 ]]; then echo "eval rc=$rc, končím." >&2; exit "$rc"; fi
}

run_one false
run_one true
echo "Výstupy: $OUT_ROOT"
