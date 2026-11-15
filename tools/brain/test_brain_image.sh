#!/usr/bin/env bash
# Acceptance test for the single-container finals image (tools/brain/Dockerfile.policy_brain). Run on box E.
#
#   tools/brain/test_brain_image.sh IMAGE PORT
#   e.g. tools/brain/test_brain_image.sh hackjunction/policy:finals-agent 8001
#
# Checks: (a) image size; (b) cold start to a healthy policy on PORT (runner shape: -p PORT:8000, no args, no volumes);
# (c) healthy with --network none; (e) brain latency p50/p95 over 20 decisions to 127.0.0.1:8080 inside the container;
# (f) llama.cpp commit and GGUF sha256 (printed); (d) kill llama-server inside the container, then policy health and
# act must still answer; finally `docker stop` must end the policy without SIGKILL.
#
# The policy has no HTTP health endpoint. Its health is the websocket handshake (aaltonomous/deploy/policy_server.py:
# serve() sends the metadata message on connect), plus a reset round trip in the act probe. Probes run inside the
# container with the image's own python3 (websockets, msgpack, numpy). The host needs only docker and python3.
#
# Env (optional): GPU_OFF=1 drops --gpus all; HEALTH_TIMEOUT_S=180; LLAMA_TIMEOUT_S=240; ACT_TIMEOUT_S=20;
#   P95_BUDGET_MS=1500 (= brain.request_timeout_s); PROMPTS_FILE=<jsonl, one recorded decision request
#   {"world","choices","point"} per line> to replace the 20 synthetic prompts.
# Exit 0 only if every check passes.
set -uo pipefail

if [ $# -ne 2 ]; then echo "usage: $0 IMAGE PORT" >&2; exit 2; fi
IMAGE=$1
PORT=$2
case "$PORT" in ''|*[!0-9]*) echo "PORT must be a number" >&2; exit 2 ;; esac
command -v docker >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1 || { echo "needs docker and python3" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image not found: $IMAGE" >&2; exit 2; }
if [ -n "${PROMPTS_FILE:-}" ] && [ ! -r "$PROMPTS_FILE" ]; then echo "PROMPTS_FILE not readable" >&2; exit 2; fi

HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-180}
LLAMA_TIMEOUT_S=${LLAMA_TIMEOUT_S:-240}
ACT_TIMEOUT_S=${ACT_TIMEOUT_S:-20}
P95_BUDGET_MS=${P95_BUDGET_MS:-1500}
N_PROMPTS=20
GPU_ARGS=(--gpus all)
[ "${GPU_OFF:-0}" = 1 ] && GPU_ARGS=()
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

RUN="brainchk-$$"
C_PUB="$RUN-pub"
C_NONET="$RUN-nonet"
FAILS=0
cleanup() { docker rm -f "$C_PUB" "$C_NONET" >/dev/null 2>&1; }
trap cleanup EXIT

pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; FAILS=$((FAILS + 1)); }
ms() { date +%s%3N; }
secs() { printf '%d.%03d' $(($1 / 1000)) $(($1 % 1000)); }

# ---- probes (run inside the container with its own python3) ----
read -r -d '' PROBE_HEALTH <<'PY'
from websockets.sync.client import connect
from aaltonomous.deploy import protocol as P
ws = connect("ws://127.0.0.1:8000", open_timeout=5, close_timeout=2, max_size=None, compression=None)
try:
    P.check_metadata(P.unpack(ws.recv(timeout=5)))
    print("HEALTH_OK")
finally:
    ws.close()
PY

read -r -d '' PROBE_ACT <<'PY'
import os, sys, time
import numpy as np
from websockets.sync.client import connect
from aaltonomous.deploy import protocol as P
from aaltonomous.interface.tasks import FIXED_INSTRUCTION
H, W = 480, 640
obs = {"head_rgb": np.zeros((H, W, 3), np.uint8), "head_depth": np.zeros((H, W), np.uint16),
       "left_wrist_rgb": np.zeros((H, W, 3), np.uint8), "right_wrist_rgb": np.zeros((H, W, 3), np.uint8),
       "arm_q": np.zeros(14, np.float32), "arm_dq": np.zeros(14, np.float32), "hand_q": np.zeros(14, np.float32),
       "instruction": FIXED_INSTRUCTION, "t": 0.0}
ws = connect("ws://127.0.0.1:8000", open_timeout=5, close_timeout=2, max_size=None, compression=None)
try:
    P.check_metadata(P.unpack(ws.recv(timeout=5)))
    ws.send(P.pack({"type": "reset", "task": "sort_fixed", "instruction": FIXED_INSTRUCTION, "seed": 0}))
    r = P.unpack(ws.recv(timeout=20))
    if r.get("type") != "reset_ok":
        print("reset reply:", r.get("type"), str(r.get("message"))[:200]); sys.exit(1)
    t0 = time.monotonic()
    ws.send(P.pack({"type": "act", "obs": obs}))
    r = P.unpack(ws.recv(timeout=float(os.environ.get("ACT_TIMEOUT_S", "20"))))
    dt = (time.monotonic() - t0) * 1000
    if r.get("type") != "action":
        print("act reply:", r.get("type"), str(r.get("message"))[:200]); sys.exit(1)
    arm = np.asarray(r["action"]["arm_q_target"], float)
    if arm.shape[-1] != 14 or not np.isfinite(arm).all():
        print("act reply has a bad arm_q_target", arm.shape); sys.exit(1)
    print(f"ACT_OK {dt:.0f} ms")
finally:
    ws.close()
PY

read -r -d '' PROBE_LLAMA_READY <<'PY'
import sys, urllib.request
sys.exit(0 if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3).status == 200 else 1)
PY

read -r -d '' PROBE_LLAMA_DOWN <<'PY'
import sys, urllib.error, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
    print("STILL_UP"); sys.exit(1)
except urllib.error.HTTPError:
    print("STILL_UP (unhealthy)"); sys.exit(1)
except OSError:
    print("LLAMA_DOWN"); sys.exit(0)
PY

read -r -d '' PROBE_KILL_FALLBACK <<'PY'
import os, signal, sys
hit = []
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")
    except OSError:
        continue
    if b"llama-server" in cmd and int(pid) != os.getpid():
        os.kill(int(pid), signal.SIGTERM); hit.append(pid)
print("killed", hit); sys.exit(0 if hit else 1)
PY

read -r -d '' PROBE_BRAIN <<'PY'
import json, math, os, sys, time
from aaltonomous.baselines.classical.brain.core import LlamaCppHTTPBackend
from aaltonomous.baselines.classical.brain.registry import allowed_calls
N, SRC = 20, "/tmp/brain_prompts.jsonl"
COLOURS = ["red", "yellow", "blue", "orange", "purple"]
def synth(i):
    ids = [f"o{j}" for j in range(1 + i % 5)]
    world = {"objects": [{"id": o, "color": COLOURS[j % 5]} for j, o in enumerate(ids)]}
    if i % 3 == 2:
        return {"world": world, "choices": allowed_calls("after_place", {"obj_id": ids[0]}), "point": "after_place"}
    return {"world": world, "choices": allowed_calls("after_look", {"candidates": ids}), "point": "after_look"}
real = os.path.exists(SRC)
reqs = [json.loads(l) for l in open(SRC) if l.strip()][:N] if real else [synth(i) for i in range(N)]
if len(reqs) < N:
    print("need", N, "prompts, got", len(reqs)); sys.exit(1)
be = LlamaCppHTTPBackend("http://127.0.0.1:8080/completion", timeout_s=10.0)   # production decide() path, n_predict 32, ChatML
def ask(req):
    t = time.perf_counter()
    c = int(json.loads(be.decide(json.dumps(req)))["choice"])
    if not 0 <= c < len(req["choices"]):
        raise ValueError(f"choice {c} out of range")
    return (time.perf_counter() - t) * 1000
try:
    ask(reqs[0])            # warm-up, not timed
except Exception as e:
    print("warm-up failed:", type(e).__name__, str(e)[:200]); sys.exit(1)
ms, errors = [], []
for i, r in enumerate(reqs):
    try:
        ms.append(ask(r))
    except Exception as e:
        errors.append(f"{i}: {type(e).__name__}: {str(e)[:120]}")
def pct(v, q):
    s = sorted(v)
    return round(s[min(len(s) - 1, math.ceil(q * len(s)) - 1)], 1) if s else None
print("BRAIN_JSON " + json.dumps({"ok": len(ms), "n": N, "p50_ms": pct(ms, .5), "p95_ms": pct(ms, .95),
                                  "max_ms": pct(ms, 1.0), "source": "recorded" if real else "synthetic",
                                  "errors": errors[:5]}))
PY

read -r -d '' PROBE_ARTIFACTS <<'PY'
import glob, hashlib, os, re, shutil, subprocess
exe = next((p for p in ["/opt/brain/bin/llama-server", "/usr/local/bin/llama-server", shutil.which("llama-server") or ""]
            if p and os.path.exists(p)), None)
print("LLAMA_BIN", exe)
commit = "unknown"
if exe:
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    txt = out.stdout + out.stderr
    line = next((l for l in txt.splitlines() if "version" in l.lower()), "")
    print("LLAMA_VERSION", line.strip() or "unknown")
    m = re.search(r"\(([0-9a-f]{7,40})\)", line)
    commit = m.group(1) if m else "unknown"
print("LLAMA_COMMIT", commit)
ggufs = sorted(glob.glob("/opt/brain/models/*.gguf"))
print("GGUF_COUNT", len(ggufs))
for p in ggufs:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    print("SHA256", h.hexdigest(), os.path.getsize(p), os.path.basename(p))
PY

# wait_probe NAME LIMIT_S T0_MS PROBE_TEXT -> 0 once the probe prints success (exit 0); sets WAIT_MS
wait_probe() {
  local name=$1 limit=$2 t0=$3 probe=$4 now
  while :; do
    if timeout 30 docker exec -i "$name" python3 - <<<"$probe" >/dev/null 2>&1; then
      WAIT_MS=$(( $(ms) - t0 )); return 0
    fi
    [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = true ] || return 1
    now=$(ms)
    [ $(( now - t0 )) -lt $(( limit * 1000 )) ] || return 1
    sleep 1
  done
}

run_act() {  # NAME LABEL
  local out rc
  out=$(timeout 90 docker exec -i "$1" python3 - <<<"$PROBE_ACT" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then pass "$2: act returns an action ($(printf '%s' "$out" | tail -1))"
  else fail "$2: act did not return an action: $(printf '%s' "$out" | tail -3 | tr '\n' ' ')"; fi
}

echo "IMAGE $IMAGE"

# (a) image size
SIZE=$(docker image inspect -f '{{.Size}}' "$IMAGE")
echo "IMAGE_ID $(docker image inspect -f '{{.Id}}' "$IMAGE")"
echo "IMAGE_SIZE_BYTES $SIZE ($((SIZE / 1048576)) MiB)"
if [ "${SIZE:-0}" -gt 0 ] 2>/dev/null; then pass "(a) image size recorded"; else fail "(a) image size unreadable"; fi

# (b) cold start, runner shape
T0=$(ms)
if ! docker run -d --name "$C_PUB" "${GPU_ARGS[@]}" -p "$PORT:8000" "$IMAGE" >/dev/null; then
  fail "(b) docker run failed"; exit 1
fi
if wait_probe "$C_PUB" "$HEALTH_TIMEOUT_S" "$T0" "$PROBE_HEALTH"; then
  pass "(b) cold start: policy health OK after $(secs "$WAIT_MS") s (port $PORT)"
else
  fail "(b) policy never reported healthy within ${HEALTH_TIMEOUT_S} s"
  docker logs --tail 40 "$C_PUB" 2>&1 | sed 's/^/  | /'
  exit 1
fi
run_act "$C_PUB" "(b) pre-kill"

# (c) --network none
T0=$(ms)
if docker run -d --name "$C_NONET" "${GPU_ARGS[@]}" --network none "$IMAGE" >/dev/null \
   && wait_probe "$C_NONET" "$HEALTH_TIMEOUT_S" "$T0" "$PROBE_HEALTH"; then
  pass "(c) --network none: health OK after $(secs "$WAIT_MS") s"
else
  fail "(c) --network none: health not OK"
fi
docker rm -f "$C_NONET" >/dev/null 2>&1   # free the GPU before the brain runs

# llama-server must be up before the brain checks
T0=$(ms)
if wait_probe "$C_PUB" "$LLAMA_TIMEOUT_S" "$T0" "$PROBE_LLAMA_READY"; then
  pass "llama-server ready on 127.0.0.1:8080 after $(secs "$WAIT_MS") s"
else
  fail "llama-server not ready within ${LLAMA_TIMEOUT_S} s (brain cannot be measured)"
fi

# (e) brain latency
if [ -n "${PROMPTS_FILE:-}" ]; then
  docker cp "$PROMPTS_FILE" "$C_PUB:/tmp/brain_prompts.jsonl" >/dev/null || fail "(e) could not copy PROMPTS_FILE"
fi
BRAIN_OUT=$(timeout 900 docker exec -i "$C_PUB" python3 - <<<"$PROBE_BRAIN" 2>&1)
BRAIN_JSON=$(printf '%s\n' "$BRAIN_OUT" | sed -n 's/^BRAIN_JSON //p' | tail -1)
read -r B_OK B_P50 B_P95 B_MAX B_SRC <<<"$(printf '%s' "$BRAIN_JSON" | python3 -c '
import json, sys
d = json.loads(sys.stdin.read() or "{}")
print(d.get("ok", 0), d.get("p50_ms", "na"), d.get("p95_ms", "na"), d.get("max_ms", "na"), d.get("source", "na"))')"
echo "BRAIN prompts=$B_SRC ok=$B_OK/$N_PROMPTS p50_ms=$B_P50 p95_ms=$B_P95 max_ms=$B_MAX budget_p95_ms=$P95_BUDGET_MS"
printf '%s\n' "$BRAIN_OUT" | sed -n 's/^BRAIN_JSON .*"errors": \(.*\)}$/  errors: \1/p' | tail -1
if [ "$B_OK" = "$N_PROMPTS" ] && python3 -c "import sys; sys.exit(0 if float('$B_P95') <= $P95_BUDGET_MS else 1)" 2>/dev/null; then
  pass "(e) brain: $N_PROMPTS/$N_PROMPTS valid decisions, p95 ${B_P95} ms <= ${P95_BUDGET_MS} ms"
else
  fail "(e) brain: ok=$B_OK/$N_PROMPTS p95=$B_P95 ms (budget ${P95_BUDGET_MS} ms)"
fi

# (f) versions and checksums (printed; a missing GGUF or binary fails)
ART=$(timeout 900 docker exec -i "$C_PUB" python3 - <<<"$PROBE_ARTIFACTS" 2>&1)
printf '%s\n' "$ART"
GGUF_N=$(printf '%s\n' "$ART" | sed -n 's/^GGUF_COUNT //p')
if [ -d "$ROOT/tools/brain/llama.cpp/.git" ] || [ -f "$ROOT/tools/brain/llama.cpp/.git" ]; then
  echo "HOST_LLAMA_SRC_COMMIT $(git -C "$ROOT/tools/brain/llama.cpp" rev-parse HEAD 2>/dev/null)"
fi
if printf '%s\n' "$ART" | grep -q '^LLAMA_BIN /' && [ "${GGUF_N:-0}" -ge 2 ] 2>/dev/null; then
  pass "(f) llama-server and $GGUF_N GGUF files present in the image"
else
  fail "(f) llama-server or GGUF files missing in the image"
fi

# (d) kill llama-server, policy must keep answering
if docker exec "$C_PUB" pkill -f llama-server >/dev/null 2>&1; then
  echo "killed llama-server with pkill"
else
  KILL_OUT=$(docker exec -i "$C_PUB" python3 - <<<"$PROBE_KILL_FALLBACK" 2>&1)
  echo "pkill unavailable or no match; python fallback: $KILL_OUT"
fi
sleep 3
DOWN=$(docker exec -i "$C_PUB" python3 - <<<"$PROBE_LLAMA_DOWN" 2>&1 | tail -1)
echo "LLM state after kill: $DOWN"
[ "$DOWN" = LLAMA_DOWN ] || fail "(d) llama-server still answering after kill"

T0=$(ms)
if wait_probe "$C_PUB" 30 "$T0" "$PROBE_HEALTH"; then
  pass "(d) policy health still OK with the LLM dead"
else
  fail "(d) policy health lost after llama-server was killed"
fi
run_act "$C_PUB" "(d) post-kill"

# signal handling: docker stop must end the policy without SIGKILL (exit 137)
T0=$(ms)
docker stop -t 15 "$C_PUB" >/dev/null 2>&1
STOP_MS=$(( $(ms) - T0 ))
CODE=$(docker inspect -f '{{.State.ExitCode}}' "$C_PUB" 2>/dev/null)
if [ "$CODE" != 137 ]; then
  pass "docker stop: exit $CODE after $(secs "$STOP_MS") s"
else
  fail "docker stop: SIGTERM ignored (PID 1 without a handler), SIGKILL after $(secs "$STOP_MS") s"
fi

echo "RESULT: $([ "$FAILS" -eq 0 ] && echo PASS || echo "FAIL ($FAILS checks)")"
[ "$FAILS" -eq 0 ]
