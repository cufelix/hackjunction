#!/usr/bin/env bash
# Build, measure, optionally publish, and verify the policy submission image.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null) || {
  echo "handin: not inside a git repository" >&2
  exit 1
}
PYTHON="${PYTHON:-$REPO/../v16/bin/python}"
SUPPORT="$REPO/tools/handin_lib.py"
URDF="$REPO/third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
DOCKERFILE="$REPO/tools/submission.Dockerfile"
READY_TEXT="policy server 'classical-rung1' listening"
PLATFORM=linux/amd64
MULTI_ARCH=0
OUTPUT="$REPO/HANDIN.json"
REGISTRY_REF=
DEADLINE="${HANDIN_STARTUP_DEADLINE:-120}"

die() { echo "handin: $*" >&2; exit 1; }
usage() {
  cat <<'EOF'
usage: tools/handin.sh [--platform linux/amd64] [--multi-arch] [--output PATH] [REGISTRY/IMAGE:TAG]

With no image reference, build and measure locally and write HANDIN.json with a null digest.
With an image reference, push using Docker's existing login state, record the registry digest,
remove the locally built image, pull by digest, retag it freshly, and repeat the smoke test.
--multi-arch publishes a linux/amd64+linux/arm64 buildx manifest and requires an image reference.
EOF
}

while (($#)); do
  case "$1" in
    --platform) (($# >= 2)) || die "--platform needs a value"; PLATFORM="$2"; shift 2 ;;
    --multi-arch) MULTI_ARCH=1; shift ;;
    --output) (($# >= 2)) || die "--output needs a value"; OUTPUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) [ -z "$REGISTRY_REF" ] || die "only one registry image reference is accepted"; REGISTRY_REF="$1"; shift ;;
  esac
done

[ "$MULTI_ARCH" = 0 ] || [ -n "$REGISTRY_REF" ] || die "--multi-arch requires REGISTRY/IMAGE:TAG"
[[ "$PLATFORM" == linux/* ]] || die "--platform must be a linux platform (for example linux/amd64)"
[[ "$DEADLINE" =~ ^[1-9][0-9]*$ ]] || die "HANDIN_STARTUP_DEADLINE must be a positive whole number of seconds"
[ -x "$PYTHON" ] || die "Python is not executable: $PYTHON (set PYTHON=... to override)"
if [ -n "$REGISTRY_REF" ]; then
  "$PYTHON" "$SUPPORT" validate-push-reference "$REGISTRY_REF" >/dev/null \
    || die "registry destination must be HOST/REPOSITORY:UNIQUE_TAG"
fi
command -v docker >/dev/null 2>&1 || die "docker is not installed"
docker info >/dev/null 2>&1 || die "the Docker daemon is not reachable"
if [ "$MULTI_ARCH" = 1 ]; then
  docker buildx version >/dev/null 2>&1 || die "--multi-arch requires docker buildx"
fi

cd "$REPO"
DIRTY=$(git status --porcelain --untracked-files=all)
[ -z "$DIRTY" ] || {
  echo "$DIRTY" >&2
  die "git tree is not clean; commit or stash every source change before building the hand-in image"
}
URDF_SHA=$("$PYTHON" "$SUPPORT" require-urdf "$URDF") \
  || die "required URDF is missing, linked, empty, or has the wrong checksum"
[ -f "$DOCKERFILE" ] || die "missing $DOCKERFILE"

GIT_SHA=$(git rev-parse HEAD)
SHORT_SHA=${GIT_SHA:0:12}
LOCAL_TAG="g1-policy-handin:${SHORT_SHA}-$$"
FRESH_TAG="g1-policy-pulled:${SHORT_SHA}-$$"
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/g1-handin.XXXXXX")
ACTIVE_CONTAINER=
cleanup() {
  if [ -n "$ACTIVE_CONTAINER" ]; then
    docker rm -f "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
  fi
  rm -rf "$SCRATCH"
}
trap cleanup EXIT INT TERM

step() { echo; echo "== $*"; }

smoke_image() {
  local image=$1 label=$2 out=$3 name start_ns now_ns cold ready_line state
  name="g1-handin-${label}-${SHORT_SHA}-$$"
  docker rm -f "$name" >/dev/null 2>&1 || true
  ACTIVE_CONTAINER="$name"
  start_ns=$("$PYTHON" "$SUPPORT" now-ns)
  docker run -d --name "$name" --platform "$PLATFORM" --network none "$image" >/dev/null
  while true; do
    ready_line=$(docker logs "$name" 2>&1 | grep -F "$READY_TEXT" | tail -1 || true)
    if [ -n "$ready_line" ]; then
      break
    fi
    state=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || true)
    [ "$state" = running ] || {
      docker logs "$name" >&2 || true
      die "$label image exited before the ready line (state=${state:-missing})"
    }
    now_ns=$("$PYTHON" "$SUPPORT" now-ns)
    if ((now_ns - start_ns >= DEADLINE * 1000000000)); then
      docker logs "$name" >&2 || true
      die "$label image did not emit the ready line within ${DEADLINE}s"
    fi
    sleep 0.05
  done
  now_ns=$("$PYTHON" "$SUPPORT" now-ns)
  cold=$("$PYTHON" -c 'import sys; print(round((int(sys.argv[2])-int(sys.argv[1]))/1e9, 3))' "$start_ns" "$now_ns")
  if ! docker exec -i "$name" python3 - < "$REPO/tools/handin_smoke.py" > "$out.action.json"; then
    docker logs "$name" >&2 || true
    die "$label image failed the first-action smoke or the 27 Hz policy-rate check"
  fi
  docker exec "$name" python3 tools/flags.py --json > "$out.flags.json"
  "$PYTHON" "$SUPPORT" smoke-record --output "$out.smoke.json" --cold-start-s "$cold" \
    --action "$out.action.json" --ready-line "$ready_line"
  docker rm -f "$name" >/dev/null
  ACTIVE_CONTAINER=
  echo "  $label cold start: ${cold}s"
  "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(f"  first action: {d[\"first_action_latency_ms\"]:.3f} ms, {d[\"first_action_ticks\"]} ticks, {d[\"policy_hz_wall\"]:.1f} Hz")' "$out.smoke.json"
}

step "1/6 build clean $PLATFORM image"
export DOCKER_BUILDKIT=1
docker build --platform "$PLATFORM" -f "$DOCKERFILE" -t "$LOCAL_TAG" .
docker image inspect "$LOCAL_TAG" > "$SCRATCH/local.inspect.json"
IFS=$'\t' read -r IMAGE_ID SIZE_BYTES LAYER_COUNT < <("$PYTHON" "$SUPPORT" image-metrics < "$SCRATCH/local.inspect.json")
echo "  image ID: $IMAGE_ID"
echo "  size: $SIZE_BYTES bytes"
echo "  layers: $LAYER_COUNT"

step "2/6 network-none cold start and first action"
smoke_image "$LOCAL_TAG" local "$SCRATCH/local"

REGISTRY_DIGEST=
PULLED_ARGS=()
if [ -n "$REGISTRY_REF" ]; then
  step "3/6 push using Docker login state"
  if [ "$MULTI_ARCH" = 1 ]; then
    docker buildx build --platform linux/amd64,linux/arm64 -f "$DOCKERFILE" -t "$REGISTRY_REF" --push .
    docker buildx imagetools inspect "$REGISTRY_REF" > "$SCRATCH/registry.inspect.txt"
  else
    docker tag "$LOCAL_TAG" "$REGISTRY_REF"
    docker push "$REGISTRY_REF" | tee "$SCRATCH/registry.inspect.txt"
  fi
  REGISTRY_DIGEST=$("$PYTHON" "$SUPPORT" registry-digest < "$SCRATCH/registry.inspect.txt") \
    || die "push completed but the registry did not report a manifest digest"
  REGISTRY_REPO=$("$PYTHON" "$SUPPORT" registry-repository "$REGISTRY_REF")
  DIGEST_REF="$REGISTRY_REPO@$REGISTRY_DIGEST"
  echo "  registry digest: $DIGEST_REF"

  step "4/6 remove the build and pull by digest"
  docker image rm "$REGISTRY_REF" >/dev/null 2>&1 || true
  docker image rm "$DIGEST_REF" >/dev/null 2>&1 || true
  docker image rm "$LOCAL_TAG" >/dev/null 2>&1 || true
  if docker image inspect "$IMAGE_ID" >/dev/null 2>&1; then
    docker image rm "$IMAGE_ID" >/dev/null \
      || die "the built image has another local tag; remove it before a clean pull by digest"
  fi
  docker image inspect "$IMAGE_ID" >/dev/null 2>&1 \
    && die "could not remove the locally built image before the clean pull"
  docker pull --platform "$PLATFORM" "$DIGEST_REF"
  docker tag "$DIGEST_REF" "$FRESH_TAG"
  docker image inspect "$FRESH_TAG" > "$SCRATCH/pulled.inspect.json"
  IFS=$'\t' read -r PULLED_ID PULLED_SIZE PULLED_LAYERS < <("$PYTHON" "$SUPPORT" image-metrics < "$SCRATCH/pulled.inspect.json")

  step "5/6 smoke the digest-pulled image"
  smoke_image "$FRESH_TAG" pulled "$SCRATCH/pulled"
  cmp -s "$SCRATCH/local.flags.json" "$SCRATCH/pulled.flags.json" \
    || die "flags dump changed after pull by digest"
  PULLED_ARGS=(--pulled-smoke "$SCRATCH/pulled.smoke.json" --pulled-image-id "$PULLED_ID" --pulled-tag "$FRESH_TAG" \
               --pulled-size "$PULLED_SIZE" --pulled-layers "$PULLED_LAYERS")
  echo "  retained digest-pulled image as $FRESH_TAG"
else
  step "3-5/6 registry verification skipped"
  echo "  no registry image reference was supplied"
fi

step "6/6 write hand-in record"
WRITE_ARGS=(--output "$OUTPUT" --git-sha "$GIT_SHA" --urdf-sha256 "$URDF_SHA" --platform "$PLATFORM" \
            --registry-reference "$REGISTRY_REF" --digest "$REGISTRY_DIGEST" --image-id "$IMAGE_ID" \
            --size "$SIZE_BYTES" --layers "$LAYER_COUNT" --flags "$SCRATCH/local.flags.json" \
            --local-smoke "$SCRATCH/local.smoke.json")
[ "$MULTI_ARCH" = 0 ] || WRITE_ARGS+=(--multi-arch)
"$PYTHON" "$SUPPORT" write-record "${WRITE_ARGS[@]}" "${PULLED_ARGS[@]}"
echo "  wrote $OUTPUT"
if [ -n "$REGISTRY_DIGEST" ]; then
  echo "registry digest: $REGISTRY_REPO@$REGISTRY_DIGEST"
else
  echo "registry digest: unavailable (no registry push requested)"
fi
