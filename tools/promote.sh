#!/usr/bin/env bash
# Promote a candidate overlay (rcN) to overlays/submission.yaml, the flag set the submission image bakes in.
#
#   bash tools/promote.sh <overlay> [--dry-run]
#
#   --dry-run   do everything in a temp copy of the repo, print the submission.yaml diff, change nothing here.
#   PYTHON=...  interpreter to use (default: python3).
#
# Steps: overlay loads -> gate (expected FAIL) -> equivalence baseline -> rewrite submission.yaml ->
# gate (must PASS, else submission.yaml is restored and exit 1) -> flags -> equivalence after -> RELEASE.md.
# Outputs: runs/release/<overlay>-<UTC stamp>/ (gitignored). Summary: RELEASE.md at the repo root.
set -euo pipefail

die() { echo "promote: $*" >&2; exit 1; }

OVERLAY=""; DRY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    -*) die "unknown option: $arg" ;;
    *) [ -z "$OVERLAY" ] || die "more than one overlay given"; OVERLAY="$arg" ;;
  esac
done
[ -n "$OVERLAY" ] || die "usage: bash tools/promote.sh <overlay> [--dry-run]"
[[ "$OVERLAY" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ && "$OVERLAY" != *..* ]] || die "bad overlay name: $OVERLAY"
[ "$OVERLAY" != submission ] || die "overlay must be an rcN, not submission itself"

PY="${PYTHON:-python3}"
"$PY" -c "import numpy, yaml" 2>/dev/null || die "$PY lacks numpy/yaml (release_gate needs them): run inside the policy image or set PYTHON=<venv python>"
export PYTHONDONTWRITEBYTECODE=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$REPO"
if [ "$DRY" = 1 ]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  cp -a "$REPO/." "$WORK/"
fi

CLS="$WORK/aaltonomous/baselines/classical"
OV="$CLS/overlays/$OVERLAY.yaml"
SUB="$CLS/overlays/submission.yaml"
[ -f "$OV" ] || die "no such overlay: aaltonomous/baselines/classical/overlays/$OVERLAY.yaml"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$WORK/runs/release/$OVERLAY-$STAMP"
mkdir -p "$OUT"
GIT_SHA="$(git -C "$WORK" rev-parse HEAD)"
git -C "$WORK" --no-optional-locks status --short > "$OUT/git-status-before.txt"
DIRTY="$(wc -l < "$OUT/git-status-before.txt" | tr -d ' ')"
[ "$DIRTY" = 0 ] && DIRTY_FLAG=no || DIRTY_FLAG=yes

echo "[1/7] overlay $OVERLAY loads"
(cd "$WORK" && "$PY" -c 'import sys; from aaltonomous.baselines.classical.config import load_config; load_config(overlay=sys.argv[1])' "$OV") >/dev/null \
  || die "overlay $OVERLAY does not load (error above)"

echo "[2/7] release gate before promotion (expected FAIL)"
GATE_BEFORE_RC=0
(cd "$WORK" && "$PY" tools/release_gate.py "$OVERLAY") > "$OUT/gate-before.txt" 2>&1 || GATE_BEFORE_RC=$?
grep -q '^RELEASE GATE \(PASS\|FAIL\)' "$OUT/gate-before.txt" || die "gate crashed, see $OUT/gate-before.txt"
cat "$OUT/gate-before.txt"

equiv() {  # equiv <out>: flags-off digest with the submission overlay selected through AALTO_OVERLAY
  (cd "$WORK" && PYTHONPATH="$WORK" AALTO_OVERLAY="$SUB" "$PY" tools/equivalence_check.py) > "$1" 2> "$1.err"
}
echo "[2b] equivalence baseline (submission.yaml as it is now)"
equiv "$OUT/equiv-before.txt" || die "equivalence_check failed (baseline), see $OUT"

echo "[3/7] rewrite submission.yaml from $OVERLAY"
cp -p "$SUB" "$OUT/submission.yaml.orig"
"$PY" "$WORK/tools/promote_overlay.py" "$OVERLAY" "$SUB" | tee "$OUT/changes.txt" || die "promote_overlay.py failed; submission.yaml unchanged"

echo "[4/7] release gate after promotion (must PASS)"
GATE_AFTER_RC=0
(cd "$WORK" && "$PY" tools/release_gate.py "$OVERLAY") > "$OUT/gate-after.txt" 2>&1 || GATE_AFTER_RC=$?
cat "$OUT/gate-after.txt"
if [ "$GATE_AFTER_RC" != 0 ] || grep -q '^CONFLICT' "$OUT/gate-after.txt"; then
  cp -p "$OUT/submission.yaml.orig" "$SUB"
  cmp -s "$OUT/submission.yaml.orig" "$SUB" || die "gate failed and the restore did not match the backup"
  die "gate still FAILS after promotion; submission.yaml restored from $OUT/submission.yaml.orig"
fi

echo "[5/7] flags of submission"
(cd "$WORK" && "$PY" tools/flags.py --overlay submission) > "$OUT/flags-submission.txt" || die "flags.py failed"
grep '^  ON ' "$OUT/flags-submission.txt" > "$OUT/flags-on.txt" || true
ON_COUNT="$(wc -l < "$OUT/flags-on.txt" | tr -d ' ')"
tail -1 "$OUT/flags-submission.txt"

echo "[6/7] equivalence after promotion (expected to differ from the baseline)"
equiv "$OUT/equiv-after.txt" || die "equivalence_check failed (after), see $OUT"
if cmp -s "$OUT/equiv-before.txt" "$OUT/equiv-after.txt"; then
  EQ_NOTE="identical to the baseline"
else
  EQ_NOTE="differs from the baseline (expected after a promotion; not a gate)"
fi
SHA_EQ_BEFORE="$(sha256sum "$OUT/equiv-before.txt" | cut -d' ' -f1)"
SHA_EQ_AFTER="$(sha256sum "$OUT/equiv-after.txt" | cut -d' ' -f1)"
echo "equivalence: $EQ_NOTE"

echo "[7/7] RELEASE.md"
(cd "$CLS" && find . -name '*.py' -not -path '*/__pycache__/*' | sort | xargs sha256sum) > "$OUT/py-sha256.txt"
SHA_SUB="$(sha256sum "$SUB" | cut -d' ' -f1)"
SHA_CFG="$(sha256sum "$CLS/config.yaml" | cut -d' ' -f1)"
SHA_OV="$(sha256sum "$OV" | cut -d' ' -f1)"
UTC_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_SHORT="${GIT_SHA:0:7}"
BT='```'
R="$WORK/RELEASE.md"
{
cat <<EOF
# Release: ${OVERLAY}

- Built (UTC): ${UTC_ISO}
- Git: ${GIT_SHA}, dirty before promotion: ${DIRTY_FLAG} (${DIRTY} path(s), listed in runs/release/${OVERLAY}-${STAMP}/git-status-before.txt)
- Overlay: overlays/${OVERLAY}.yaml, sha256 ${SHA_OV}
- Run output: runs/release/${OVERLAY}-${STAMP}/

## Flags ON in submission (${ON_COUNT})

${BT}
EOF
cat "$OUT/flags-on.txt"
cat <<EOF
${BT}

## Changed keys (submission before -> after)

${BT}
EOF
cat "$OUT/changes.txt"
cat <<EOF
${BT}

## Hashes

- overlays/submission.yaml: ${SHA_SUB}
- config.yaml: ${SHA_CFG}
- every .py under aaltonomous/baselines/classical:

${BT}
EOF
cat "$OUT/py-sha256.txt"
cat <<EOF
${BT}

## Release gate

Before promotion (expected FAIL):

${BT}
EOF
cat "$OUT/gate-before.txt"
cat <<EOF
${BT}

After promotion:

${BT}
EOF
cat "$OUT/gate-after.txt"
cat <<EOF
${BT}

## Equivalence (flags-off digest, run with AALTO_OVERLAY=submission)

- before promotion: sha256 ${SHA_EQ_BEFORE}
- after promotion: sha256 ${SHA_EQ_AFTER}
- ${EQ_NOTE}

## Build

${BT}
docker build -f tools/submission.Dockerfile -t hackjunction/policy:${OVERLAY}-${GIT_SHORT} .
docker run --rm --entrypoint python3 hackjunction/policy:${OVERLAY}-${GIT_SHORT} tools/flags.py
${BT}
EOF
} > "$R"

if [ "$DRY" = 1 ]; then
  echo "--- dry run: submission.yaml diff (repo at $REPO is not changed) ---"
  diff -u "$REPO/aaltonomous/baselines/classical/overlays/submission.yaml" "$SUB" || true
  echo "--- dry run: $ON_COUNT flags ON; RELEASE.md was written only to the temp copy ---"
else
  echo "done: $ON_COUNT flags ON; submission.yaml promoted from $OVERLAY; RELEASE.md at $R"
fi
