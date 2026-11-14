#!/usr/bin/env bash
# Promote a candidate overlay to the baked submission overlay at the release freeze.
#
#   tools/promote.sh [--dry-run] [--no-equiv] <overlay>      e.g. tools/promote.sh rc1
#
#   --dry-run   run every step on a temp copy of the tree; nothing in the repo changes.
#               The output dir (with RELEASE.md) is printed at the end.
#   --no-equiv  skip tools/equivalence_check.py; RELEASE.md says SKIPPED.
#
# Environment: PYTHON=<interpreter with numpy and pyyaml> (default: python3).
#
# A real run writes overlays/submission.yaml, RELEASE.md, RELEASE.flags.json and RELEASE.equivalence.txt.
# It commits nothing and prints the git add / git commit command for you to run.
# It stops on the first failure. A real run that fails restores those four files to their pre-run state.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OVL_DIR=aaltonomous/baselines/classical/overlays
SUB_REL="$OVL_DIR/submission.yaml"
# The only paths that may be modified in the tree. A previous run leaves them behind, so re-running is allowed.
WRITE_SET=("$SUB_REL" RELEASE.md RELEASE.flags.json RELEASE.equivalence.txt)

die()   { echo "promote: $*" >&2; exit 1; }
"$PYTHON" -c "import numpy, yaml" 2>/dev/null || die "$PYTHON lacks numpy/pyyaml (release_gate needs them): run inside the policy image or set PYTHON=<venv python>"
step()  { echo; echo "== $*"; }
usage() { echo "usage: tools/promote.sh [--dry-run] [--no-equiv] <overlay>" >&2; exit 2; }

DRY=0; NO_EQUIV=0; OVERLAY=
for arg in "$@"; do
  case "$arg" in
    --dry-run)  DRY=1 ;;
    --no-equiv) NO_EQUIV=1 ;;
    -*)         usage ;;
    *)          [ -z "$OVERLAY" ] || usage; OVERLAY="$arg" ;;
  esac
done
[ -n "$OVERLAY" ] || usage
[[ "$OVERLAY" =~ ^[A-Za-z0-9_-]+$ ]] || die "overlay name may only hold letters, digits, '_' and '-': '$OVERLAY'"
[ "$OVERLAY" != submission ] || die "'submission' is the target itself; name a candidate such as rc1"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null) || die "not inside a git repository"
SHA=$(git -C "$REPO" rev-parse --short HEAD)
SHA_FULL=$(git -C "$REPO" rev-parse HEAD)
DATE=$(date -u +%F)
SRC_REL="$OVL_DIR/$OVERLAY.yaml"
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/promote.XXXXXX")
SNAP=
key() { printf '%s' "$1" | tr '/' '_'; }
restore() {
  local f k
  for f in "${WRITE_SET[@]}"; do
    k=$(key "$f")
    if [ -e "$SCRATCH/snap/$k.absent" ]; then rm -f "$REPO/$f"; else cp -p "$SCRATCH/snap/$k" "$REPO/$f"; fi
  done
}
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ -n "$SNAP" ]; then
    echo "promote: stopped; restoring ${WRITE_SET[*]} to their state before this run" >&2
    restore
  fi
  rm -rf "$SCRATCH"
  exit "$rc"
}
trap on_exit EXIT

step "1/8 checking the git tree"
[ -f "$REPO/$SRC_REL" ] || die "no overlay '$OVERLAY': $SRC_REL does not exist"
bad=()
while IFS= read -r line; do
  [ -n "$line" ] || continue
  path=${line:3}
  allowed=0
  for w in "${WRITE_SET[@]}"; do [ "$path" = "$w" ] && allowed=1; done
  [ "$allowed" = 1 ] || bad+=("$line")
done < <(git -C "$REPO" status --porcelain --untracked-files=all)
if [ "${#bad[@]}" -gt 0 ]; then
  printf '  %s\n' "${bad[@]}" >&2
  die "git tree is dirty (files listed above). Commit or stash them; only ${WRITE_SET[*]} may differ."
fi
echo "  clean (HEAD $SHA)"

if [ "$DRY" = 1 ]; then
  OUT=$(mktemp -d "${TMPDIR:-/tmp}/promote-dry.XXXXXX")
  ROOT="$OUT/tree"
  mkdir "$ROOT"
  tar -C "$REPO" --exclude=.git --exclude=__pycache__ -cf - . | tar -C "$ROOT" -xf -
  echo "  dry run: working on a copy at $ROOT; $REPO is not changed"
else
  ROOT="$REPO"; OUT="$REPO"
  mkdir "$SCRATCH/snap"
  for f in "${WRITE_SET[@]}"; do
    k=$(key "$f")
    if [ -e "$REPO/$f" ]; then cp -p "$REPO/$f" "$SCRATCH/snap/$k"; else : > "$SCRATCH/snap/$k.absent"; fi
  done
  SNAP=1
fi

step "2/8 writing $SUB_REL from $OVERLAY"
SRC="$ROOT/$SRC_REL"
if grep -q '^placeholder:' "$SRC"; then
  die "$SRC_REL is a placeholder: fill in its values and drop 'placeholder: true' first"
fi
if grep -q '^name:' "$SRC"; then
  stray=$(sed -n '/^name:/q;p' "$SRC" | grep -v -E '^(#|[[:space:]]*$)' || true)
  [ -z "$stray" ] || die "$SRC_REL has top-level content before 'name:': $stray"
  body=$(awk 'seen{print} /^name:/{seen=1}' "$SRC")
else
  body=$(cat "$SRC")
fi
{
  echo "# Submission flag overlay, baked into the image by tools/submission.Dockerfile."
  echo "# Written by tools/promote.sh. To change it, promote a candidate overlay again."
  echo "# promoted from $OVERLAY @ $SHA on $DATE"
  echo "name: submission"
  printf '%s\n' "$body"
} > "$ROOT/$SUB_REL.tmp"
mv "$ROOT/$SUB_REL.tmp" "$ROOT/$SUB_REL"
echo "  wrote $SUB_REL"

step "3/8 release gate: submission must carry exactly the flags of $OVERLAY"
if ! GATE=$(cd "$ROOT" && "$PYTHON" tools/release_gate.py "$OVERLAY" 2>&1); then
  echo "$GATE"
  die "release gate FAILED: $SUB_REL does not match $OVERLAY, or $OVERLAY has a flag conflict (see above)"
fi
echo "$GATE"

step "4/8 reviewed finals allow-list"
if [[ "$OVERLAY" == finals* ]]; then
  if ! CHECK=$(cd "$ROOT" && "$PYTHON" tools/check_release_flags.py --overlay submission 2>&1); then
    echo "$CHECK"
    die "reviewed finals allow-list FAILED"
  fi
  echo "$CHECK"
else
  echo "  skipped (candidate '$OVERLAY' is not a finals overlay)"
fi

step "5/8 flags.py --overlay submission --json"
if ! (cd "$ROOT" && "$PYTHON" tools/flags.py --overlay submission --json) > "$SCRATCH/flags.json" 2> "$SCRATCH/flags.err"; then
  cat "$SCRATCH/flags.err" >&2
  die "tools/flags.py failed"
fi
sed "s#$ROOT#.#g" "$SCRATCH/flags.json" > "$OUT/RELEASE.flags.json"
echo "  saved $OUT/RELEASE.flags.json"

step "6/8 equivalence_check.py"
if [ "$NO_EQUIV" = 1 ]; then
  rm -f "$OUT/RELEASE.equivalence.txt"
  echo "  skipped (--no-equiv)"
else
  echo "  running, AALTO_OVERLAY=submission (mock episodes; can take minutes)"
  if ! (cd "$ROOT" && PYTHONPATH="$ROOT" AALTO_OVERLAY=submission "$PYTHON" tools/equivalence_check.py) \
      > "$OUT/RELEASE.equivalence.txt" 2> "$SCRATCH/equiv.err"; then
    cat "$SCRATCH/equiv.err" >&2
    die "tools/equivalence_check.py exited non-zero"
  fi
  cat "$OUT/RELEASE.equivalence.txt"
fi

step "7/8 writing RELEASE.md"
if [ "$NO_EQUIV" = 1 ]; then EQUIV_SRC=SKIPPED; else EQUIV_SRC="$OUT/RELEASE.equivalence.txt"; fi
R_ROOT="$ROOT" R_OUT="$OUT" R_OVERLAY="$OVERLAY" R_SHA="$SHA" R_SHA_FULL="$SHA_FULL" R_DATE="$DATE" \
R_GATE="$GATE" R_EQUIV="$EQUIV_SRC" "$PYTHON" - > "$OUT/RELEASE.md" <<'PY'
import hashlib, json, os, re
from pathlib import Path

root, out = Path(os.environ["R_ROOT"]), Path(os.environ["R_OUT"])
overlay, full, date = os.environ["R_OVERLAY"], os.environ["R_SHA_FULL"], os.environ["R_DATE"]
gate, equiv = os.environ["R_GATE"].strip(), os.environ["R_EQUIV"]

def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

flags = json.loads((out / "RELEASE.flags.json").read_text())["flags"]
on = {k: v for k, v in flags.items() if v["on"]}
sub_rel = "aaltonomous/baselines/classical/overlays/submission.yaml"
cfg_rel = "aaltonomous/baselines/classical/config.yaml"

lines = []
w = lines.append
w("# Release record")
w("")
w(f"- Git SHA: `{full}`")
w(f"- Overlay promoted: `{overlay}` -> `{sub_rel}`")
w(f"- Date (UTC): {date}")
w(f"- Flags ON: {len(on)} of {len(flags)}")
w("")
w("## Flags ON in submission")
w("")
if on:
    w("| flag | value | owner |")
    w("|---|---|---|")
    for k, v in on.items():
        w(f"| `{k}` | `{json.dumps(v['value'])}` | {v['owner']} |")
else:
    w("None (stock baseline behaviour).")
w("")
w("## Hashes (sha256)")
w("")
w(f"- `{sub_rel}`: `{sha256(root / sub_rel)}`")
w(f"- `{cfg_rel}`: `{sha256(root / cfg_rel)}`")
w("")
w("## Release gate")
w("")
verdict = "PASS" if "RELEASE GATE PASS" in gate else "FAIL"
w(f"Verdict: **{verdict}**")
w("")
w("```text")
w(gate)
w("```")
w("")
w("## Equivalence")
w("")
if equiv == "SKIPPED":
    w("SKIPPED (`--no-equiv`).")
else:
    w("`tools/equivalence_check.py` with `AALTO_OVERLAY=submission`. A record, not a pass/fail "
      "(also in `RELEASE.equivalence.txt`):")
    w("")
    w("```text")
    w(Path(equiv).read_text().rstrip())
    w("```")
w("")
w("## Docker build")
w("")
header = (root / "tools/submission.Dockerfile").read_text().splitlines()
cmd = next(l for l in header if l.startswith("#   docker build "))[1:].strip()
cmd = re.sub(r"\s+#.*$", "", cmd)
w("```sh")
w(cmd)
w("```")
print("\n".join(lines))
PY
echo "  wrote $OUT/RELEASE.md"

step "8/8 done"
if [ "$DRY" = 1 ]; then
  echo "dry-run output dir: $OUT"
  echo "RELEASE.md: $OUT/RELEASE.md"
  echo "nothing in $REPO was changed"
else
  echo "promoted $OVERLAY -> $SUB_REL (HEAD $SHA). NOT committed. Commit it yourself:"
  files=("$SUB_REL" RELEASE.md RELEASE.flags.json)
  [ "$NO_EQUIV" = 1 ] || files+=(RELEASE.equivalence.txt)
  echo "  git add ${files[*]}"
  echo "  git commit -m 'chore(release): promote $OVERLAY to submission'"
fi
