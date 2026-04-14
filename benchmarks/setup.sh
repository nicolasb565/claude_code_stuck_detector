#!/bin/bash
# Clone fixtures into benchmarks/fixtures/$TASK_ID and compile them inside
# the benchmark-runner container.
#
# Usage:
#   bash benchmarks/setup.sh                  # clone + compile all tasks
#   bash benchmarks/setup.sh --tasks 04_sqlite_cte,08_express_async
#   bash benchmarks/setup.sh --skip-build     # clone only, skip compile
#   bash benchmarks/setup.sh --force          # wipe & re-clone even if cached
#
# Requires: docker, jq, git. The benchmark-runner image must be built first
# (`docker build -t benchmark-runner:latest benchmarks/`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$SCRIPT_DIR/manifest.json"
FIXTURES_DIR="$SCRIPT_DIR/fixtures"
IMAGE="benchmark-runner:latest"

TASK_FILTER=""
SKIP_BUILD=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)      TASK_FILTER="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --force)      FORCE=1; shift ;;
    -h|--help)
      sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v jq >/dev/null     || { echo "jq not found";     exit 1; }
command -v git >/dev/null    || { echo "git not found";    exit 1; }

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "image $IMAGE not found — run: docker build -t $IMAGE $SCRIPT_DIR"
  exit 1
}

mkdir -p "$FIXTURES_DIR"

# Build task list
if [ -n "$TASK_FILTER" ]; then
  IFS=',' read -ra TASKS <<<"$TASK_FILTER"
else
  mapfile -t TASKS < <(jq -r '.tasks[].id' "$MANIFEST")
fi

log() { echo "[setup] $*"; }

FAILED_TASKS=()
trap 'log "=== setup.sh interrupted ==="; [ ${#FAILED_TASKS[@]} -gt 0 ] && log "failures so far: ${FAILED_TASKS[*]}"; exit 130' INT TERM

for TASK_ID in "${TASKS[@]}"; do
  log "=== $TASK_ID ==="
  FIX_DIR="$FIXTURES_DIR/$TASK_ID"

  TASK_JSON=$(jq -r --arg id "$TASK_ID" '.tasks[] | select(.id==$id)' "$MANIFEST")
  [ -n "$TASK_JSON" ] || { log "  unknown task: $TASK_ID"; continue; }

  SYNTHETIC=$(echo "$TASK_JSON" | jq -r '.fixture.synthetic // false')
  URL=$(echo "$TASK_JSON"       | jq -r '.fixture.url       // empty')
  REF=$(echo "$TASK_JSON"       | jq -r '.fixture.ref       // empty')
  SUBMODULES=$(echo "$TASK_JSON" | jq -r '.fixture.submodules // false')
  BUILD_CMD=$(echo "$TASK_JSON" | jq -r '.fixture.build_cmd // empty')

  # ── Synthetic: copy bundled source ───────────────────────────────────────
  if [ "$SYNTHETIC" = "true" ]; then
    SRC_REL=$(echo "$TASK_JSON" | jq -r '.fixture.source_dir')
    SRC_ABS="$SCRIPT_DIR/${SRC_REL#benchmarks/}"
    if [ "$FORCE" -eq 1 ] || [ ! -d "$FIX_DIR" ]; then
      log "  copying synthetic source from $SRC_ABS"
      rm -rf "$FIX_DIR"
      mkdir -p "$FIX_DIR"
      cp -a "$SRC_ABS/." "$FIX_DIR/"
    else
      log "  already cached"
    fi
    continue
  fi

  # ── Regular clone ────────────────────────────────────────────────────────
  if [ -d "$FIX_DIR/.git" ] || [ -f "$FIX_DIR/.git" ]; then
    if [ "$FORCE" -eq 1 ]; then
      log "  --force: wiping $FIX_DIR"
      rm -rf "$FIX_DIR"
    else
      log "  reusing cached clone; resetting state"
      git -C "$FIX_DIR" reset --hard 2>&1 | sed 's/^/    /' || true
      git -C "$FIX_DIR" clean -fdx 2>&1 | sed 's/^/    /' || true
    fi
  fi

  if [ ! -d "$FIX_DIR" ]; then
    log "  cloning $URL @ $REF"
    # Strategy:
    #   1. Init empty repo, add remote
    #   2. Try `git fetch --depth 1 origin <REF>` (works for SHAs on servers
    #      that allow uploadpack.allowReachableSHA1InWant, plus branches/tags)
    #   3. Fall back to full shallow clone + checkout
    mkdir -p "$FIX_DIR"
    git -C "$FIX_DIR" init -q
    git -C "$FIX_DIR" remote add origin "$URL"
    if git -C "$FIX_DIR" -c protocol.version=2 fetch --depth 1 \
         --filter=blob:none origin "$REF" 2>/dev/null; then
      git -C "$FIX_DIR" checkout --detach FETCH_HEAD
    else
      log "  shallow-fetch of SHA failed; falling back to fuller clone"
      rm -rf "$FIX_DIR"
      git clone --filter=blob:none "$URL" "$FIX_DIR"
      git -C "$FIX_DIR" checkout --detach "$REF"
    fi

    if [ "$SUBMODULES" = "true" ]; then
      log "  initializing submodules"
      git -C "$FIX_DIR" submodule update --init --recursive --depth 1 || \
        git -C "$FIX_DIR" submodule update --init --recursive

      # Apply pinned submodule SHAs (if any)
      while IFS=$'\t' read -r path sha; do
        [ -z "$path" ] && continue
        log "  pinning submodule $path -> $sha"
        (
          cd "$FIX_DIR/$path"
          git fetch --depth 1 origin "$sha" 2>/dev/null || git fetch origin
          git checkout --detach "$sha"
        )
      done < <(echo "$TASK_JSON" | jq -r '.fixture.submodule_pins // {} | to_entries[] | "\(.key)\t\(.value)"')
    fi
  fi

  # ── Compile step (inside container) ───────────────────────────────────────
  if [ "$SKIP_BUILD" -eq 1 ]; then
    log "  --skip-build: leaving fixture uncompiled"
    continue
  fi
  if [ -z "$BUILD_CMD" ] || [ "$BUILD_CMD" = "null" ]; then
    log "  no build_cmd — skipping compile"
    continue
  fi

  log "  compiling inside container..."
  if docker run --rm \
       --user "$(id -u):$(id -g)" \
       -e HOME=/tmp \
       -v "$MANIFEST:/manifest.json:ro" \
       -v "$FIX_DIR:/work" \
       "$IMAGE" compile "$TASK_ID" 2>&1 | sed 's/^/    /' \
     && [ ${PIPESTATUS[0]} -eq 0 ]
  then
    log "  compile OK"
  else
    log "  compile FAILED — task will be skipped at run time"
    FAILED_TASKS+=("$TASK_ID")
  fi
done

if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
  log "done — all ${#TASKS[@]} tasks prepared."
else
  log "done — ${#FAILED_TASKS[@]}/${#TASKS[@]} tasks FAILED: ${FAILED_TASKS[*]}"
  log "  rerun with --tasks <id> to iterate on the build_cmd for each."
fi
exit 0
