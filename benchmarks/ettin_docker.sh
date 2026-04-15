#!/usr/bin/env bash
# Run the Ettin encoder fine-tuning inside the rocm/pytorch:latest container.
#
# Why Docker: host torch-rocm wheels have gfx1201 kernel gaps that crash Phi-3
# forward passes (and potentially Ettin's RoPE path too). rocm/pytorch ships
# torch 2.10+rocm7.2.2 which is the AMD-validated stack for RDNA4.
#
# Usage:
#   benchmarks/ettin_docker.sh --smoke              # smoke test (500 sessions)
#   benchmarks/ettin_docker.sh                       # full training (3 epochs)
#   benchmarks/ettin_docker.sh --batch-size 4        # pass through any CLI flag
#
# Outputs land in proxy/experiments/ettin_400m_stuck/ on the host.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

IMAGE="rocm/pytorch:latest"

mkdir -p "$REPO_DIR/proxy/experiments/ettin_400m_stuck"
mkdir -p "$REPO_DIR/data/generated"

# Extra packages the script imports that are not in the base container.
# The base rocm/pytorch image has torch+rocm preinstalled but not the
# recent transformers / sklearn revision we rely on.
PIP_INSTALL='pip install --quiet --no-input "transformers>=5,<6" "accelerate>=1.10" "scikit-learn>=1.5"'

# Pass through any flags the user gave us to ettin_train.py
PY_ARGS="benchmarks/ettin_train.py"
for arg in "$@"; do
    PY_ARGS="$PY_ARGS \"$arg\""
done

echo "=== Launching $IMAGE ==="
echo "  repo:     $REPO_DIR"
echo "  hf_cache: $HF_CACHE"
echo "  video gid: $VIDEO_GID"
echo "  render gid: $RENDER_GID"
echo "  cmd:      $PY_ARGS"
echo

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --security-opt seccomp=unconfined \
    --shm-size=8g \
    -v "$REPO_DIR":/workspace \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v /tmp:/tmp \
    -w /workspace \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRANSFORMERS_VERBOSITY=warning \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE" \
    bash -c "$PIP_INSTALL && python -u $PY_ARGS"
