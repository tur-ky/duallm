#!/usr/bin/env bash
# Builds llama.cpp's RPC worker (ggml-rpc-server) with CUDA on a Linux machine that will serve GPUs or
# RAM to the DualLLM Portal over the network.
#
# It MUST be the same llama.cpp commit as the portal PC's llama-server (see scripts/build-llamacpp.ps1),
# and use the same GGML_MAX_NAME if you changed it. Mismatched builds fail in confusing ways.
#
# Prerequisites: git, cmake 3.24+, a C++ compiler, the NVIDIA driver and CUDA Toolkit.
#
# Usage:
#   ./build-llamacpp.sh [ref]                  # build the default branch, or the given branch/tag/commit
# Environment:
#   LLAMA_DIR=~/llama.cpp     where to clone and build
#   CUDA_ARCH=native          CMAKE_CUDA_ARCHITECTURES, e.g. "86;75" to cover several cards
#   REPO_URL=...              alternative llama.cpp remote
set -euo pipefail

REF="${1:-}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
CUDA_ARCH="${CUDA_ARCH:-native}"
REPO_URL="${REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"

for tool in git cmake; do
  command -v "$tool" >/dev/null || { echo "error: $tool not found" >&2; exit 1; }
done

[ -d "$LLAMA_DIR/.git" ] || git clone "$REPO_URL" "$LLAMA_DIR"
if [ -n "$REF" ]; then
  git -C "$LLAMA_DIR" fetch --tags origin
  git -C "$LLAMA_DIR" checkout "$REF"
fi
echo "Building llama.cpp at $(git -C "$LLAMA_DIR" rev-parse --short HEAD)"

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DGGML_RPC=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build "$LLAMA_DIR/build" --config Release --target ggml-rpc-server --parallel

echo "Done: $LLAMA_DIR/build/bin/ggml-rpc-server"
echo "Put that absolute path in duallm.config.json as the host's rpcServerPath."
