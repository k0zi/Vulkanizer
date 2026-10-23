#!/usr/bin/env bash
# Regenerates src/ptx/texture_lib.ptx (Vulkanizer's texture functions, embedded by the build) from
# src/ptx/texture_lib.cu. Needs the CUDA toolkit subset in .cuda/ (tools/setup_cuda.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/.cuda/bin/nvcc" -ptx -rdc=true -arch=compute_61 -O3 -ccbin g++-13 -I "$ROOT/src/common" \
  "$ROOT/src/ptx/texture_lib.cu" -o "$ROOT/src/ptx/texture_lib.ptx"
echo "==> $ROOT/src/ptx/texture_lib.ptx"
