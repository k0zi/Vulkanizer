#!/usr/bin/env bash
# Regenerates src/ptx/heap_lib.ptx (Vulkanizer's device heap (malloc, free), embedded by the build) from
# src/ptx/heap_lib.cu. Needs the CUDA toolkit subset in .cuda/ (tools/setup_cuda.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/.cuda/bin/nvcc" -ptx -rdc=true -arch=compute_61 -O3 -ccbin g++-13 \
  "$ROOT/src/ptx/heap_lib.cu" -o "$ROOT/src/ptx/heap_lib.ptx"
echo "==> $ROOT/src/ptx/heap_lib.ptx"
