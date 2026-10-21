#!/usr/bin/env bash
# Runs Python with PyTorch on Vulkan through Vulkanizer (no NVIDIA GPU or driver needed).
#
#   ./run_torch.sh script.py args...      a PyTorch script ("cuda" is the Vulkan GPU)
#   ./run_torch.sh -c "import torch; print(torch.cuda.is_available())"
#   ./run_torch.sh                         interactive Python
#
# Options (before the Python arguments):
#   --env DIR        PyTorch environment (default .work/torch-cf, created by tools/setup_torch.sh)
#   --device NAME    Vulkan device by name substring (VULKANIZER_DEVICE)
#   --log LEVEL      off|error|warn|info|debug|trace (VULKANIZER_LOG)
#   --profile        per-kernel GPU and per-call host times at exit (VULKANIZER_PROFILE=1)
#   --gdb            run under gdb
#
# Vulkanizer's libcuda, libcudart, libcublas and libcublasLt are preloaded, so PyTorch uses them
# instead of the NVIDIA libraries in its environment. tools/torch/sitecustomize.py turns off what
# needs NVIDIA hardware or libraries (cuDNN, fused attention); VULKANIZER_TORCH_DEFAULTS=0 keeps
# PyTorch's own defaults.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${VULKANIZER_BUILD_DIR:-$ROOT/build-rel}"
ENV="$ROOT/.work/torch-cf"

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }

GDB=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV="$2"; shift ;;
    --device) export VULKANIZER_DEVICE="$2"; shift ;;
    --log) export VULKANIZER_LOG="$2"; shift ;;
    --profile) export VULKANIZER_PROFILE=1 ;;
    --gdb) GDB=1 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
  shift
done

[[ -x "$ENV/bin/python" ]] || "$ROOT/tools/setup_torch.sh"
VULKANIZER_BUILD_DIR="$BUILD" "$ROOT/build.sh" --release >/dev/null

LIB="$BUILD/lib"
export PYTHONPATH="$ROOT/tools/torch${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$LIB/libcuda.so.1:$LIB/libcudart.so.12:$LIB/libcublas.so.12:$LIB/libcublasLt.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
if [[ $GDB -eq 1 ]]; then exec gdb --args "$ENV/bin/python" "$@"; fi
exec "$ENV/bin/python" "$@"
