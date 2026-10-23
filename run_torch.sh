#!/usr/bin/env bash
# Runs Python with PyTorch on Vulkan through Vulkanizer (no NVIDIA GPU or driver needed).
#
#   ./run_torch.sh script.py args...      a PyTorch script ("cuda" is the Vulkan GPU)
#   ./run_torch.sh torch_ops               a libtorch test from the build (torch_ops, torch_mlp, torch_split)
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
# Vulkanizer's libcuda, libcudart, libcublas, libcublasLt, libcusolver, libcufft and libcusparse are preloaded, so PyTorch uses
# them instead of the NVIDIA libraries in its environment. What needs NVIDIA hardware or libraries is
# turned off (cuDNN and fused attention in tools/torch/sitecustomize.py, the rest below);
# VULKANIZER_TORCH_DEFAULTS=0 keeps PyTorch's own defaults.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${VULKANIZER_BUILD_DIR:-$ROOT/build-rel}"
ENV="$ROOT/.work/torch-cf"

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; }

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
if [[ "${VULKANIZER_TORCH_DEFAULTS:-1}" != 0 ]]; then
  export PYTHONPATH="$ROOT/tools/torch${PYTHONPATH:+:$PYTHONPATH}"  # sitecustomize.py: cuDNN, fused attention
  # torch._native's CuTe DSL kernels compile to SASS; the same switch also turns off its (working) Triton ones
  export TORCH_DISABLE_NATIVE_JIT="${TORCH_DISABLE_NATIVE_JIT:-1}"
  # torch.linalg on Vulkanizer's libcusolver: MAGMA's own kernels and the cuBLAS routines it calls do not run
  export TORCH_LINALG_PREFER_CUSOLVER="${TORCH_LINALG_PREFER_CUSOLVER:-1}"
  # Triton: PTX is handed through instead of assembled, the launcher is built against our cuda.h
  export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$ROOT/tools/triton/ptxas}"
  export TRITON_PTXAS_BLACKWELL_PATH="${TRITON_PTXAS_BLACKWELL_PATH:-$ROOT/tools/triton/ptxas}"
  export C_INCLUDE_PATH="$ROOT/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
fi
export LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_LIBCUDA_PATH="$LIB"  # Triton links its launcher against this libcuda
# a CUDA 13 build of PyTorch (tools/setup_torch.sh --cuda13) gets the .so.13 libraries from lib/cuda13
CUDA_LIB="$LIB" CUDA_MAJOR=12
if [[ -e "$ENV/lib/libcudart.so.13" ]]; then CUDA_LIB="$LIB/cuda13" CUDA_MAJOR=13; fi
export LD_PRELOAD="$LIB/libcuda.so.1:$CUDA_LIB/libcudart.so.$CUDA_MAJOR:$CUDA_LIB/libcublas.so.$CUDA_MAJOR:$CUDA_LIB/libcublasLt.so.$CUDA_MAJOR:$CUDA_LIB/libcusolver.so.$((CUDA_MAJOR - 1)):$CUDA_LIB/libcufft.so.$((CUDA_MAJOR - 1)):$CUDA_LIB/libcusparse.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
# run-time compiled kernels (the jiterator) as PTX, not cubins (src/nvrtc/nvrtc_ptx.cpp)
export LD_AUDIT="$LIB/libvulkanizer_nvrtc.so${LD_AUDIT:+:$LD_AUDIT}"
# a libtorch test by name (torch_ops, ...) or any other program instead of Python
PROGRAM=("$ENV/bin/python")
if [[ $# -gt 0 && "$1" != */* && -x "$BUILD/bin/$1" ]]; then
  PROGRAM=("$BUILD/bin/$1"); shift
elif [[ $# -gt 0 && -x "$1" && ! -d "$1" && "$1" != *.py ]]; then
  PROGRAM=("$1"); shift
fi
if [[ $GDB -eq 1 ]]; then exec gdb ${GDB_ARGS:-} --args "${PROGRAM[@]}" "$@"; fi
exec "${PROGRAM[@]}" "$@"
