#!/usr/bin/env bash
# Run any program with Vulkanizer's libcuda.so in front of the library search path.
#
#   ./run.sh ./my_cuda_app args...     an external CUDA program
#   ./run.sh device_query              bare names are looked up in build/bin first
#   ./run.sh python3 script.py         works for anything that dlopen()s libcuda.so.1
#
# Options (before the program):
#   --device NAME   pick a Vulkan device by name substring (VULKANIZER_DEVICE)
#   --log LEVEL     off|error|warn|info|debug|trace (VULKANIZER_LOG); trace logs every API call
#   --validation    enable the Vulkan validation layer (VULKANIZER_VALIDATION=1)
#   --gdb           run under gdb
#   --no-build      skip the build step
#   --check         check the given files or directories for PTX and a shared CUDA runtime instead of
#                   running them (vulkanizer-check)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${VULKANIZER_BUILD_DIR:-$ROOT/build}"

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; }

BUILD=1
GDB=0
CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) export VULKANIZER_DEVICE="$2"; shift ;;
    --log) export VULKANIZER_LOG="$2"; shift ;;
    --validation) export VULKANIZER_VALIDATION=1 ;;
    --gdb) GDB=1 ;;
    --no-build) BUILD=0 ;;
    --check) CHECK=1 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; usage; exit 1 ;;
    *) break ;;
  esac
  shift
done

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

if [[ $BUILD -eq 1 ]]; then
  "$ROOT/build.sh" >/dev/null
fi

if [[ $CHECK -eq 1 ]]; then
  exec "$BUILD_DIR/bin/vulkanizer-check" "$@"
fi

PROGRAM="$1"
shift
if [[ "$PROGRAM" != */* && -x "$BUILD_DIR/bin/$PROGRAM" ]]; then
  PROGRAM="$BUILD_DIR/bin/$PROGRAM"
fi

export LD_LIBRARY_PATH="$BUILD_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# With an ASAN build of Vulkanizer, uninstrumented programs (e.g. built by nvcc) need the
# sanitizer runtime preloaded.
if grep -q '^VULKANIZER_ASAN:BOOL=ON' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
  CXX_COMPILER="$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$BUILD_DIR/CMakeCache.txt")"
  export LD_PRELOAD="$("${CXX_COMPILER:-c++}" -print-file-name=libasan.so)${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if [[ $GDB -eq 1 ]]; then
  exec gdb --args "$PROGRAM" "$@"
fi
exec "$PROGRAM" "$@"
