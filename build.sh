#!/usr/bin/env bash
# Configure (when needed) and build Vulkanizer.
#
#   ./build.sh                 incremental build (keeps the previous build type, default Debug)
#   ./build.sh --release       switch to Release
#   ./build.sh --clean         wipe the build directory first
#   ./build.sh --asan          build with AddressSanitizer (sticky, like the build type)
#   ./build.sh --no-asan       turn AddressSanitizer off again
#   ./build.sh -- -DFOO=BAR    pass extra arguments to cmake
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${VULKANIZER_BUILD_DIR:-$ROOT/build}"

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }

BUILD_TYPE=""
CLEAN=0
ASAN=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) BUILD_TYPE=Debug ;;
    --release) BUILD_TYPE=Release ;;
    --relwithdebinfo) BUILD_TYPE=RelWithDebInfo ;;
    --clean) CLEAN=1 ;;
    --asan) ASAN=ON ;;
    --no-asan) ASAN=OFF ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if [[ $CLEAN -eq 1 ]]; then
  echo "==> removing $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

if [[ -z "$BUILD_TYPE" ]]; then
  if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    BUILD_TYPE="$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "$BUILD_DIR/CMakeCache.txt")"
  fi
  BUILD_TYPE="${BUILD_TYPE:-Debug}"
fi
if [[ -z "$ASAN" ]]; then
  if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    ASAN="$(sed -n 's/^VULKANIZER_ASAN:BOOL=//p' "$BUILD_DIR/CMakeCache.txt")"
  fi
  ASAN="${ASAN:-OFF}"
fi

GENERATOR=()
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]] && command -v ninja >/dev/null 2>&1; then
  GENERATOR=(-G Ninja)
fi

echo "==> configure ($BUILD_TYPE, asan=$ASAN)"
cmake -S "$ROOT" -B "$BUILD_DIR" "${GENERATOR[@]}" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DVULKANIZER_ASAN="$ASAN" "${EXTRA[@]}" >/dev/null

echo "==> build"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "==> done: $BUILD_DIR/lib/libcuda.so.1"
