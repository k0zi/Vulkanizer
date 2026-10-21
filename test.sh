#!/usr/bin/env bash
# Build, then run the test suite.
#
#   ./test.sh                     all tests on the default device
#   ./test.sh --device llvmpipe   run on a specific Vulkan device (name substring)
#   ./test.sh --all-devices       run once per usable Vulkan device (incl. CPU ones)
#   ./test.sh --log debug         set VULKANIZER_LOG for the run
#   ./test.sh --no-build          skip the build step
#   ./test.sh --release --asan    build options (--debug/--release/--clean/--asan) go to build.sh
#   ./test.sh -R memory           anything else is passed to ctest
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${VULKANIZER_BUILD_DIR:-$ROOT/build}"

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; }

BUILD=1
ALL_DEVICES=0
BUILD_ARGS=()
CTEST_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) export VULKANIZER_DEVICE="$2"; shift ;;
    --all-devices) ALL_DEVICES=1 ;;
    --log) export VULKANIZER_LOG="$2"; shift ;;
    --no-build) BUILD=0 ;;
    --debug|--release|--relwithdebinfo|--clean|--asan|--no-asan) BUILD_ARGS+=("$1") ;;
    -h|--help) usage; exit 0 ;;
    *) CTEST_ARGS+=("$1") ;;
  esac
  shift
done

if [[ $BUILD -eq 1 ]]; then
  "$ROOT/build.sh" "${BUILD_ARGS[@]}"
fi

run_ctest() {
  ctest --test-dir "$BUILD_DIR" --output-on-failure "${CTEST_ARGS[@]}"
}

if [[ $ALL_DEVICES -eq 0 ]]; then
  echo "==> test (device: ${VULKANIZER_DEVICE:-default})"
  run_ctest
  exit $?
fi

FAILED=0
while IFS= read -r device; do
  [[ -z "$device" ]] && continue
  echo "==> test (device: $device)"
  VULKANIZER_DEVICE="$device" run_ctest || FAILED=1
done < <("$BUILD_DIR/bin/vulkanizer-info" --names)
exit $FAILED
