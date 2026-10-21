#!/usr/bin/env bash
# Full verification: Debug+ASAN and Release builds, tests on every usable Vulkan device.
# Uses separate build directories (build-ci/), so the normal build/ is left alone.
#
#   ./ci.sh            run everything
#   ./ci.sh --clean    start from fresh build directories
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN=()
case "${1:-}" in
  --clean) CLEAN=(--clean) ;;
  "") ;;
  -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown option: $1" >&2; exit 1 ;;
esac

run() {
  local name="$1"; shift
  echo "=============== $name ==============="
  VULKANIZER_BUILD_DIR="$ROOT/build-ci/$name" "$ROOT/test.sh" --all-devices "${CLEAN[@]}" "$@"
}

run debug-asan --debug --asan
run release --release --no-asan
echo "=============== ci: all passed ==============="
