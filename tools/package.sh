#!/usr/bin/env bash
# Builds the release packages: a Debian package and a tarball (cmake/install.cmake), in build-pkg/.
#
#   tools/package.sh            vulkanizer_<version>_<arch>.deb and vulkanizer-<version>-<arch>.tar.gz
#   tools/package.sh --check    also unpacks the .deb and runs vulkanizer-info through it
#
# Install the .deb with `sudo apt install ./build-pkg/vulkanizer_*.deb`, or unpack the tarball anywhere;
# either way programs start with vulkanizer-run (see the README).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${VULKANIZER_PKG_DIR:-$ROOT/build-pkg}"

CHECK=0
case "${1:-}" in
  --check) CHECK=1 ;;
  "") ;;
  *) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DVULKANIZER_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr >/dev/null
cmake --build "$BUILD" -j"$(nproc)"
(cd "$BUILD" && cpack)

if [[ $CHECK -eq 1 ]]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  dpkg-deb -x "$BUILD"/vulkanizer_*.deb "$tmp"
  "$tmp/usr/bin/vulkanizer-run" "$tmp/usr/bin/vulkanizer-info"
fi
