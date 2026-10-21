#!/usr/bin/env bash
# Remove all build output.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${VULKANIZER_BUILD_DIR:-$ROOT/build}"

echo "==> removing $BUILD_DIR"
rm -rf "$BUILD_DIR"
