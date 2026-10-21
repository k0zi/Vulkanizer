#!/usr/bin/env bash
# Regenerate test PTX from CUDA sources, without a CUDA SDK.
#
#   ./gen_ptx.sh foo.cu [more.cu ...]      writes foo.ptx next to the source
#   SM=sm_75 ./gen_ptx.sh foo.cu           pick the virtual architecture (default sm_52)
#   CLANG=clang-18 ./gen_ptx.sh foo.cu     pick the compiler
#
# nvcc works too:  nvcc -ptx -arch=compute_52 foo.cu -o foo.ptx
set -euo pipefail

CLANG="${CLANG:-clang}"
SM="${SM:-sm_52}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ $# -gt 0 ]] || { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
command -v "$CLANG" >/dev/null || { echo "$CLANG not found (install clang, or set CLANG=...)" >&2; exit 1; }

for src in "$@"; do
  out="${src%.cu}.ptx"
  "$CLANG" -x cuda --cuda-device-only --cuda-gpu-arch="$SM" -nocudainc -nocudalib \
    -O2 -S -I"$DIR" "$src" -o "$out"
  echo "$src -> $out"
done
