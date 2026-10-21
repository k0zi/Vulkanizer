#!/usr/bin/env bash
# Downloads a CUDA toolkit subset (nvcc + runtime + CCCL headers) into .cuda/ for building
# test programs with a real nvcc. No root, no driver: Vulkanizer provides libcuda.so.
# NVIDIA's redistributable archives, verified against the published checksums.
#
#   tools/setup_cuda.sh          install into .cuda/ (skipped if present)
#   tools/setup_cuda.sh --force  reinstall
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/.cuda"
BASE=https://developer.download.nvidia.com/compute/cuda/redist
ARCHIVES=(
  "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz 7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35"
  "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.9.79-archive.tar.xz 1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99"
  "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.9.27-archive.tar.xz 8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa"
)

if [[ "${1:-}" == "--force" ]]; then rm -rf "$DEST"; fi
if [[ -x "$DEST/bin/nvcc" ]]; then echo "already installed: $DEST"; exit 0; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DEST"
for a in "${ARCHIVES[@]}"; do
  read -r path sum <<<"$a"
  file="$TMP/$(basename "$path")"
  echo "==> $(basename "$path")"
  curl -fsSL "$BASE/$path" -o "$file"
  echo "$sum  $file" | sha256sum -c --quiet -
  tar -xJf "$file" -C "$DEST" --strip-components=1
done

# glibc >= 2.41 declares rsqrt/sinpi/cospi (C23) with noexcept; CUDA 12.9's declarations
# lack it and nvcc rejects the mismatch. Add the same exception specification (harmless with
# older glibc, which does not declare these functions at all).
sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +(rsqrtf?|sinpif?|cospif?)\((double|float) x\));$/\1 __VULKANIZER_GLIBC_NOEXCEPT;/' \
  "$DEST/include/crt/math_functions.h"
sed -i '0,/^#define __MATH_FUNCTIONS_H__/s//&\n#if defined(__cplusplus)\n#define __VULKANIZER_GLIBC_NOEXCEPT noexcept(true)\n#else\n#define __VULKANIZER_GLIBC_NOEXCEPT\n#endif/' \
  "$DEST/include/crt/math_functions.h"
# nvcc.profile looks for libraries in lib64/.
ln -sfn lib "$DEST/lib64"
echo "==> installed into $DEST"
