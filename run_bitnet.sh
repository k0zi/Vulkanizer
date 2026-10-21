#!/usr/bin/env bash
# Runs bitnet.c's CUDA backend on Vulkan through Vulkanizer (no NVIDIA GPU or driver needed).
#
#   ./run_bitnet.sh models/model.gguf -p "Hello" -n 64     bitnet arguments; --cuda is added
#   ./run_bitnet.sh --rebuild ...                           rebuild bitnet.c from scratch
#   ./run_bitnet.sh --build-only                            only build
#
# Options (before bitnet's arguments):
#   --bitnet-dir DIR   bitnet.c checkout (default: $BITNET_DIR or ~/Develop/GitHub/bitnet.c)
#   --rebuild          clean build of the bitnet.c copy
#   --build-only       build, do not run
#   --device NAME      Vulkan device by name substring (VULKANIZER_DEVICE)
#   --log LEVEL        Vulkanizer log level: off|error|warn|info|debug|trace (VULKANIZER_LOG)
#   --profile          per-kernel GPU times at exit (VULKANIZER_PROFILE=1)
#   --fast-fma         driver fma instead of the exact emulation (VULKANIZER_FAST_FMA=1)
#   --gdb              run under gdb
#
# The checkout itself is not modified: its sources are synced into .work/bitnet-vulkanizer and
# built there with the CUDA toolkit subset in .cuda/ (tools/setup_cuda.sh), as PTX only
# (compute_61, which Vulkanizer translates to SPIR-V), against Vulkanizer's cuBLAS header and libraries.
# Relative paths among the arguments are resolved against the current directory, then
# against the checkout (where bitnet's docs keep models/).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VULKANIZER_BUILD="${VULKANIZER_BUILD_DIR:-$ROOT/build-rel}"
BITNET_SRC="${BITNET_DIR:-$HOME/Develop/GitHub/bitnet.c}"
WORK="$ROOT/.work/bitnet-vulkanizer"

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }

REBUILD=0
BUILD_ONLY=0
GDB=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bitnet-dir) BITNET_SRC="$2"; shift ;;
    --rebuild) REBUILD=1 ;;
    --build-only) BUILD_ONLY=1 ;;
    --device) export VULKANIZER_DEVICE="$2"; shift ;;
    --log) export VULKANIZER_LOG="$2"; shift ;;
    --profile) export VULKANIZER_PROFILE=1 ;;
    --fast-fma) export VULKANIZER_FAST_FMA=1 ;;
    --gdb) GDB=1 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
  shift
done

[[ -f "$BITNET_SRC/src/gpu_cuda.cu" ]] || { echo "no bitnet.c checkout with a CUDA backend at $BITNET_SRC" >&2; exit 1; }
BITNET_SRC="$(cd "$BITNET_SRC" && pwd)"
[[ $# -gt 0 || $BUILD_ONLY -eq 1 ]] || { usage; exit 1; }

# ---- Vulkanizer and the CUDA toolkit subset ----------------------------------------------------------
[[ -x "$ROOT/.cuda/bin/nvcc" ]] || "$ROOT/tools/setup_cuda.sh"
VULKANIZER_BUILD_DIR="$VULKANIZER_BUILD" "$ROOT/build.sh" --release >/dev/null
echo "==> Vulkanizer: $VULKANIZER_BUILD/lib" >&2

# ---- bitnet.c, built in a synced copy -----------------------------------------------------------
mkdir -p "$WORK"
# Sources only; mtimes are kept, so make rebuilds just what changed.
rsync -a --delete --exclude .git --exclude '*.o' --exclude /models --exclude /bitnet \
  --exclude '/test_*' --exclude .build-config "$BITNET_SRC/" "$WORK/"

# Upstream only builds for sm_120: two plain MoE kernels sit inside the sm_120-only MXFP4 block
# while being used outside it. Move them out of it (idempotent).
python3 - "$WORK/src/gpu_cuda.cu" <<'EOF'
import re, sys
path = sys.argv[1]
src = open(path).read()
guard = "#ifdef BN_CUDA_MXFP4_SM120"
for name in ("moe_routed_activation_pair_kernel", "moe_routed_ordered_reduce_kernel"):
    m = re.search(r"static __global__ void " + name + r"\(", src)
    if not m:
        continue
    before = src[:m.start()]
    # inside the guard: its last #ifdef BN_CUDA_MXFP4_SM120 before the kernel is still open
    opened = before.rfind(guard)
    if opened < 0 or before.rfind("#endif") > opened:
        continue
    # the kernel ends at the first line that is just "}" after its start
    end = src.index("\n}\n", m.start()) + 3
    src = src[:m.start()] + "#endif  /* run_bitnet.sh: kernel used outside the MXFP4 block */\n" + \
        src[m.start():end] + guard + "\n" + src[end:]
open(path, "w").write(src)
EOF
# keep the source's mtime, so make only rebuilds gpu_cuda.o when bitnet.c itself changed
touch -r "$BITNET_SRC/src/gpu_cuda.cu" "$WORK/src/gpu_cuda.cu"

MAKE_ARGS=(
  BN_ENABLE_CUDA=1
  CUDA_ARCH=compute_61
  "NVCC=$ROOT/.cuda/bin/nvcc -ccbin g++-13 -I$ROOT/include/cublas"
  "CUDA_LDFLAGS=-L$VULKANIZER_BUILD/lib -lcudart -lcublas -lstdc++"
)
if [[ $REBUILD -eq 1 ]]; then make -C "$WORK" clean >/dev/null; fi
echo "==> building bitnet.c (CUDA backend, PTX) in $WORK" >&2
if ! make -C "$WORK" -j"$(nproc)" "${MAKE_ARGS[@]}" bitnet > "$WORK/build.log" 2>&1; then
  grep -E "error|Error" "$WORK/build.log" | head -20 >&2
  echo "bitnet.c build failed, see $WORK/build.log" >&2
  exit 1
fi
[[ $BUILD_ONLY -eq 1 ]] && { echo "==> $WORK/bitnet" >&2; exit 0; }

# ---- run ------------------------------------------------------------------------------------------
ARGS=()
HAS_BACKEND=0
for a in "$@"; do
  case "$a" in --cuda|--webgpu|--metal) HAS_BACKEND=1 ;; esac
  if [[ "$a" != -* && "$a" != /* ]]; then
    if [[ -e "$a" ]]; then a="$(cd "$(dirname "$a")" && pwd)/$(basename "$a")"
    elif [[ -e "$BITNET_SRC/$a" ]]; then a="$BITNET_SRC/$a"
    fi
  fi
  ARGS+=("$a")
done
[[ $HAS_BACKEND -eq 1 ]] || ARGS+=(--cuda)

export LD_LIBRARY_PATH="$VULKANIZER_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ $GDB -eq 1 ]]; then exec gdb --args "$WORK/bitnet" "${ARGS[@]}"; fi
exec "$WORK/bitnet" "${ARGS[@]}"
