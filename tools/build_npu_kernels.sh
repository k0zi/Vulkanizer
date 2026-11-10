#!/usr/bin/env bash
# Build the AMD XDNA NPU GEMM kernels (plan.md 7.1) with mlir-aie (IRON) + Peano (llvm-aie).
#
#   tools/build_npu_kernels.sh                 cubes 256^3 .. 4096^3, bf16 -> f32
#   tools/build_npu_kernels.sh 1024x1024x4096  only the given M x K x N shapes
#   tools/build_npu_kernels.sh --all           every M, K, N in {256, 512, 1024, 2048, 4096} (125 kernels, slow)
#
# Output: $VULKANIZER_NPU_KERNELS (default $XDG_DATA_HOME/vulkanizer/npu, i.e. ~/.local/share/vulkanizer/npu):
#   gemm_bf16_f32_<M>x<K>x<N>.xclbin + gemm_bf16_f32_<M>x<K>x<N>.insts.bin  (row-major lhs M x K, rhs K x N)
# The toolchain (a Python 3.12 venv with the pinned mlir-aie and llvm-aie wheels, and the mlir-aie sources
# at the same commit for the example design) goes to $VULKANIZER_AIE_HOME (default ~/.cache/vulkanizer).
# Needs: uv, git, make, cmake, XRT in /opt/xilinx/xrt. The NPU generation comes from /sys/class/accel.
set -euo pipefail

# Pinned versions: the kernel signature (opcode, instr, count, a, b, c, tmp, trace) and the IRON API change
# between mlir-aie releases; src/npu/xdna.cpp matches this commit.
MLIR_AIE_COMMIT=886d932
MLIR_AIE_WHEEL="mlir_aie==0.0.1.2026052005+886d932"
MLIR_AIE_WHEELS_URL=https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-3
LLVM_AIE_WHEEL="llvm-aie==22.0.0.2026111001+152bedac"
LLVM_AIE_WHEELS_URL=https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

AIE_HOME="${VULKANIZER_AIE_HOME:-$HOME/.cache/vulkanizer}"
OUT="${VULKANIZER_NPU_KERNELS:-${XDG_DATA_HOME:-$HOME/.local/share}/vulkanizer/npu}"
SIZES=(256 512 1024 2048 4096)

SHAPES=()
if [[ $# -eq 0 ]]; then
  for s in "${SIZES[@]}"; do SHAPES+=("${s}x${s}x${s}"); done
elif [[ "$1" == "--all" ]]; then
  for m in "${SIZES[@]}"; do for k in "${SIZES[@]}"; do for n in "${SIZES[@]}"; do SHAPES+=("${m}x${k}x${n}"); done; done; done
elif [[ "$1" == -h || "$1" == --help ]]; then
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0
else
  SHAPES=("$@")
fi

# NPU generation: XDNA1 (Phoenix/Hawk Point, 1502) = npu, 4 columns; XDNA2 (Strix/Krackan, 17f0) = npu2, 8.
dev=""
for d in /sys/class/accel/accel*/device; do
  [[ -e "$d/device" ]] || continue
  case "$(cat "$d/device")" in
    0x1502) dev=npu ;;
    0x17f0) dev=npu2 ;;
  esac
done
dev="${VULKANIZER_NPU_DEV:-$dev}"
[[ -n "$dev" ]] || { echo "no AMD XDNA NPU found (set VULKANIZER_NPU_DEV=npu|npu2 to build anyway)" >&2; exit 1; }
if [[ "$dev" == npu2 ]]; then npu2=1 cols=8; else npu2=0 cols=4; fi
# Per-core tile m x k x n: A 64x64 + B 64x32 bf16, C 64x32 f32, double-buffered = 40 KiB of the 64 KiB core
# memory. The design moves C in groups of 2 tile rows of 4 cores: M % (m * 4 * 2) == 0, K % k == 0,
# N % (n * cols) == 0. M = 256 takes m = 32.
tk=64 tn=32

# Toolchain.
venv="$AIE_HOME/aie-venv"
src="$AIE_HOME/mlir-aie"
mkdir -p "$AIE_HOME"
if [[ ! -x "$venv/bin/python" ]]; then
  uv venv -p 3.12 "$venv"
fi
VIRTUAL_ENV="$venv" uv pip install -q "$MLIR_AIE_WHEEL" -f "$MLIR_AIE_WHEELS_URL"
VIRTUAL_ENV="$venv" uv pip install -q "$LLVM_AIE_WHEEL" -f "$LLVM_AIE_WHEELS_URL"
if [[ ! -d "$src/.git" ]]; then
  git clone -q --filter=blob:none https://github.com/Xilinx/mlir-aie.git "$src"
fi
git -C "$src" fetch -q origin "$MLIR_AIE_COMMIT" 2>/dev/null || true
git -C "$src" checkout -q "$MLIR_AIE_COMMIT"

# shellcheck disable=SC1091
source "$venv/bin/activate"
# shellcheck disable=SC1091
source /opt/xilinx/xrt/setup.sh >/dev/null
site="$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PEANO_INSTALL_DIR="$site/llvm-aie"
export MLIR_AIE_INSTALL_DIR="$site/mlir_aie"
export PATH="$MLIR_AIE_INSTALL_DIR/bin:$PATH"

design="$src/programming_examples/basic/matrix_multiplication/whole_array"
work="$AIE_HOME/npu-build"
mkdir -p "$OUT" "$work"
echo "==> $dev ($cols columns), mlir-aie $MLIR_AIE_COMMIT, ${#SHAPES[@]} kernel(s) -> $OUT"

failed=0
for shape in "${SHAPES[@]}"; do
  IFS=x read -r M K N <<<"$shape"
  name="gemm_bf16_f32_${M}x${K}x${N}"
  if [[ -f "$OUT/$name.xclbin" && -f "$OUT/$name.insts.bin" ]]; then
    echo "    $name: up to date"
    continue
  fi
  if (( M % 512 == 0 )); then tm=64; else tm=32; fi
  if (( M % (tm * 8) || K % tk || N % (tn * cols) )); then
    echo "    $name: skipped (M % 256, K % $tk, N % $((tn * cols)) must be 0)" >&2
    continue
  fi
  echo "    $name"
  dir="$work/$dev/$shape"
  mkdir -p "$dir"
  suffix="${M}x${K}x${N}_${tm}x${tk}x${tn}_${cols}c"
  if ! (cd "$dir" && make -f "$design/Makefile" NPU2=$npu2 use_iron=1 dtype_in=bf16 dtype_out=f32 \
          M="$M" K="$K" N="$N" m=$tm k=$tk n=$tn n_aie_cols=$cols "build/final_${suffix}.xclbin" >build.log 2>&1); then
    echo "    $name: FAILED, see $dir/build.log" >&2
    failed=1
    continue
  fi
  cp "$dir/build/final_${suffix}.xclbin" "$OUT/$name.xclbin"
  cp "$dir/build/insts_${suffix}.txt" "$OUT/$name.insts.bin"  # binary despite the .txt name
done
echo "==> done: $OUT"
exit $failed
