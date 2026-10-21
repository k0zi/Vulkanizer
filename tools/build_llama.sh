#!/usr/bin/env bash
# Builds llama.cpp's CUDA backend against Vulkanizer, for testing (plan.md, phase 4).
# Needs the CUDA toolkit subset in .cuda/ (tools/setup_cuda.sh) and a built Vulkanizer (./build.sh).
#
#   tools/build_llama.sh              clone (if needed) into .work/llama.cpp and build
#   tools/build_llama.sh --tests      also build test-backend-ops
#   tools/build_llama.sh --fa         with flash attention, into .work/llama-fa (correct, but on
#                                     GPUs without tensor cores its prompt processing is slower)
#
# Run it with Vulkanizer's libraries, e.g.:
#   LD_LIBRARY_PATH=build/lib .work/llama-build/bin/llama-simple -m model.gguf -ngl 99 "Hello"
#   LD_LIBRARY_PATH=build/lib .work/llama-build/bin/test-backend-ops test -b CUDA0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/.work/llama.cpp"
OUT="$ROOT/.work/llama-build"
TESTS=OFF
FA=OFF
for arg in "$@"; do
  case "$arg" in
    --tests) TESTS=ON ;;
    --fa) FA=ON; OUT="$ROOT/.work/llama-fa" ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

[[ -x "$ROOT/.cuda/bin/nvcc" ]] || { echo "run tools/setup_cuda.sh first" >&2; exit 1; }
[[ -f "$ROOT/build/lib/libcublas.so" ]] || { echo "run ./build.sh first" >&2; exit 1; }
if [[ ! -d "$SRC" ]]; then
  mkdir -p "$ROOT/.work"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SRC"
fi

# PTX only (61-virtual: dp4a, no tensor-core paths), shared CUDA runtime, Vulkanizer's cuBLAS header
# and library; VMM, CUDA graphs and NCCL are not supported by Vulkanizer yet.
cmake -S "$SRC" -B "$OUT" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_FA=$FA \
  -DCMAKE_CUDA_ARCHITECTURES=61-virtual \
  -DCMAKE_CUDA_COMPILER="$ROOT/.cuda/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCUDAToolkit_ROOT="$ROOT/.cuda" -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DCUDA_cublas_LIBRARY="$ROOT/build/lib/libcublas.so" \
  -DCMAKE_CUDA_FLAGS="-I$ROOT/include/cublas" -DCMAKE_CXX_FLAGS="-I$ROOT/include/cublas" \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=$TESTS
targets=(llama-simple llama-completion llama-bench llama-perplexity)
[[ $TESTS == ON ]] && targets+=(test-backend-ops)
cmake --build "$OUT" -j"$(nproc)" --target "${targets[@]}"
echo "==> built into $OUT/bin"
