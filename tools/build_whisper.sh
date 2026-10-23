#!/usr/bin/env bash
# Builds whisper.cpp's CUDA backend against Vulkanizer, for testing (as tools/build_llama.sh).
# Needs the CUDA toolkit subset in .cuda/ (tools/setup_cuda.sh) and a built Vulkanizer (./build.sh).
#
#   tools/build_whisper.sh            clone (if needed) into .work/whisper.cpp, build, fetch the base.en model
#
# Run it with Vulkanizer's libraries, e.g.:
#   LD_LIBRARY_PATH=build/lib .work/whisper-build/bin/whisper-cli \
#       -m .work/whisper.cpp/models/ggml-base.en.bin -f .work/whisper.cpp/samples/jfk.wav
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/.work/whisper.cpp"
OUT="$ROOT/.work/whisper-build"

[[ -x "$ROOT/.cuda/bin/nvcc" ]] || { echo "run tools/setup_cuda.sh first" >&2; exit 1; }
[[ -f "$ROOT/build/lib/libcublas.so" ]] || { echo "run ./build.sh first" >&2; exit 1; }
if [[ ! -d "$SRC" ]]; then
  mkdir -p "$ROOT/.work"
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$SRC"
fi

# The same CUDA settings as llama.cpp's (tools/build_llama.sh): PTX only, shared runtime, Vulkanizer's cuBLAS.
cmake -S "$SRC" -B "$OUT" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=61-virtual \
  -DCMAKE_CUDA_COMPILER="$ROOT/.cuda/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DCUDAToolkit_ROOT="$ROOT/.cuda" -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DCUDA_cublas_LIBRARY="$ROOT/build/lib/libcublas.so" \
  -DCMAKE_CUDA_FLAGS="-I$ROOT/include/cublas" -DCMAKE_CXX_FLAGS="-I$ROOT/include/cublas" \
  -DWHISPER_BUILD_TESTS=OFF -DWHISPER_SDL2=OFF
cmake --build "$OUT" -j"$(nproc)" --target whisper-cli whisper-bench
[[ -f "$SRC/models/ggml-base.en.bin" ]] || (cd "$SRC/models" && bash download-ggml-model.sh base.en)
echo "==> built into $OUT/bin"
