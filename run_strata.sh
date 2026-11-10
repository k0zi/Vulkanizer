#!/usr/bin/env bash
# Runs Strata's CUDA engine (https://github.com/Niko1221/Strata) on Vulkan through Vulkanizer.
#
#   ./run_strata.sh MODEL.gguf                    serve it: OpenAI/Anthropic API + web page on 127.0.0.1:8080
#   ./run_strata.sh --engine MODEL.gguf ARGS...   run the engine itself (e.g. --bench-pp 512 --bench-tg 64)
#   ./run_strata.sh --build-only                  only build
#
# Options (before the model):
#   --strata-dir DIR   Strata checkout (default: $STRATA_DIR or ~/Develop/GitHub/Strata)
#   --port N           server port (default 8080)
#   --context N        --max-context (default 8192)
#   --rebuild          clean build of the engine
#   --build-only       build, do not run
#   --engine           run the engine binary directly with the arguments after the model
#   --device NAME      Vulkan device by name substring (VULKANIZER_DEVICE)
#   --log LEVEL        Vulkanizer log level: off|error|warn|info|debug|trace (VULKANIZER_LOG)
#   --profile          per-kernel GPU times at exit (VULKANIZER_PROFILE=1)
#   --gdb              run the engine under gdb (with --engine)
#   --                 the rest goes to the engine (serve mode: appended to the config's args)
#
# Models: Qwen3.6-35B-A3B (GGUF architecture qwen35moe, e.g. Unsloth's UD-Q4_K_XL) and Qwen3.8-Flash-Next
# (qwen4exp, the first shard of a -0000N-of-0000M set). The MTP draft layer is off (--mtp 0 for Qwen3.6; no
# --mtp for Flash-Next, whose draft layer setup.py downloads separately).
#
# The checkout is not modified: the engine is built into .work/strata-build with the CUDA toolkit subset
# in .cuda/ (tools/setup_cuda.sh) as PTX only (compute_61 with Strata's STRATA_EXPERIMENTAL_SM60 code paths),
# against Vulkanizer's cuBLAS header and libraries. The tokenizer, the pack and the server config go to
# .work/strata-data. The server is Strata's own serve/server.py, run from the checkout's .venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VULKANIZER_BUILD="${VULKANIZER_BUILD_DIR:-$ROOT/build-rel}"
STRATA_SRC="${STRATA_DIR:-$HOME/Develop/GitHub/Strata}"
OUT="$ROOT/.work/strata-build"
DATA="$ROOT/.work/strata-data"

usage() { sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; }

PORT=8080
CONTEXT=8192
REBUILD=0
BUILD_ONLY=0
ENGINE=0
GDB=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --strata-dir) STRATA_SRC="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --context) CONTEXT="$2"; shift ;;
    --rebuild) REBUILD=1 ;;
    --build-only) BUILD_ONLY=1 ;;
    --engine) ENGINE=1 ;;
    --device) export VULKANIZER_DEVICE="$2"; shift ;;
    --log) export VULKANIZER_LOG="$2"; shift ;;
    --profile) export VULKANIZER_PROFILE=1 ;;
    --gdb) GDB=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage; exit 1 ;;
    *) break ;;
  esac
  shift
done

[[ -f "$STRATA_SRC/src/program/generate.cpp" ]] || { echo "no Strata checkout at $STRATA_SRC" >&2; exit 1; }
STRATA_SRC="$(cd "$STRATA_SRC" && pwd)"
[[ $# -gt 0 || $BUILD_ONLY -eq 1 ]] || { usage; exit 1; }

# ---- Vulkanizer and the CUDA toolkit subset ----------------------------------------------------------
[[ -x "$ROOT/.cuda/bin/nvcc" ]] || "$ROOT/tools/setup_cuda.sh"
VULKANIZER_BUILD_DIR="$VULKANIZER_BUILD" "$ROOT/build.sh" --release >/dev/null
echo "==> Vulkanizer: $VULKANIZER_BUILD/lib" >&2

# ---- the engine ---------------------------------------------------------------------------------------
# g++-13 for the C++ sources too: nvcc (CUDA 12.9 accepts gcc <= 13) links the program, and objects from a
# newer gcc need its libstdc++ (__cxa_call_terminate). The toolkit's libcuda is a stub without the symbols
# Vulkanizer's libcublas takes from its own libcuda, so the driver library is Vulkanizer's.
[[ $REBUILD -eq 1 ]] && rm -rf "$OUT"
if [[ ! -f "$OUT/CMakeCache.txt" ]]; then
  GEN=()
  command -v ninja >/dev/null && GEN=(-G Ninja)
  cmake -S "$STRATA_SRC" -B "$OUT" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release -Wno-dev \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DSTRATA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=61-virtual \
    -DCMAKE_CUDA_COMPILER="$ROOT/.cuda/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
    -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
    -DCUDAToolkit_ROOT="$ROOT/.cuda" -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
    -DCUDA_cublas_LIBRARY="$VULKANIZER_BUILD/lib/libcublas.so" \
    -DCUDA_cuda_driver_LIBRARY="$VULKANIZER_BUILD/lib/libcuda.so" \
    -DCMAKE_CUDA_FLAGS="-I$ROOT/include/cublas" -DCMAKE_CXX_FLAGS="-I$ROOT/include/cublas" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$VULKANIZER_BUILD/lib" >&2
fi
cmake --build "$OUT" -j"$(nproc)" --target strata >&2
EXE="$OUT/strata"
echo "==> engine: $EXE" >&2
[[ $BUILD_ONLY -eq 1 ]] && exit 0

# ---- the model ----------------------------------------------------------------------------------------
MODEL="$(realpath "$1")"
shift
[[ "${1:-}" == "--" ]] && shift
[[ -f "$MODEL" ]] || { echo "no such model file: $MODEL" >&2; exit 1; }

PY="$STRATA_SRC/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  python3 -m venv "$STRATA_SRC/.venv"
fi
"$PY" -c "import numpy, jinja2, regex, yaml, requests, psutil" 2>/dev/null ||
  "$PY" -m pip install -q -r "$STRATA_SRC/requirements.txt"

ARCH="$("$PY" -I -c "import sys, pathlib; sys.path.insert(0, sys.argv[1]); import gguf_reader as G; \
print(G.GGUFFile(pathlib.Path(sys.argv[2])).metadata.get('general.architecture', ''))" "$STRATA_SRC/tools" "$MODEL")"
TAG="$(basename "$MODEL" .gguf | sed 's/-0000[0-9]-of-0000[0-9]$//' | tr 'A-Z' 'a-z')"
PACK="$DATA/packs/$TAG"
mkdir -p "$PACK"
export STRATA_GGUF_PY="$OUT/_deps/strata_llamacpp-src/gguf-py"

case "$ARCH" in
  qwen35moe)
    # only the tokenizer: the engine reads the GGUF itself
    [[ -f "$PACK/tokenizer/vocab.json" ]] ||
      "$PY" "$STRATA_SRC/tools/strata_tokenizer.py" --gguf "$MODEL" --out "$PACK" >&2
    MODEL_ARGS=(--native "$MODEL" --prefill auto --mtp 0)
    ;;
  qwen4exp)
    # every tensor as the GGUF stores it (seconds); the experts are read from the GGUF at start
    [[ -f "$PACK/native_experts.txt" && -f "$PACK/tokenizer/vocab.json" ]] ||
      "$PY" "$STRATA_SRC/tools/iq_pack.py" --gguf "$MODEL" --out "$PACK" >&2
    PLE="$("$PY" -I -c "import sys, pathlib; sys.path.insert(0, sys.argv[1]); import gguf_reader as G
m = pathlib.Path(sys.argv[2]); shards = sorted(m.parent.glob(m.name.replace('00001-of', '0000?-of'))) or [m]
print(next((str(s) for s in shards if any(t.name == 'per_layer_token_embd.weight' for t in G.GGUFFile(s).tensors)), ''))" \
      "$STRATA_SRC/tools" "$MODEL")"
    [[ -n "$PLE" ]] || { echo "the model has no per_layer_token_embd tensor (all shards must be beside it)" >&2; exit 1; }
    MODEL_ARGS=(--pack "$PACK" --native "$MODEL" --ple-gguf "$PLE"
                --expert-profile "$STRATA_SRC/data/expert-profile.bin" --expert-cache auto --prefill auto)
    ;;
  *) echo "unsupported model architecture '$ARCH' (Strata runs qwen35moe and qwen4exp)" >&2; exit 1 ;;
esac

export LD_LIBRARY_PATH="$VULKANIZER_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ $ENGINE -eq 1 ]]; then
  CMD=("$EXE" "${MODEL_ARGS[@]}" --max-context "$CONTEXT" "$@")
  [[ $GDB -eq 1 ]] && exec gdb --args "${CMD[@]}"
  exec "${CMD[@]}"
fi

# ---- the server ---------------------------------------------------------------------------------------
CFG="$DATA/strata-$TAG.json"
"$PY" -I - "$CFG" "$EXE" "$STRATA_SRC" "$PACK" "$TAG" "$ARCH" "$VULKANIZER_BUILD/lib" "$PORT" \
  "${MODEL_ARGS[@]}" --max-context "$CONTEXT" "$@" <<'EOF'
import json, sys
cfg, exe, src, pack, tag, arch, lib, port, *args = sys.argv[1:]
json.dump({"exe": exe, "args": args, "cwd": src, "tokenizer": pack + "/tokenizer", "model_name": tag,
           "arch": arch, "log": cfg[:-5] + ".log", "lib_dirs": [lib], "port": int(port), "cuda": 12},
          open(cfg, "w"), indent=1)
EOF
echo "==> config: $CFG" >&2
cd "$STRATA_SRC"
exec "$PY" "$STRATA_SRC/serve/server.py" --engine strata --config "$CFG" --port "$PORT"
