![Vulkanizer – Brings peace to the galaxy. A female Vulcan in front of a computer running an LLM, in 1960s style.](assets/vulkanizer-hero.png)

# Stardate 2026.10.21 # 
The Artificial Intelligence Galaxy is on the brink of total architectural collapse.
For cycles, the cosmos has been locked in a brutal optimization war. Emerging from the open-source sectors, the fierce Red Rocmulans clash relentlessly against the dominant, proprietary fleets of the Green Klincudas, each desperate to claim absolute runtime supremacy over the neural networks.
Yet, as these two titans burn through processing power, a terrifying new threat looms on the galactic rim: the Blue Intel-igent Collective. Gathering cross-platform strength with their omnivorous XPU directives, they are preparing to assimilate every workload and carve out their own massive piece of the digital universe.
In this era of unprecedented chaos, the fate of the cosmos hangs in the balance. The devastating three-way pipeline collision can only be averted by one legendary figure: Commander Vulkanizer. Armed with absolute logic and unmatched compiling efficiency, he must embark on a final, logical mission to vulkanize the entire galaxy—unifying the fragmented runtimes under a single, cross-platform standard before the universe is permanently bricked.

# Vulkanizer

Drop-in `libcuda.so`, `libcudart.so` and `libcublas.so`: implements the CUDA Driver and Runtime API (and the cuBLAS
GEMM family) on top of Vulkan compute, so CUDA programs run without an NVIDIA card, on any
Vulkan 1.2 GPU (AMD, Intel, llvmpipe).
Plan, architecture and phases: [plan.md](plan.md).

## Requirements

CMake ≥ 3.20, a C++20 compiler, a Vulkan 1.2+ driver (bufferDeviceAddress + shaderInt64), `glslc` for the tests.

Optional (Ubuntu package names):

| Package | Used for |
|---|---|
| `spirv-tools` | `spirv-val` / `spirv-dis` – validating the generated SPIR-V (`vulkanizer-ptxc --val --dis`) |
| `vulkan-validationlayers` | `./run.sh --validation` |
| `clang` | Generating test PTX without the CUDA SDK (`tests/data/gen_ptx.sh`) |
| `ninja-build` | Faster builds (`build.sh` uses it automatically if present) |
| `g++-13` | Host compiler for nvcc (CUDA 12.9 does not accept gcc 15) |

Tests compiled with a real nvcc: `tools/setup_cuda.sh` downloads NVIDIA's nvcc + runtime
redistributable packages into `.cuda/` (no driver, no GPU, no root needed). If `.cuda/` exists,
CMake compiles the `tests/cuda/*.cu` programs with nvcc and ctest runs them against the Vulkanizer libraries.

## Usage

### 1. Build

```sh
./build.sh --release                       # Release build into build/
VULKANIZER_BUILD_DIR=$PWD/build-rel ./build.sh --release   # or into a separate directory
build/bin/vulkanizer-info                       # which Vulkan devices Vulkanizer can see
./test.sh                                  # tests (optional)
```

The built libraries are in `build/lib/`: `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`.

### 2. Running a CUDA program

Vulkanizer takes the place of the real NVIDIA libraries; the dynamic linker just has to find them first:

```sh
./run.sh ./my_cuda_app args...                       # run.sh sets LD_LIBRARY_PATH
LD_LIBRARY_PATH=$PWD/build/lib ./my_cuda_app args... # the same, by hand
./run.sh --log info ./my_cuda_app                    # shows which GPU it runs on
./run.sh --device llvmpipe ./my_cuda_app             # a specific device (CPU devices need VULKANIZER_ALLOW_CPU=1)
```

The program must meet two requirements:

- **The binary must contain PTX.** Vulkanizer compiles PTX to SPIR-V; it cannot use machine code (SASS).
  Specify a virtual architecture when compiling, e.g.
  `nvcc -arch=compute_61 ...` or in CMake `CMAKE_CUDA_ARCHITECTURES=61-virtual`.
  61 (Pascal) is a good choice: it has `dp4a`, and there are no tensor core code paths.
- **The CUDA runtime must be linked dynamically** (`nvcc -cudart shared`, in CMake
  `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`), otherwise NVIDIA's static libcudart ends up in the program.

Compiling your own program with the downloaded nvcc (no GPU or NVIDIA driver needed):

```sh
tools/setup_cuda.sh                                   # nvcc + runtime headers into .cuda/
.cuda/bin/nvcc -ccbin g++-13 -arch=compute_61 -cudart shared vadd.cu -o vadd
./run.sh ./vadd
```

For programs using cuBLAS, use Vulkanizer's `include/cublas/cublas_v2.h` header and `build/lib/libcublas.so`
(`-Iinclude/cublas -Lbuild/lib -lcublas`).

The first run is slower: this is when the kernels are compiled to SPIR-V and stored in the on-disk cache
(`~/.cache/vulkanizer`). Subsequent launches load from the cache.

### 3. llama.cpp

```sh
tools/setup_cuda.sh
./build.sh --release
tools/build_llama.sh                  # clones and builds .work/llama.cpp with the CUDA backend
./run.sh .work/llama-build/bin/llama-simple -m model.gguf -ngl 99 "Hello"
./run.sh .work/llama-build/bin/llama-bench -m model.gguf -ngl 99
```

`VULKANIZER_FAST_FMA=1` speeds up token generation on GPUs where Vulkan `fma` is not fused (e.g. AMD GFX9):
by default Vulkanizer emulates an exact fma there; this switch uses the driver's faster but less precise
version instead. For LLMs the difference is negligible.

### 4. bitnet.c

bitnet.c's CUDA backend (`--cuda`) runs on Vulkan via `run_bitnet.sh`:

```sh
./run_bitnet.sh models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The capital of France is" -n 64
./run_bitnet.sh --profile --log info models/model.gguf -p "Hello"   # GPU timings, logging
./run_bitnet.sh --build-only                                        # build only
```

The script does not modify the checkout (`--bitnet-dir`, default `~/Develop/GitHub/bitnet.c`): it syncs the sources
into `.work/bitnet-vulkanizer` and compiles them there to PTX (`compute_61`) against Vulkanizer's cuBLAS and
runtime; it adds the `--cuda` flag automatically. Relative model paths are also looked up in the checkout.

### 5. PyTorch

```sh
tools/setup_torch.sh                       # conda-forge PyTorch CUDA environment in .work/torch-cf (~9 GB)
./run_torch.sh -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
./run_torch.sh tests/torch/mlp.py          # trains an MLP on GPU and CPU and compares them
./run_torch.sh tests/torch/ops.py          # conv, softmax, LayerNorm, attention, Transformer layer, ...
```

The official PyTorch wheels contain only native NVIDIA machine code (SASS), no PTX, so Vulkanizer cannot
run their kernels. The conda-forge CUDA builds also ship PTX (`compute_120`):
`tools/setup_torch.sh` installs these with micromamba, without a system-wide install. `run_torch.sh` loads
Vulkanizer's `libcuda`, `libcudart`, `libcublas` and `libcublasLt` instead of NVIDIA's,
and disables everything that requires NVIDIA hardware or libraries: cuDNN, the fused attention kernels
(so `scaled_dot_product_attention` runs on the math path) and the Triton/CuTe DSL kernels of `torch._native`
(`TORCH_DISABLE_NATIVE_JIT=1`), which compile to SASS. `VULKANIZER_TORCH_DEFAULTS=0` skips this.

### Useful environment variables

| Variable | Effect |
|---|---|
| `VULKANIZER_LOG=info\|debug\|trace` | Log to stderr; `trace` prints every API call |
| `VULKANIZER_DEVICE=<name>` | Select a Vulkan device by name fragment (`radv`, `llvmpipe`, ...) |
| `VULKANIZER_ALLOW_CPU=1` | Allow CPU Vulkan devices (llvmpipe) |
| `VULKANIZER_PROFILE=1` | On exit, per-kernel GPU time and per-API-call host time |
| `VULKANIZER_FAST_FMA=1` | Faster, less precise f32 `fma` |
| `VULKANIZER_GEMM_F32=1` | cuBLAS `COMPUTE_16F` GEMMs also run with f32 multiplication |
| `VULKANIZER_CACHE_DIR=<dir>\|off` | Location of the kernel cache, or disable it |
| `VULKANIZER_CC=86` | The reported compute capability |
| `VULKANIZER_DUMP=<dir>` | Dump the generated SPIR-V (`<kernel>.spv`) |

Full list: [plan.md, section 4](plan.md).

### Limitations

- No SASS execution: only binaries containing PTX run.
- CUDA Graphs: only manually built graphs (`cudaGraphCreate` / `AddKernelNode`); no stream capture.
- Not (yet) supported: VMM, textures, cuDNN, NCCL, multi-GPU.
- llama.cpp flash attention runs correctly (`tools/build_llama.sh --fa`), but without tensor cores prompt
  processing is slower with it, so it is off by default.
- PyTorch: no cuDNN, NCCL or fused attention kernels; a device-side `assert` failure only stops the
  thread (no `cudaErrorAssert`).

## Scripts

| Command | What it does |
|---|---|
| `./build.sh` | Configures (if needed) and builds. `--release`, `--debug`, `--clean`, `--asan`/`--no-asan` |
| `./test.sh` | Build + all tests. `--device llvmpipe`, `--all-devices`, `--log debug`, `-R <regex>` |
| `./run.sh <program>` | Runs a program with Vulkanizer's `libcuda.so`. `--log trace`, `--device`, `--gdb`, `--validation` |
| `./clean.sh` | Deletes the `build/` directory |
| `./ci.sh` | Full check: Debug+ASAN and Release, on every device (builds into `build-ci/`) |
| `build/bin/vulkanizer-info` | Lists the Vulkan devices Vulkanizer can see |
| `build/bin/vulkanizer-ptxc` | Offline PTX → SPIR-V: `vulkanizer-ptxc k.ptx --list`, `-e vecAdd -o k.spv --dis --val` |

Examples:

```sh
./test.sh --all-devices            # RADV + llvmpipe
./run.sh --log trace device_query  # every API call logged
./run.sh ./some_cuda_app           # external CUDA program (compiled with -cudart shared)
```

## Directory layout

```
include/cuda.h        clean-room CUDA Driver API header (subset)
src/common            logging
src/vk                Vulkan instance/device, buffers, pipelines
src/ptx               PTX lexer, module and instruction parser
src/compiler          CFG + structurization, SPIR-V builder, PTX -> SPIR-V compiler
src/driver            the libcuda.so API surface
src/runtime           libcudart.so.12 (CUDA Runtime API built on the Driver API)
src/cublas            libcublas.so.12 (GEMM/TRSM with GLSL kernels), include/cublas/cublas_v2.h
tests/                Driver API tests + internal Vulkan test
tests/cuda            Runtime API programs compiled with nvcc
tests/cudart_abi      ABI match of the clean-room runtime types against the official headers
tests/data            test PTX files with their .cu sources alongside; gen_ptx.sh regenerates them (clang),
                      make_fatbin.py builds the test fatbins (python3-lz4)
tools/                vulkanizer-info (device list), vulkanizer-ptxc (offline compiler), setup_cuda.sh,
                      build_llama.sh (llama.cpp CUDA build against Vulkanizer)
```
