![Vulkanizer – Brings peace to the galaxy. A female Vulcan in front of a computer running an LLM, in 1960s style.](assets/vulkanizer-hero.png)

**English** | [Deutsch](README.de.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Magyar](README.hu.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# Stardate 2026.10.21 # 
The Artificial Intelligence Galaxy is on the brink of total architectural collapse.
For cycles, the cosmos has been locked in a brutal optimization war. Emerging from the open-source sectors, the fierce Red Rocmulans clash relentlessly against the dominant, proprietary fleets of the Green Klincudas, each desperate to claim absolute runtime supremacy over the neural networks.
Yet, as these two titans burn through processing power, a terrifying new threat looms on the galactic rim: the Blue Intel-igent Collective. Gathering cross-platform strength with their omnivorous XPU directives, they are preparing to assimilate every workload and carve out their own massive piece of the digital universe.
In this era of unprecedented chaos, the fate of the cosmos hangs in the balance. The devastating three-way pipeline collision can only be averted by one legendary figure: Commander Vulkanizer. Armed with absolute logic and unmatched compiling efficiency, he must embark on a final, logical mission to vulkanize the entire galaxy—unifying the fragmented runtimes under a single, cross-platform standard before the universe is permanently bricked.

# Vulkanizer

Drop-in `libcuda.so`, `libcudart.so`, `libcublas.so`, `libcusolver.so`, `libcufft.so`, `libcusparse.so` and
`libcurand.so`: implements the CUDA Driver and Runtime API (and the cuBLAS GEMM family, dense and sparse linear algebra,
FFTs and random numbers) on top of Vulkan compute, so CUDA programs run without an NVIDIA card, on any Vulkan 1.2 GPU
(AMD, Intel, llvmpipe).
Plan, architecture and phases: [plan.md](plan.md) (in Hungarian).

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
`tools/setup_cuda.sh --cuda13` does the same with CUDA 13.4 into `.cuda13/`: the programs are built again
with nvcc 13 and run on `libcudart.so.13` (tests ending in `_13`).

## Usage

### 1. Build

```sh
./build.sh --release                       # Release build into build/
VULKANIZER_BUILD_DIR=$PWD/build-rel ./build.sh --release   # or into a separate directory
build/bin/vulkanizer-info                       # which Vulkan devices Vulkanizer can see
./test.sh                                  # tests (optional)
```

The built libraries are in `build/lib/`: `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`,
`libcusolver.so.11`, `libcufft.so.11`, `libcusparse.so.12`, `libcurand.so.10`, `libnccl.so.2`. `build/lib/cuda13/` holds
the CUDA 13 versions of the runtime and the libraries (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`,
`libcusolver.so.12`, `libcufft.so.12`, `libcusparse.so.12`, `libcurand.so.10`), for programs built with CUDA 13; they
use the same `libcuda.so.1`.

#### Installing (package)

```sh
tools/package.sh                               # build-pkg/vulkanizer_<version>_<arch>.deb and a .tar.gz
sudo apt install ./build-pkg/vulkanizer_*.deb
vulkanizer-run ./my_cuda_app args...           # any CUDA 12 or 13 program
vulkanizer-run --torch python script.py        # PyTorch from pip or conda (CUDA 12 or 13, detected)
```

The libraries go to `/usr/lib/<arch>/vulkanizer/`, outside the linker's search path: an NVIDIA driver on the same
machine keeps working for everything else, and `vulkanizer-run` puts Vulkanizer in front only for the program it
starts. `cmake --install build --prefix DIR` installs the same layout anywhere, without a package.

### 2. Running a CUDA program

Vulkanizer takes the place of the real NVIDIA libraries; the dynamic linker just has to find them first:

```sh
./run.sh ./my_cuda_app args...                       # run.sh sets LD_LIBRARY_PATH
LD_LIBRARY_PATH=$PWD/build/lib ./my_cuda_app args... # the same, by hand
./run.sh --log info ./my_cuda_app                    # shows which GPU it runs on
./run.sh --device llvmpipe ./my_cuda_app             # a specific device (CPU devices need VULKANIZER_ALLOW_CPU=1)
./run.sh --check ./my_cuda_app                       # can it run? (PTX, shared CUDA runtime; also a directory or a Python env)
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

On GPUs where Vulkan `fma` is not fused (e.g. AMD GFX9), Vulkanizer emulates CUDA's single-rounding `fma`
for IEEE code, but uses the driver's fast `fma` for code built with fast math (`-ftz`, such as llama.cpp's CUDA backend).
`VULKANIZER_FAST_FMA=1` uses the fast one everywhere, `VULKANIZER_EXACT_FMA=1` emulates it everywhere.

whisper.cpp (speech recognition, the same ggml CUDA backend) builds and runs the same way:

```sh
tools/build_whisper.sh   # into .work/whisper-build, with the base.en model
LD_LIBRARY_PATH=build/lib .work/whisper-build/bin/whisper-cli \
    -m .work/whisper.cpp/models/ggml-base.en.bin -f .work/whisper.cpp/samples/jfk.wav
```

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
./run_torch.sh torch_mlp                   # trains an MLP on GPU and CPU and compares them (libtorch, C++)
./run_torch.sh torch_ops                   # conv, softmax, LayerNorm, attention, Transformer layer, ...
./run_torch.sh torch_graph                 # a forward pass captured into a CUDA graph and replayed
./run_torch.sh tests/torch/triton_ops.py   # a hand-written Triton kernel and torch._native's Triton bmm
./run_torch.sh tests/torch/compile.py      # torch.compile (Inductor -> Triton), also mode="reduce-overhead"

tools/setup_torch.sh --cuda13              # the CUDA 13.4 build in .work/torch-cf13
./run_torch.sh --env .work/torch-cf13 torch13_ops
```

`torch_mlp`, `torch_ops`, `torch_split` and `torch_graph` are C++ tests against the environment's libtorch
(`tests/torch/*.cpp`); CMake builds them when `.work/torch-cf` exists at configure time, and as `torch13_*`
against `.work/torch-cf13`. For an environment with `libcudart.so.13` (a CUDA 13 build), `run_torch.sh`
loads the libraries from `build/lib/cuda13`.

The official PyTorch wheels contain only native NVIDIA machine code (SASS), no PTX, so Vulkanizer cannot run their
kernels. The conda-forge CUDA builds also ship PTX (`compute_120`): `tools/setup_torch.sh` installs these with
micromamba, without a system-wide install. `run_torch.sh` loads Vulkanizer's `libcuda`, `libcudart`, `libcublas`,
`libcublasLt`, `libcusolver`, `libcufft`, `libcusparse` and `libcurand` instead of NVIDIA's, and disables everything
that requires NVIDIA hardware or libraries: cuDNN, the fused attention kernels (so `scaled_dot_product_attention` runs
on the math path) and the DSL kernels of `torch._native` (`TORCH_DISABLE_NATIVE_JIT=1`): its Triton kernels work, but
its CuTe DSL kernels compile to SASS and the one switch turns off both. `VULKANIZER_TORCH_DEFAULTS=0` skips this. Triton
itself (`torch.compile`, `@triton.jit`) works: `tools/triton/ptxas` hands Triton's PTX through instead of assembling it,
and Triton's launcher is built against Vulkanizer's `include/cuda.h` (needs a C compiler). `torch.linalg` (LU, Cholesky,
QR, least squares, `eigh`, SVD, `pinv`, ...) runs on Vulkanizer's cuSOLVER and `torch.fft` (including `stft`) on its
cuFFT, `torch.sparse` on its cuSPARSE; `run_torch.sh` sets `TORCH_LINALG_PREFER_CUSOLVER=1`, since MAGMA's own kernels
do not run. PyTorch's jiterator compiles its kernels at run time with NVRTC to native code; an `LD_AUDIT` library
(`libvulkanizer_nvrtc.so`) makes it hand over the PTX instead.

### 6. ComfyUI

ComfyUI runs through `run_torch.sh` too, in its own conda-forge environment inside the ComfyUI checkout:

```sh
VK=~/Develop/GitHub/Vulkanizer             # this repository, built with ./build.sh --release
cd ~/Develop/GitHub                        # wherever ComfyUI should live
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI

# PyTorch (CUDA 12.9 build with PTX) and torchvision's native dependencies
export MAMBA_ROOT_PREFIX=$VK/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
[ -x $VK/.work/micromamba/bin/micromamba ] || { mkdir -p $VK/.work/micromamba && curl -fsSL \
    https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $VK/.work/micromamba bin/micromamba; }
$VK/.work/micromamba/bin/micromamba create -y -p $PWD/venv-vulkanizer -c conda-forge \
    python=3.13 "pytorch=2.14.1=cuda129_generic*" pip pillow giflib libwebp-base libjpeg-turbo libpng \
    "libnvjpeg>=12.4,<13" "libcusolver<12" "libcusparse<13"
# torchvision by URL, without solving: its CUDA 12.9 build asks for a newer cuDNN that needs the
# __cuda_arch virtual package (a real driver). cuDNN is off under Vulkanizer anyway.
$VK/.work/micromamba/bin/micromamba install -y -p $PWD/venv-vulkanizer --file <(printf '@EXPLICIT\n%s\n' \
    https://conda.anaconda.org/conda-forge/linux-64/torchvision-0.29.1-cuda129_py313_hcd9f930_0.conda)

# the rest of ComfyUI's requirements; pip must not replace torch/torchvision with the SASS-only wheels
venv-vulkanizer/bin/pip install -r <(grep -vE '^(torch|torchvision)$' requirements.txt) -r manager_requirements.txt

# start it
$VK/run_torch.sh --env venv-vulkanizer main.py --enable-manager --disable-cuda-malloc
```

Then open <http://127.0.0.1:8188>. Models go into `models/` as usual. Notes:

- `--disable-cuda-malloc`: ComfyUI's `cudaMallocAsync` allocator backend is not used.
- DynamicVRAM (comfy-aimdo) runs on Vulkanizer's virtual memory management (VMM, sparse buffers). If it causes
  trouble, `--disable-dynamic-vram` switches back to estimate based model loading.
- `tools/torch/sitecustomize.py` applies the PyTorch settings only when ComfyUI imports torch, and keeps
  the fused attention backends off even though ComfyUI re-enables them: attention runs on the math path.
- The ComfyUI-Manager "PyTorch is not installed" message comes from the missing `torchaudio` (conda-forge has
  no build for torch 2.14); image generation does not need it.
- The first generation is slow while the kernels are compiled to SPIR-V; later runs load them from the cache.
- `$VK/run_torch.sh --log info --env venv-vulkanizer main.py ...` shows which Vulkan GPU is used.
- `python3 tests/torch/zimage_forward.py` checks the GPU against the CPU: one Z-Image Turbo forward at 256×256 in
  fp16 on the GPU and in fp32 and fp16 on the CPU. The GPU must be as far from fp32 as fp16 itself is (ComfyUI runs
  the model in fp16 on the GPU, so full images differ from a CPU fp32 run in fine detail).

### 7. Numba

Numba (`numba-cuda`, `@cuda.jit`) compiles kernels with libNVVM to PTX, which Vulkanizer runs (`CUDA_HOME` lets it
find libNVVM in a conda environment):

```sh
export MAMBA_ROOT_PREFIX=$PWD/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
.work/micromamba/bin/micromamba create -y -p .work/numba-cf -c conda-forge \
    python=3.13 numba numba-cuda "cuda-version=12.9" cuda-nvvm cuda-nvrtc numpy
CUDA_HOME=$PWD/.work/numba-cf ./run_torch.sh --env .work/numba-cf script.py
```

Its linker is nvJitLink, which would make native code: under `run_torch.sh` (and `vulkanizer-run --torch`) nvJitLink's
link calls go to the driver's `cuLink*`, whose linked image is PTX here, and nvJitLink reports no version, so Numba
emits PTX rather than LTO-IR. CuPy does not run: it links the CUDA runtime statically (see Limitations).

### Useful environment variables

| Variable | Effect |
|---|---|
| `VULKANIZER_LOG=info\|debug\|trace` | Log to stderr; `trace` prints every API call |
| `VULKANIZER_DEVICE=<name>` | Select a Vulkan device by name fragment (`radv`, `llvmpipe`, ...) |
| `VULKANIZER_ALLOW_CPU=1` | Allow CPU Vulkan devices (llvmpipe) |
| `VULKANIZER_PROFILE=1` | On exit, per-kernel GPU time and per-API-call host time |
| `VULKANIZER_BATCH_MS=<ms>` | GPU time budget of one queue submission (default 50; `0` = no limit). Longer kernels are split, so the kernel driver's GPU timeout (amdgpu: 2 s) is not hit |
| `CUDA_LAUNCH_BLOCKING=1` | Every kernel launch waits for the GPU, so errors and hangs show at the launch that caused them |
| `VULKANIZER_FAST_FMA=1` | Faster, less precise f32 `fma` |
| `VULKANIZER_EXACT_FMA=1` | Exact f32 `fma` for fast-math (`.ftz`) code too (slower) |
| `VULKANIZER_GEMM_F32=1` | cuBLAS `COMPUTE_16F` GEMMs also run with f32 multiplication |
| `VULKANIZER_IPC=0` | Device memory is not exportable to other processes (no CUDA IPC) |
| `VULKANIZER_CACHE_DIR=<dir>\|off` | Location of the kernel cache, or disable it |
| `VULKANIZER_CACHE_SIZE=<MiB>` | Kernel cache size limit in MiB (default 256); the least recently used entries go first |
| `VULKANIZER_CC=86` | The reported compute capability |
| `VULKANIZER_DUMP=<dir>` | Dump the generated SPIR-V (`<kernel>.spv`) |

Full list: [plan.md, section 4](plan.md).

### Limitations

- No SASS execution: only binaries containing PTX run.
- Programs and libraries with the CUDA runtime linked in statically (CuPy, NVIDIA's NCCL) do not run: that runtime
  needs undocumented internals of NVIDIA's driver.
- CUDA Graphs: stream capture and replay work (llama.cpp's graphs, PyTorch `CUDAGraph`,
  `torch.compile(mode="reduce-overhead")`); child graphs and memory nodes (stream-ordered allocation during
  capture) too, and conditional nodes (if, while, switch; the host waits for the GPU at each of them).
- Not (yet) supported: cuDNN.
- Device code: no recursion. Indirect calls (virtual functions, function pointers) compare the target with every
  function of the module of that shape, and only reach functions of the same module. `malloc`/`free` work on a heap
  of `cudaLimitMallocHeapSize` (default 8 MiB), which needs 64-bit atomics.
- Textures and surfaces (texture and surface objects, CUDA arrays) are sampled in software, with CUDA's addressing
  and filtering, mipmaps (level of detail and gradients) and cube maps (filtered within a face, not across its
  edges). No texture references (removed in CUDA 12) or formatted surface stores (`sust.p`, which CUDA C does not emit).
- CUDA IPC (`cudaIpc*`, PyTorch's `torch.multiprocessing`) works between processes of the same user; an interprocess
  event is complete once recorded (recording it waits for the device).
- NCCL (`libnccl.so.2`, PyTorch's `torch.distributed` "nccl" backend: DDP, FSDP) works between the ranks of one
  machine, which may share a GPU; with more than one rank the data goes through host memory.
- Several GPUs are separate devices: `cudaMemcpyPeer` and plain copies between two devices' pointers (unified
  addressing, e.g. PyTorch's `tensor.to("cuda:1")`) go through host memory, but a kernel never reaches another GPU's
  memory (no peer access). Cooperative launches (`grid.sync()`) work up to the number of blocks that fit on the GPU at
  once.
- Managed, pinned and registered memory are host memory the GPU uses in place at the same address; managed memory
  never migrates to the GPU (on an integrated GPU it is the same memory anyway).
- Registered host memory (`cuMemHostRegister`) that is a mapping of its own (`mmap`) at an address the GPU can place
  memory at (RADV: 64 KiB aligned, from 2 MiB on 2 MiB aligned) is replaced by GPU-visible memory with the same
  contents: registering and unregistering copy it, and after `fork()` the child shares those pages instead of copying
  them. Other registered memory is imported, and while it exists every GPU submission is slower (the kernel
  revalidates the imported pages, ~4 ms per 256 MB). Pinned and managed memory are not affected.
- llama.cpp flash attention runs correctly (`tools/build_llama.sh --fa`), but without tensor cores prompt
  processing is slower with it, so it is off by default.
- PyTorch: no cuDNN or fused attention kernels.
- cuSOLVER: real types (f32, f64) only, no `gesvda`, LDL^T (`sytrf`) or non-symmetric eigenproblems (`geev`); `eigh`
  of one matrix tridiagonalizes it (Householder) and solves on the host (QL); batched `eigh` and SVD use Jacobi
  iterations, the SVD is slow for large matrices (n ≳ 1000). cuFFT: no callbacks or multi-GPU plans; transforms longer
  than 2048 (f64: 1024) take two passes over memory where their length splits into two factors within that limit,
  otherwise one per radix. cuSPARSE: no block sparse (BSR) or
  complex values; SpGEMM and the sorting routines run on the host.
- cuRAND: XORWOW, MRG32k3a, Philox and Sobol give NVIDIA's sequences (normals and log-normals equal to rounding);
  MTGP32 and MT19937 are accepted but give other sequences, as do scrambled Sobol and Poisson; double outputs need
  `shaderFloat64`. The device API (`curand_kernel.h`) works, except for distributions made on the host
  (`curand_discrete`).
- ComfyUI: no CUDA 13 (cu130) optimized paths.

## Scripts

| Command | What it does |
|---|---|
| `./build.sh` | Configures (if needed) and builds. `--release`, `--debug`, `--clean`, `--asan`/`--no-asan` |
| `./test.sh` | Build + all tests. `--device llvmpipe`, `--all-devices`, `--log debug`, `-R <regex>` |
| `./run.sh <program>` | Runs a program with Vulkanizer's `libcuda.so`. `--log trace`, `--device`, `--gdb`, `--validation` |
| `./clean.sh` | Deletes the `build/` directory |
| `./ci.sh` | Full check: Debug+ASAN and Release, on every device (builds into `build-ci/`) |
| `tools/package.sh` | Release packages: a `.deb` and a `.tar.gz` in `build-pkg/` (`--check` also tries the `.deb`) |
| `build/bin/vulkanizer-info` | Lists the Vulkan devices Vulkanizer can see |
| `build/bin/vulkanizer-ptxc` | Offline PTX → SPIR-V: `vulkanizer-ptxc k.ptx --list`, `-e vecAdd -o k.spv --dis --val` |
| `build/bin/vulkanizer-bandwidth` | Copy bandwidth (HtoD/DtoH/DtoD, pageable and pinned) and allocation cost: `vulkanizer-bandwidth [MiB]` |

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
src/ptx/texture_lib.cu  texture and surface sampling in software (embedded as PTX, tools/gen_texture_lib.sh)
src/compiler          CFG + structurization, SPIR-V builder, PTX -> SPIR-V compiler
src/driver            the libcuda.so API surface
src/runtime           libcudart.so.12 and .so.13 (CUDA Runtime API built on the Driver API, one source)
src/cublas            libcublas.so.12/.13 (GEMM/TRSM with GLSL kernels), include/cublas/cublas_v2.h
src/linalg            dense factorizations (LU, Cholesky, QR, eigh, Jacobi SVD) in GLSL, shared by cuBLAS and cuSOLVER
src/cusolver          libcusolver.so.11/.12, include/cusolver/cusolverDn.h
src/cufft             libcufft.so.11/.12 (Stockham and Bluestein FFTs in GLSL), include/cufft/cufft.h
src/cusparse          libcusparse.so.12 (SpMV/SpMM/SDDMM/SpSV/csrgeam2 in GLSL), include/cusparse/cusparse.h
src/curand            libcurand.so.10 (XORWOW/MRG32k3a/Philox/Sobol in GLSL; Sobol data: tools/gen_sobol_table.sh), include/curand/curand.h
src/nccl              libnccl.so.2 (collectives between the ranks of one machine), include/nccl/nccl.h
src/nvrtc             LD_AUDIT library: NVRTC hands over PTX instead of a cubin (PyTorch's jiterator), nvJitLink links through cuLink* (Numba)
tests/                Driver API tests + internal Vulkan test
tests/cuda            Runtime API programs compiled with nvcc
tests/cudart_abi      ABI match of the clean-room runtime types against the official headers
tests/data            test PTX files with their .cu sources alongside; gen_ptx.sh regenerates them (clang)
tools/                vulkanizer-info (device list), vulkanizer-ptxc (offline compiler), vulkanizer-bandwidth, setup_cuda.sh,
                      build_llama.sh (llama.cpp CUDA build against Vulkanizer)
                      build_whisper.sh (whisper.cpp)
```

## License and legal notices

Vulkanizer is released under the [MIT License](LICENSE). Third-party material it includes (the Sobol direction numbers
of Joe and Kuo, BSD license) is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Vulkanizer contains no NVIDIA source code, headers or binaries: its CUDA headers are clean-room declarations written
from the public documentation, sharing only what binary compatibility requires. The CUDA toolkit that
`tools/setup_cuda.sh` downloads comes from NVIDIA's servers under NVIDIA's own license and is not part of Vulkanizer.

NVIDIA, CUDA, cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NCCL and PTX are trademarks or registered trademarks of NVIDIA
Corporation; Vulkan and SPIR-V are registered trademarks of the Khronos Group Inc. They are used only to identify what
Vulkanizer is compatible with. Vulkanizer is an independent project, not affiliated with, endorsed or sponsored by
NVIDIA or the Khronos Group.
