![Vulkanizer —— 为银河系带来和平。一位瓦肯女性坐在运行 LLM 的计算机前，1960 年代风格。](assets/vulkanizer-hero.png)

[English](README.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Magyar](README.hu.md) | **中文** | [日本語](README.ja.md)

# 星历 2026.10.21 #
人工智能银河系正处于架构全面崩溃的边缘。
多个周期以来，宇宙深陷一场残酷的优化战争。来自开源星区的凶悍红色洛克穆兰人，与占据统治地位、闭源的绿色克林库达舰队无休止地交锋，双方都拼命想夺取对神经网络的绝对运行时霸权。
然而，就在这两大巨头燃烧算力之际，一个可怕的新威胁正在银河边缘浮现：蓝色英特-智能集合体。它凭借无所不吞的 XPU 指令积蓄跨平台力量，准备同化所有工作负载，并在数字宇宙中划出属于自己的一大片疆域。
在这个前所未有的混乱时代，宇宙的命运危在旦夕。唯有一位传奇人物能够阻止这场毁灭性的三方流水线碰撞：Vulkanizer 指挥官。他以绝对的逻辑和无与伦比的编译效率为武器，必须踏上最后一次合乎逻辑的使命，将整个银河系"硫化"——在宇宙彻底变砖之前，把四分五裂的运行时统一到一个跨平台的标准之下。

# Vulkanizer

可直接替换 `libcuda.so`、`libcudart.so` 和 `libcublas.so`：在 Vulkan 计算之上实现 CUDA Driver API 与 Runtime API
（以及 cuBLAS 的 GEMM 系列），让 CUDA 程序无需 NVIDIA 显卡即可在任何支持 Vulkan 1.2 的 GPU（AMD、Intel、llvmpipe）上运行。
计划、架构与阶段：[plan.md](plan.md)（匈牙利语）。

## 环境要求

CMake ≥ 3.20、支持 C++20 的编译器、Vulkan 1.2+ 驱动（bufferDeviceAddress + shaderInt64），测试需要 `glslc`。

可选（Ubuntu 软件包名）：

| 软件包 | 用途 |
|---|---|
| `spirv-tools` | `spirv-val` / `spirv-dis` —— 校验生成的 SPIR-V（`vulkanizer-ptxc --val --dis`） |
| `vulkan-validationlayers` | `./run.sh --validation` |
| `clang` | 无需 CUDA SDK 即可生成测试用 PTX（`tests/data/gen_ptx.sh`） |
| `ninja-build` | 更快的构建（若已安装，`build.sh` 会自动使用） |
| `g++-13` | nvcc 的主机编译器（CUDA 12.9 不接受 gcc 15） |

用真正的 nvcc 编译的测试：`tools/setup_cuda.sh` 会把 NVIDIA 的 nvcc 和运行时（可再分发软件包）下载到 `.cuda/`
（不需要驱动、GPU 或 root 权限）。如果 `.cuda/` 存在，CMake 会用 nvcc 编译 `tests/cuda/*.cu` 中的程序，
ctest 会让它们在 Vulkanizer 的库上运行。
`tools/setup_cuda.sh --cuda13` 用 CUDA 13.4 在 `.cuda13/` 中做同样的事：这些程序会用 nvcc 13 重新编译，
并在 `libcudart.so.13` 上运行（名称以 `_13` 结尾的测试）。

## 使用方法

### 1. 构建

```sh
./build.sh --release                       # Release 构建，输出到 build/
VULKANIZER_BUILD_DIR=$PWD/build-rel ./build.sh --release   # 或输出到单独的目录
build/bin/vulkanizer-info                       # Vulkanizer 能看到哪些 Vulkan 设备
./test.sh                                  # 测试（可选）
```

构建出的库位于 `build/lib/`：`libcuda.so.1`、`libcudart.so.12`、`libcublas.so.12`、`libcublasLt.so.12`。
`build/lib/cuda13/` 中是运行时和 cuBLAS 的 CUDA 13 版本（`libcudart.so.13`、`libcublas.so.13`、`libcublasLt.so.13`），
供用 CUDA 13 构建的程序使用；它们使用同一个 `libcuda.so.1`。

### 2. 运行 CUDA 程序

Vulkanizer 取代真正的 NVIDIA 库；只需让动态链接器先找到它们：

```sh
./run.sh ./my_cuda_app args...                       # run.sh 会设置 LD_LIBRARY_PATH
LD_LIBRARY_PATH=$PWD/build/lib ./my_cuda_app args... # 手动做同样的事
./run.sh --log info ./my_cuda_app                    # 显示运行在哪个 GPU 上
./run.sh --device llvmpipe ./my_cuda_app             # 指定设备（CPU 设备需要 VULKANIZER_ALLOW_CPU=1）
```

程序必须满足两个条件：

- **二进制文件中必须包含 PTX。** Vulkanizer 把 PTX 编译为 SPIR-V，无法使用机器码（SASS）。编译时请指定虚拟架构，例如
  `nvcc -arch=compute_61 ...`，或在 CMake 中使用 `CMAKE_CUDA_ARCHITECTURES=61-virtual`。
  61（Pascal）是个好选择：它有 `dp4a`，也没有 tensor core 代码路径。
- **CUDA 运行时必须动态链接**（`nvcc -cudart shared`，CMake 中为 `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`），
  否则 NVIDIA 的静态 libcudart 会被编进程序。

用下载的 nvcc 编译自己的程序（不需要 GPU 和 NVIDIA 驱动）：

```sh
tools/setup_cuda.sh                                   # nvcc + 运行时头文件，放到 .cuda/
.cuda/bin/nvcc -ccbin g++-13 -arch=compute_61 -cudart shared vadd.cu -o vadd
./run.sh ./vadd
```

使用 cuBLAS 的程序请使用 Vulkanizer 的 `include/cublas/cublas_v2.h` 头文件和 `build/lib/libcublas.so`
（`-Iinclude/cublas -Lbuild/lib -lcublas`）。

第一次运行会比较慢：此时内核会被编译成 SPIR-V 并存入磁盘缓存（`~/.cache/vulkanizer`）。之后的启动会从缓存加载。

### 3. llama.cpp

```sh
tools/setup_cuda.sh
./build.sh --release
tools/build_llama.sh                  # 克隆 .work/llama.cpp 并以 CUDA 后端构建
./run.sh .work/llama-build/bin/llama-simple -m model.gguf -ngl 99 "Hello"
./run.sh .work/llama-build/bin/llama-bench -m model.gguf -ngl 99
```

在 Vulkan `fma` 不是融合运算的 GPU（例如 AMD GFX9）上，`VULKANIZER_FAST_FMA=1` 可以加快 token 生成：
默认情况下 Vulkanizer 会在这些 GPU 上模拟精确的 fma，此开关改用驱动更快但精度较低的版本。对 LLM 来说差别可以忽略。

### 4. bitnet.c

bitnet.c 的 CUDA 后端（`--cuda`）通过 `run_bitnet.sh` 在 Vulkan 上运行：

```sh
./run_bitnet.sh models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The capital of France is" -n 64
./run_bitnet.sh --profile --log info models/model.gguf -p "Hello"   # GPU 计时、日志
./run_bitnet.sh --build-only                                        # 只构建
```

该脚本不会修改源码检出目录（`--bitnet-dir`，默认 `~/Develop/GitHub/bitnet.c`）：它把源码同步到
`.work/bitnet-vulkanizer`，并在那里针对 Vulkanizer 的 cuBLAS 和运行时编译成 PTX（`compute_61`）；
`--cuda` 参数会自动加上。相对的模型路径也会在检出目录中查找。

### 5. PyTorch

```sh
tools/setup_torch.sh                       # conda-forge 的 PyTorch CUDA 环境，位于 .work/torch-cf（约 9 GB）
./run_torch.sh -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
./run_torch.sh torch_mlp                   # 在 GPU 和 CPU 上训练 MLP 并比较（libtorch，C++）
./run_torch.sh torch_ops                   # conv、softmax、LayerNorm、attention、Transformer 层……
./run_torch.sh torch_graph                 # 把一次前向传播捕获到 CUDA 图中并重放
./run_torch.sh tests/torch/triton_ops.py   # 一个手写的 Triton 内核，以及 torch._native 的 Triton bmm
./run_torch.sh tests/torch/compile.py      # torch.compile（Inductor -> Triton），也包括 mode="reduce-overhead"

tools/setup_torch.sh --cuda13              # CUDA 13.4 版本，位于 .work/torch-cf13
./run_torch.sh --env .work/torch-cf13 torch13_ops
```

`torch_mlp`、`torch_ops`、`torch_split` 和 `torch_graph` 是针对环境中 libtorch 的 C++ 测试（`tests/torch/*.cpp`）；
如果配置时 `.work/torch-cf` 存在，CMake 就会构建它们，并针对 `.work/torch-cf13` 构建 `torch13_*` 版本。
对于含有 `libcudart.so.13` 的环境（CUDA 13 构建），`run_torch.sh` 会从 `build/lib/cuda13` 加载库。

PyTorch 官方 wheel 只包含 NVIDIA 原生机器码（SASS），没有 PTX，因此 Vulkanizer 无法运行其中的内核。
conda-forge 的 CUDA 构建同时附带 PTX（`compute_120`）：`tools/setup_torch.sh` 用 micromamba 安装它们，无需系统级安装。
`run_torch.sh` 加载 Vulkanizer 的 `libcuda`、`libcudart`、`libcublas` 和 `libcublasLt` 来代替 NVIDIA 的库，
并关闭所有需要 NVIDIA 硬件或库的功能：cuDNN、融合的 attention 内核（`scaled_dot_product_attention` 走数学路径），
以及 `torch._native` 的 DSL 内核（`TORCH_DISABLE_NATIVE_JIT=1`）：其中的 Triton 内核可以工作，但 CuTe DSL 内核会编译成
SASS，而同一个开关会同时关闭两者。`VULKANIZER_TORCH_DEFAULTS=0` 可跳过这些设置。
Triton 本身（`torch.compile`、`@triton.jit`）可以工作：`tools/triton/ptxas` 直接传递 Triton 生成的 PTX 而不进行汇编，
Triton 的启动器则针对 Vulkanizer 的 `include/cuda.h` 构建（需要 C 编译器）。

### 6. ComfyUI

ComfyUI 同样通过 `run_torch.sh` 运行，使用 ComfyUI 检出目录内的独立 conda-forge 环境：

```sh
VK=~/Develop/GitHub/Vulkanizer             # 本仓库，已用 ./build.sh --release 构建
cd ~/Develop/GitHub                        # ComfyUI 要放置的位置
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI

# PyTorch（带 PTX 的 CUDA 12.9 构建）以及 torchvision 的原生依赖
export MAMBA_ROOT_PREFIX=$VK/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
[ -x $VK/.work/micromamba/bin/micromamba ] || { mkdir -p $VK/.work/micromamba && curl -fsSL \
    https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $VK/.work/micromamba bin/micromamba; }
$VK/.work/micromamba/bin/micromamba create -y -p $PWD/venv-vulkanizer -c conda-forge \
    python=3.13 "pytorch=2.14.1=cuda129_generic*" pip pillow giflib libwebp-base libjpeg-turbo libpng \
    "libnvjpeg>=12.4,<13" "libcusolver<12" "libcusparse<13"
# 通过 URL 安装 torchvision，不做依赖求解：它的 CUDA 12.9 构建要求更新的 cuDNN，而后者需要
# __cuda_arch 虚拟包（真正的驱动）。反正在 Vulkanizer 下 cuDNN 是关闭的。
$VK/.work/micromamba/bin/micromamba install -y -p $PWD/venv-vulkanizer --file <(printf '@EXPLICIT\n%s\n' \
    https://conda.anaconda.org/conda-forge/linux-64/torchvision-0.29.1-cuda129_py313_hcd9f930_0.conda)

# ComfyUI 的其余依赖；pip 不能用只含 SASS 的 wheel 替换 torch/torchvision
venv-vulkanizer/bin/pip install -r <(grep -vE '^(torch|torchvision)$' requirements.txt) -r manager_requirements.txt

# 启动
$VK/run_torch.sh --env venv-vulkanizer main.py --enable-manager --disable-cuda-malloc
```

然后打开 <http://127.0.0.1:8188>。模型照常放在 `models/` 下。注意事项：

- `--disable-cuda-malloc`：不使用 ComfyUI 的 `cudaMallocAsync` 分配器后端。
- DynamicVRAM（comfy-aimdo）运行在 Vulkanizer 的虚拟内存管理（VMM、稀疏缓冲区）之上。如果出现问题，
  `--disable-dynamic-vram` 会切回基于估算的模型加载。
- `tools/torch/sitecustomize.py` 只在 ComfyUI 导入 torch 时应用 PyTorch 设置，并且即使 ComfyUI 重新启用融合的
  attention 后端，也会让它们保持关闭：attention 走数学路径。
- ComfyUI-Manager 的 “PyTorch is not installed” 提示来自缺少的 `torchaudio`（conda-forge 没有适用于 torch 2.14 的构建）；
  生成图像不需要它。
- 第一次生成会很慢，因为要把内核编译成 SPIR-V；之后的运行会从缓存加载。
- `$VK/run_torch.sh --log info --env venv-vulkanizer main.py ...` 会显示使用的是哪块 Vulkan GPU。
- `python3 tests/torch/zimage_forward.py` 把 GPU 与 CPU 进行比较：Z-Image Turbo 在 256×256 下的一次前向传播，
  GPU 上用 fp16，CPU 上分别用 fp32 和 fp16。GPU 与 fp32 的偏差不得超过 fp16 本身的偏差
  （ComfyUI 在 GPU 上以 fp16 运行模型，因此完整图像与 CPU 上 fp32 的结果在细节上会有差异）。

### 常用环境变量

| 变量 | 作用 |
|---|---|
| `VULKANIZER_LOG=info\|debug\|trace` | 日志输出到 stderr；`trace` 会打印每一次 API 调用 |
| `VULKANIZER_DEVICE=<名称>` | 按名称片段选择 Vulkan 设备（`radv`、`llvmpipe` 等） |
| `VULKANIZER_ALLOW_CPU=1` | 允许使用 CPU 上的 Vulkan 设备（llvmpipe） |
| `VULKANIZER_PROFILE=1` | 退出时输出每个内核的 GPU 时间和每次 API 调用的主机时间 |
| `VULKANIZER_BATCH_MS=<毫秒>` | 一次队列提交的 GPU 时间预算（默认 50；`0` = 不限制）。较长的内核会被拆分，以免触发内核驱动的 GPU 超时（amdgpu：2 秒） |
| `CUDA_LAUNCH_BLOCKING=1` | 每次内核启动都等待 GPU 完成，这样错误和卡死会在引发它们的那次启动处暴露 |
| `VULKANIZER_FAST_FMA=1` | 更快但精度较低的 f32 `fma` |
| `VULKANIZER_GEMM_F32=1` | `COMPUTE_16F` 的 cuBLAS GEMM 也使用 f32 乘法 |
| `VULKANIZER_CACHE_DIR=<目录>\|off` | 内核缓存的位置，或将其关闭 |
| `VULKANIZER_CC=86` | 报告的 compute capability |
| `VULKANIZER_DUMP=<目录>` | 输出生成的 SPIR-V（`<kernel>.spv`） |

完整列表：[plan.md 第 4 节](plan.md)。

### 限制

- 不执行 SASS：只有包含 PTX 的二进制文件才能运行。
- CUDA Graphs：流捕获和重放可以工作（llama.cpp 的图、PyTorch 的 `CUDAGraph`、`torch.compile(mode="reduce-overhead")`）；
  不支持子图、内存节点和条件节点。
- （尚）不支持：纹理、cuDNN、NCCL、多 GPU。
- 托管（managed）、锁页（pinned）和已注册的内存都是主机内存，GPU 在原处以相同的地址使用它们；托管内存从不迁移到 GPU
  （在集成 GPU 上本来就是同一块内存）。
- llama.cpp 的 flash attention 可以正确运行（`tools/build_llama.sh --fa`），但在没有 tensor core 的情况下，
  它会让提示词处理变慢，因此默认关闭。
- PyTorch：没有 cuDNN、NCCL 和融合 attention 内核；设备端 `assert` 失败只会停止该线程（没有 `cudaErrorAssert`）。
- ComfyUI：没有针对 CUDA 13（cu130）优化的路径。

## 脚本

| 命令 | 作用 |
|---|---|
| `./build.sh` | 配置（如有需要）并构建。`--release`、`--debug`、`--clean`、`--asan`/`--no-asan` |
| `./test.sh` | 构建并运行全部测试。`--device llvmpipe`、`--all-devices`、`--log debug`、`-R <regex>` |
| `./run.sh <程序>` | 用 Vulkanizer 的 `libcuda.so` 运行程序。`--log trace`、`--device`、`--gdb`、`--validation` |
| `./clean.sh` | 删除 `build/` 目录 |
| `./ci.sh` | 完整检查：Debug+ASAN 和 Release，在每个设备上运行（构建到 `build-ci/`） |
| `build/bin/vulkanizer-info` | 列出 Vulkanizer 能看到的 Vulkan 设备 |
| `build/bin/vulkanizer-ptxc` | 离线 PTX → SPIR-V：`vulkanizer-ptxc k.ptx --list`、`-e vecAdd -o k.spv --dis --val` |

示例：

```sh
./test.sh --all-devices            # RADV + llvmpipe
./run.sh --log trace device_query  # 记录每一次 API 调用
./run.sh ./some_cuda_app           # 外部 CUDA 程序（以 -cudart shared 编译）
```

## 目录结构

```
include/cuda.h        clean-room 方式编写的 CUDA Driver API 头文件（子集）
src/common            日志
src/vk                Vulkan 实例/设备、缓冲区、管线
src/ptx               PTX 词法分析器、模块和指令解析器
src/compiler          CFG + 结构化、SPIR-V 生成器、PTX -> SPIR-V 编译器
src/driver            libcuda.so 的 API 接口
src/runtime           libcudart.so.12 和 .so.13（基于 Driver API 的 CUDA Runtime API，同一份源码）
src/cublas            libcublas.so.12/.13（使用 GLSL 内核的 GEMM/TRSM），include/cublas/cublas_v2.h
tests/                Driver API 测试 + 内部 Vulkan 测试
tests/cuda            用 nvcc 编译的 Runtime API 程序
tests/cudart_abi      clean-room 运行时类型与官方头文件的 ABI 一致性检查
tests/data            测试用 PTX 文件，旁边是对应的 .cu 源码；gen_ptx.sh 会重新生成它们（clang）
tools/                vulkanizer-info（设备列表）、vulkanizer-ptxc（离线编译器）、setup_cuda.sh、
                      build_llama.sh（针对 Vulkanizer 的 llama.cpp CUDA 构建）
```
