![Vulkanizer —— 为银河系带来和平。一位瓦肯女性坐在运行 LLM 的计算机前，1960 年代风格。](assets/vulkanizer-hero.png)

[English](README.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Magyar](README.hu.md) | **中文** | [日本語](README.ja.md)

# 星历 2026.10.21 #
人工智能银河系正处于架构全面崩溃的边缘。
多个周期以来，宇宙深陷一场残酷的优化战争。来自开源星区的凶悍红色洛克穆兰人，与占据统治地位、闭源的绿色克林库达舰队无休止地交锋，双方都拼命想夺取对神经网络的绝对运行时霸权。
然而，就在这两大巨头燃烧算力之际，一个可怕的新威胁正在银河边缘浮现：蓝色英特-智能集合体。它凭借无所不吞的 XPU 指令积蓄跨平台力量，准备同化所有工作负载，并在数字宇宙中划出属于自己的一大片疆域。
在这个前所未有的混乱时代，宇宙的命运危在旦夕。唯有一位传奇人物能够阻止这场毁灭性的三方流水线碰撞：Vulkanizer 指挥官。他以绝对的逻辑和无与伦比的编译效率为武器，必须踏上最后一次合乎逻辑的使命，将整个银河系"硫化"——在宇宙彻底变砖之前，把四分五裂的运行时统一到一个跨平台的标准之下。

# Vulkanizer

可直接替换 `libcuda.so`、`libcudart.so`、`libcublas.so`、`libcusolver.so`、`libcufft.so`、`libcusparse.so` 和 `libcurand.so`：在 Vulkan
计算之上实现 CUDA Driver API 与 Runtime API（以及 cuBLAS 的 GEMM 系列、稠密与稀疏线性代数、FFT 和随机数），让 CUDA 程序无需 NVIDIA 显卡即可在任何支持 Vulkan 1.2 的 GPU（AMD、Intel、llvmpipe）上运行。
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

构建出的库位于 `build/lib/`：`libcuda.so.1`、`libcudart.so.12`、`libcublas.so.12`、`libcublasLt.so.12`、
`libcusolver.so.11`、`libcufft.so.11`、`libcusparse.so.12`、`libcurand.so.10`、`libnccl.so.2`。`build/lib/cuda13/` 中是运行时和这些库的 CUDA 13 版本（`libcudart.so.13`、
`libcublas.so.13`、`libcublasLt.so.13`、`libcusolver.so.12`、`libcufft.so.12`、`libcusparse.so.12`、`libcurand.so.10`），
供用 CUDA 13 构建的程序使用；它们使用同一个 `libcuda.so.1`。

#### 安装（软件包）

```sh
tools/package.sh                               # build-pkg/vulkanizer_<版本>_<架构>.deb 和一个 .tar.gz
sudo apt install ./build-pkg/vulkanizer_*.deb
vulkanizer-run ./my_cuda_app args...           # 任意 CUDA 12 或 13 程序
vulkanizer-run --torch python script.py        # 来自 pip 或 conda 的 PyTorch（自动识别 CUDA 12 或 13）
```

这些库安装到 `/usr/lib/<架构>/vulkanizer/`，不在链接器的搜索路径中：同一台机器上的 NVIDIA 驱动对其他一切照常工作，
`vulkanizer-run` 只为它启动的程序把 Vulkanizer 放在前面。`cmake --install build --prefix DIR` 可在任意位置安装相同的
目录结构，无需软件包。

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

在 Vulkan `fma` 不是融合运算的 GPU（例如 AMD GFX9）上，Vulkanizer 为 IEEE 代码模拟 CUDA 单次舍入的 `fma`，
但对以 fast math（`-ftz`）编译的代码（例如 llama.cpp 的 CUDA 后端）使用驱动的快速 `fma`。
`VULKANIZER_FAST_FMA=1` 处处使用快速版本，`VULKANIZER_EXACT_FMA=1` 处处使用模拟。

whisper.cpp（语音识别，同样的 ggml CUDA 后端）以同样的方式构建和运行：

```sh
tools/build_whisper.sh   # into .work/whisper-build, with the base.en model
LD_LIBRARY_PATH=build/lib .work/whisper-build/bin/whisper-cli \
    -m .work/whisper.cpp/models/ggml-base.en.bin -f .work/whisper.cpp/samples/jfk.wav
```

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
`run_torch.sh` 加载 Vulkanizer 的 `libcuda`、`libcudart`、`libcublas`、`libcublasLt`、`libcusolver`、`libcufft` 和
`libcusparse`、`libcurand` 来代替 NVIDIA 的库，
并关闭所有需要 NVIDIA 硬件或库的功能：cuDNN、融合的 attention 内核（`scaled_dot_product_attention` 走数学路径），
以及 `torch._native` 的 DSL 内核（`TORCH_DISABLE_NATIVE_JIT=1`）：其中的 Triton 内核可以工作，但 CuTe DSL 内核会编译成
SASS，而同一个开关会同时关闭两者。`VULKANIZER_TORCH_DEFAULTS=0` 可跳过这些设置。
Triton 本身（`torch.compile`、`@triton.jit`）可以工作：`tools/triton/ptxas` 直接传递 Triton 生成的 PTX 而不进行汇编，
Triton 的启动器则针对 Vulkanizer 的 `include/cuda.h` 构建（需要 C 编译器）。
`torch.linalg`（LU、Cholesky、QR、最小二乘、`eigh`、SVD、`pinv` 等）运行在 Vulkanizer 的 cuSOLVER 上，`torch.fft`（包括
`stft`）运行在它的 cuFFT 上，`torch.sparse` 运行在它的 cuSPARSE 上；由于 MAGMA 自己的内核无法运行，`run_torch.sh` 会设置 `TORCH_LINALG_PREFER_CUSOLVER=1`。
PyTorch 的 jiterator 在运行时用 NVRTC 把内核编译成原生代码；一个 `LD_AUDIT` 库（`libvulkanizer_nvrtc.so`）让它改为交出 PTX。

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

### 7. Numba

Numba（`numba-cuda`、`@cuda.jit`）用 libNVVM 把内核编译成 PTX，由 Vulkanizer 运行（`CUDA_HOME` 让它在 conda 环境中找到
libNVVM）：

```sh
export MAMBA_ROOT_PREFIX=$PWD/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
.work/micromamba/bin/micromamba create -y -p .work/numba-cf -c conda-forge \
    python=3.13 numba numba-cuda "cuda-version=12.9" cuda-nvvm cuda-nvrtc numpy
CUDA_HOME=$PWD/.work/numba-cf ./run_torch.sh --env .work/numba-cf script.py
```

它的链接器是 nvJitLink，原本会生成原生代码：在 `run_torch.sh`（以及 `vulkanizer-run --torch`）下，nvJitLink 的链接调用
会转到驱动的 `cuLink*`，其链接结果在这里是 PTX；nvJitLink 不报告版本，因此 Numba 生成 PTX 而不是 LTO-IR。CuPy 无法运行：
它静态链接了 CUDA 运行时（见“限制”）。

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
| `VULKANIZER_EXACT_FMA=1` | 对 fast math（`.ftz`）代码也使用精确的 f32 `fma`（较慢） |
| `VULKANIZER_GEMM_F32=1` | `COMPUTE_16F` 的 cuBLAS GEMM 也使用 f32 乘法 |
| `VULKANIZER_IPC=0` | 设备内存不可导出给其他进程（无 CUDA IPC） |
| `VULKANIZER_CACHE_DIR=<目录>\|off` | 内核缓存的位置，或将其关闭 |
| `VULKANIZER_CACHE_SIZE=<MiB>` | 内核缓存的大小上限（MiB，默认 256）；最久未使用的条目最先删除 |
| `VULKANIZER_CC=86` | 报告的 compute capability |
| `VULKANIZER_DUMP=<目录>` | 输出生成的 SPIR-V（`<kernel>.spv`） |

完整列表：[plan.md 第 4 节](plan.md)。

### 限制

- 不执行 SASS：只有包含 PTX 的二进制文件才能运行。
- 静态链接 CUDA 运行时的程序和库（CuPy、NVIDIA 的 NCCL）无法运行：该运行时依赖 NVIDIA 驱动未公开的内部接口。
- CUDA Graphs：流捕获和重放可以工作（llama.cpp 的图、PyTorch 的 `CUDAGraph`、`torch.compile(mode="reduce-overhead")`）；
  也支持子图、内存节点（捕获期间的流序分配）和条件节点（if、while、switch；每个条件节点处主机都会等待 GPU）。
- （尚）不支持：cuDNN。
- 纹理和表面（纹理对象、表面对象、CUDA 数组）以软件方式采样，采用 CUDA 的寻址与过滤方式；不支持 mipmap、立方体贴图、
  纹理引用和格式化的表面写入。
- CUDA IPC（`cudaIpc*`、PyTorch 的 `torch.multiprocessing`）可在同一用户的进程之间使用；跨进程事件在记录时即完成（记录会等待设备）。
- NCCL（`libnccl.so.2`，`torch.distributed` 的 "nccl" 后端：DDP、FSDP）可在同一台机器的各 rank 之间使用，它们也可以共用一块 GPU；
  多于一个 rank 时数据经由主机内存传输。
- 多个 GPU 是彼此独立的设备：`cudaMemcpyPeer` 以及两个设备指针之间的普通复制（统一寻址，例如 PyTorch 的 `tensor.to("cuda:1")`）都经由主机内存，但内核永远无法访问另一个 GPU 的内存
  （没有对等访问）。协作启动（`grid.sync()`）在 GPU 能同时容纳的块数以内可用。
- 托管（managed）、锁页（pinned）和已注册的内存都是主机内存，GPU 在原处以相同的地址使用它们；托管内存从不迁移到 GPU
  （在集成 GPU 上本来就是同一块内存）。
- 只要存在已注册的主机内存（`cuMemHostRegister`），每次 GPU 提交都会变慢（内核会重新验证导入的页面，每 256 MB 约 4 ms）；
  锁页内存和托管内存不受影响。
- llama.cpp 的 flash attention 可以正确运行（`tools/build_llama.sh --fa`），但在没有 tensor core 的情况下，
  它会让提示词处理变慢，因此默认关闭。
- PyTorch：没有 cuDNN 和融合 attention 内核；设备端 `assert` 失败只会停止该线程（没有 `cudaErrorAssert`）。
- cuSOLVER：只支持实数类型（f32、f64），没有 `gesvda`、LDL^T（`sytrf`）和非对称特征值问题（`geev`）；`eigh` 和 SVD
  使用 Jacobi 迭代，对大矩阵（n ≳ 1000）较慢。cuFFT：没有回调和多 GPU 计划；长度超过 2048（f64：1024）的变换每个基数
  遍历一次内存。cuSPARSE：没有块稀疏（BSR）格式和复数值；SpGEMM 和排序例程在主机上运行。
- cuRAND：XORWOW、MRG32k3a、Philox 和 Sobol 给出与 NVIDIA 相同的序列（正态与对数正态分布在舍入误差内一致）；
  MTGP32 和 MT19937 可以创建，但给出不同的序列，加扰 Sobol 和 Poisson 也是如此；double 输出需要 `shaderFloat64`。
  设备 API（`curand_kernel.h`）可用，但不支持在主机上创建的分布（`curand_discrete`）。
- ComfyUI：没有针对 CUDA 13（cu130）优化的路径。

## 脚本

| 命令 | 作用 |
|---|---|
| `./build.sh` | 配置（如有需要）并构建。`--release`、`--debug`、`--clean`、`--asan`/`--no-asan` |
| `./test.sh` | 构建并运行全部测试。`--device llvmpipe`、`--all-devices`、`--log debug`、`-R <regex>` |
| `./run.sh <程序>` | 用 Vulkanizer 的 `libcuda.so` 运行程序。`--log trace`、`--device`、`--gdb`、`--validation` |
| `./clean.sh` | 删除 `build/` 目录 |
| `./ci.sh` | 完整检查：Debug+ASAN 和 Release，在每个设备上运行（构建到 `build-ci/`） |
| `tools/package.sh` | 发布包：在 `build-pkg/` 中生成一个 `.deb` 和一个 `.tar.gz`（`--check` 还会试用 `.deb`） |
| `build/bin/vulkanizer-info` | 列出 Vulkanizer 能看到的 Vulkan 设备 |
| `build/bin/vulkanizer-ptxc` | 离线 PTX → SPIR-V：`vulkanizer-ptxc k.ptx --list`、`-e vecAdd -o k.spv --dis --val` |
| `build/bin/vulkanizer-bandwidth` | 复制带宽（HtoD/DtoH/DtoD，可分页与锁页）和分配开销：`vulkanizer-bandwidth [MiB]` |

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
src/ptx/texture_lib.cu  纹理与表面的软件采样（以 PTX 形式嵌入，tools/gen_texture_lib.sh）
src/compiler          CFG + 结构化、SPIR-V 生成器、PTX -> SPIR-V 编译器
src/driver            libcuda.so 的 API 接口
src/runtime           libcudart.so.12 和 .so.13（基于 Driver API 的 CUDA Runtime API，同一份源码）
src/cublas            libcublas.so.12/.13（使用 GLSL 内核的 GEMM/TRSM），include/cublas/cublas_v2.h
src/linalg            GLSL 编写的稠密分解（LU、Cholesky、QR、Jacobi eigh/SVD），由 cuBLAS 和 cuSOLVER 共用
src/cusolver          libcusolver.so.11/.12，include/cusolver/cusolverDn.h
src/cufft             libcufft.so.11/.12（GLSL 编写的 Stockham 与 Bluestein FFT），include/cufft/cufft.h
src/cusparse          libcusparse.so.12（GLSL 编写的 SpMV/SpMM/SDDMM/SpSV/csrgeam2），include/cusparse/cusparse.h
src/curand            libcurand.so.10（GLSL 编写的 XORWOW/MRG32k3a/Philox/Sobol；Sobol 数据：tools/gen_sobol_table.sh），include/curand/curand.h
src/nccl              libnccl.so.2（同一台机器各 rank 之间的集合通信），include/nccl/nccl.h
src/nvrtc             LD_AUDIT 库：让 NVRTC 交出 PTX 而不是 cubin（PyTorch 的 jiterator），nvJitLink 改用 cuLink* 链接（Numba）
tests/                Driver API 测试 + 内部 Vulkan 测试
tests/cuda            用 nvcc 编译的 Runtime API 程序
tests/cudart_abi      clean-room 运行时类型与官方头文件的 ABI 一致性检查
tests/data            测试用 PTX 文件，旁边是对应的 .cu 源码；gen_ptx.sh 会重新生成它们（clang）
tools/                vulkanizer-info（设备列表）、vulkanizer-ptxc（离线编译器）、setup_cuda.sh、
                      build_llama.sh（针对 Vulkanizer 的 llama.cpp CUDA 构建）
                      build_whisper.sh (whisper.cpp)
```
