![Vulkanizer – békét hoz a galaxisba. Egy vulkáni nő egy LLM-et futtató számítógép előtt, 1960-as évekbeli stílusban.](assets/vulkanizer-hero.png)

[English](README.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | **Magyar** | [中文](README.zh.md) | [日本語](README.ja.md)

# Csillagidő 2026.10.21 #
A Mesterséges Intelligencia Galaxisa a teljes architekturális összeomlás szélén áll.
Ciklusok óta kegyetlen optimalizációs háború tartja fogva a kozmoszt. A nyílt forráskódú szektorokból előtörve a vad Vörös Rocmuliak szüntelenül csapnak össze a Zöld Klincudák uralkodó, zárt flottáival – mindkét fél kétségbeesetten küzd a neurális hálók feletti abszolút futásidejű fölényért.
Ám miközben a két titán égeti a számítási kapacitást, a galaxis peremén új, rettenetes fenyegetés tűnik fel: a Kék Intel-ligens Kollektíva. Mindenevő XPU-direktíváival platformfüggetlen erőt gyűjt, hogy minden munkaterhet asszimiláljon, és kihasítsa a maga hatalmas szeletét a digitális univerzumból.
A példátlan káosz e korszakában a kozmosz sorsa hajszálon függ. A pusztító háromirányú pipeline-ütközést csak egyetlen legendás alak háríthatja el: Vulkanizer parancsnok. Abszolút logikával és páratlan fordítási hatékonysággal felvértezve egy utolsó, logikus küldetésre kell indulnia, hogy vulkanizálja az egész galaxist – egyetlen, platformfüggetlen szabvány alá egyesítse a széttöredezett futtatókörnyezeteket, mielőtt az univerzum végleg téglává válik.

# Vulkanizer

A `libcuda.so`, `libcudart.so` és `libcublas.so` közvetlen helyettesítője: a CUDA Driver és Runtime API-t (és a
cuBLAS GEMM-családját) Vulkan compute-ra építve valósítja meg, így a CUDA-programok NVIDIA-kártya nélkül futnak, bármely
Vulkan 1.2-es GPU-n (AMD, Intel, llvmpipe).
Terv, architektúra és fázisok: [plan.md](plan.md).

## Követelmények

CMake ≥ 3.20, C++20-as fordító, Vulkan 1.2+ driver (bufferDeviceAddress + shaderInt64), a tesztekhez `glslc`.

Opcionális (Ubuntu csomagnevek):

| Csomag | Mire kell |
|---|---|
| `spirv-tools` | `spirv-val` / `spirv-dis` – a generált SPIR-V ellenőrzése (`vulkanizer-ptxc --val --dis`) |
| `vulkan-validationlayers` | `./run.sh --validation` |
| `clang` | Teszt-PTX előállítása CUDA SDK nélkül (`tests/data/gen_ptx.sh`) |
| `ninja-build` | Gyorsabb fordítás (a `build.sh` magától használja, ha megvan) |
| `g++-13` | Host-fordító az nvcc-hez (a CUDA 12.9 nem fogadja el a gcc 15-öt) |

Valódi nvcc-vel fordított tesztek: a `tools/setup_cuda.sh` letölti az NVIDIA nvcc-jét és runtime-ját
(újraterjeszthető csomagokból) a `.cuda/` könyvtárba (driver, GPU és root nem kell hozzá). Ha a `.cuda/` létezik, a
CMake nvcc-vel lefordítja a `tests/cuda/*.cu` programokat, a ctest pedig a Vulkanizer könyvtáraival futtatja őket.
A `tools/setup_cuda.sh --cuda13` ugyanezt teszi a CUDA 13.4-gyel a `.cuda13/`-ba: a programok nvcc 13-mal újra
fordulnak, és a `libcudart.so.13`-on futnak (a `_13` végű tesztek).

## Használat

### 1. Fordítás

```sh
./build.sh --release                       # Release fordítás a build/ könyvtárba
VULKANIZER_BUILD_DIR=$PWD/build-rel ./build.sh --release   # vagy egy külön könyvtárba
build/bin/vulkanizer-info                       # milyen Vulkan eszközöket lát a Vulkanizer
./test.sh                                  # tesztek (opcionális)
```

A lefordított könyvtárak a `build/lib/`-ben vannak: `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`,
`libcublasLt.so.12`. A `build/lib/cuda13/` a runtime és a cuBLAS CUDA 13-as változatait tartalmazza
(`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`) a CUDA 13-mal fordított programokhoz; ezek ugyanazt a
`libcuda.so.1`-et használják.

#### Telepítés (csomag)

```sh
tools/package.sh                               # build-pkg/vulkanizer_<verzió>_<arch>.deb és egy .tar.gz
sudo apt install ./build-pkg/vulkanizer_*.deb
vulkanizer-run ./my_cuda_app args...           # bármely CUDA 12-es vagy 13-as program
vulkanizer-run --torch python script.py        # PyTorch pipből vagy condából (CUDA 12 vagy 13, felismeri)
```

A könyvtárak a `/usr/lib/<arch>/vulkanizer/` alá kerülnek, a linker keresési útvonalán kívül: egy ugyanazon a gépen
lévő NVIDIA driver minden másnak megmarad, a `vulkanizer-run` csak az általa indított programnak teszi előre a
Vulkanizert. A `cmake --install build --prefix DIR` ugyanezt a szerkezetet bárhová, csomag nélkül telepíti.

### 2. CUDA-program futtatása

A Vulkanizer a valódi NVIDIA-könyvtárak helyére lép; a dinamikus linkernek csak előbb kell megtalálnia őket:

```sh
./run.sh ./my_cuda_app args...                       # a run.sh beállítja az LD_LIBRARY_PATH-t
LD_LIBRARY_PATH=$PWD/build/lib ./my_cuda_app args... # ugyanez kézzel
./run.sh --log info ./my_cuda_app                    # kiírja, melyik GPU-n fut
./run.sh --device llvmpipe ./my_cuda_app             # adott eszközön (CPU-s eszközhöz VULKANIZER_ALLOW_CPU=1 kell)
```

A programnak két feltételt kell teljesítenie:

- **A binárisban PTX-nek kell lennie.** A Vulkanizer a PTX-et fordítja SPIR-V-re; gépi kódot (SASS) nem tud
  használni. Fordításkor virtuális architektúrát kell megadni, pl.
  `nvcc -arch=compute_61 ...` vagy CMake-ben `CMAKE_CUDA_ARCHITECTURES=61-virtual`.
  A 61 (Pascal) jó választás: van benne `dp4a`, és nincsenek tensor core-os kódutak.
- **A CUDA runtime-ot dinamikusan kell linkelni** (`nvcc -cudart shared`, CMake-ben
  `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`), különben az NVIDIA statikus libcudart-ja kerül a programba.

Saját program fordítása a letöltött nvcc-vel (GPU és NVIDIA-driver nem kell):

```sh
tools/setup_cuda.sh                                   # nvcc + runtime headerek a .cuda/ könyvtárba
.cuda/bin/nvcc -ccbin g++-13 -arch=compute_61 -cudart shared vadd.cu -o vadd
./run.sh ./vadd
```

cuBLAS-t használó programokhoz a Vulkanizer `include/cublas/cublas_v2.h` headerét és a `build/lib/libcublas.so`-t kell
használni (`-Iinclude/cublas -Lbuild/lib -lcublas`).

Az első futás lassabb: ekkor fordulnak le a kernelek SPIR-V-re, és kerülnek a lemezes gyorsítótárba
(`~/.cache/vulkanizer`). A későbbi indítások onnan töltik be őket.

### 3. llama.cpp

```sh
tools/setup_cuda.sh
./build.sh --release
tools/build_llama.sh                  # klónozza és lefordítja a .work/llama.cpp-t a CUDA backenddel
./run.sh .work/llama-build/bin/llama-simple -m model.gguf -ngl 99 "Hello"
./run.sh .work/llama-build/bin/llama-bench -m model.gguf -ngl 99
```

A `VULKANIZER_FAST_FMA=1` gyorsítja a tokengenerálást azokon a GPU-kon, ahol a Vulkan `fma` nem fúzionált (pl. AMD
GFX9): ott a Vulkanizer alapból pontos fma-t emulál, ez a kapcsoló helyette a driver gyorsabb, de pontatlanabb
változatát használja. LLM-eknél a különbség elhanyagolható.

### 4. bitnet.c

A bitnet.c CUDA backendje (`--cuda`) a `run_bitnet.sh`-val fut Vulkanon:

```sh
./run_bitnet.sh models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The capital of France is" -n 64
./run_bitnet.sh --profile --log info models/model.gguf -p "Hello"   # GPU-idők, naplózás
./run_bitnet.sh --build-only                                        # csak fordítás
```

A szkript nem módosítja a checkoutot (`--bitnet-dir`, alapból `~/Develop/GitHub/bitnet.c`): a forrásokat a
`.work/bitnet-vulkanizer`-be szinkronizálja, és ott fordítja PTX-re (`compute_61`) a Vulkanizer cuBLAS-a és runtime-ja
ellen; a `--cuda` kapcsolót magától hozzáadja. A relatív modellútvonalakat a checkoutban is keresi.

### 5. PyTorch

```sh
tools/setup_torch.sh                       # conda-forge PyTorch CUDA környezet a .work/torch-cf-be (~9 GB)
./run_torch.sh -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
./run_torch.sh torch_mlp                   # MLP-t tanít GPU-n és CPU-n, és összeveti őket (libtorch, C++)
./run_torch.sh torch_ops                   # conv, softmax, LayerNorm, attention, Transformer-réteg, ...
./run_torch.sh torch_graph                 # egy forward pass CUDA gráfba rögzítve és visszajátszva
./run_torch.sh tests/torch/triton_ops.py   # egy kézzel írt Triton kernel és a torch._native Triton-os bmm-je
./run_torch.sh tests/torch/compile.py      # torch.compile (Inductor -> Triton), mode="reduce-overhead"-del is

tools/setup_torch.sh --cuda13              # a CUDA 13.4-es build a .work/torch-cf13-ba
./run_torch.sh --env .work/torch-cf13 torch13_ops
```

A `torch_mlp`, `torch_ops`, `torch_split` és `torch_graph` C++ tesztek a környezet libtorch-ja ellen
(`tests/torch/*.cpp`); a CMake akkor fordítja őket, ha a `.work/torch-cf` konfiguráláskor létezik, illetve `torch13_*`
néven a `.work/torch-cf13` ellen. Olyan környezetben, ahol `libcudart.so.13` van (CUDA 13-as build), a `run_torch.sh` a
`build/lib/cuda13`-ból tölti a könyvtárakat.

A hivatalos PyTorch wheelekben csak natív NVIDIA gépi kód (SASS) van, PTX nincs, így a Vulkanizer nem tudja futtatni a
kerneleiket. A conda-forge CUDA-buildjei PTX-et is tartalmaznak (`compute_120`): a `tools/setup_torch.sh` ezeket
telepíti micromambával, rendszerszintű telepítés nélkül. A `run_torch.sh` az NVIDIA-é helyett a Vulkanizer `libcuda`,
`libcudart`, `libcublas` és `libcublasLt` könyvtárait tölti be, és kikapcsol mindent, ami NVIDIA hardvert vagy
könyvtárat igényel: a cuDNN-t, a fúzionált attention kerneleket (a `scaled_dot_product_attention` a math úton fut) és a
`torch._native` DSL-kerneleit (`TORCH_DISABLE_NATIVE_JIT=1`): a Triton-kernelei működnek, a CuTe DSL kernelei viszont
SASS-ra fordulnak, és ugyanaz az egy kapcsoló mindkettőt tiltja. A `VULKANIZER_TORCH_DEFAULTS=0` mindezt kihagyja.
Maga a Triton (`torch.compile`, `@triton.jit`) működik: a `tools/triton/ptxas` továbbadja a Triton PTX-ét ahelyett,
hogy lefordítaná, a Triton launchere pedig a Vulkanizer `include/cuda.h`-ja ellen fordul (C-fordító kell hozzá).

### 6. ComfyUI

A ComfyUI is a `run_torch.sh`-val fut, saját conda-forge környezetben a ComfyUI checkoutján belül:

```sh
VK=~/Develop/GitHub/Vulkanizer             # ez a repó, ./build.sh --release-szel lefordítva
cd ~/Develop/GitHub                        # ahová a ComfyUI kerüljön
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI

# PyTorch (CUDA 12.9-es build PTX-szel) és a torchvision natív függőségei
export MAMBA_ROOT_PREFIX=$VK/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
[ -x $VK/.work/micromamba/bin/micromamba ] || { mkdir -p $VK/.work/micromamba && curl -fsSL \
    https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $VK/.work/micromamba bin/micromamba; }
$VK/.work/micromamba/bin/micromamba create -y -p $PWD/venv-vulkanizer -c conda-forge \
    python=3.13 "pytorch=2.14.1=cuda129_generic*" pip pillow giflib libwebp-base libjpeg-turbo libpng \
    "libnvjpeg>=12.4,<13" "libcusolver<12" "libcusparse<13"
# a torchvision URL-ről, feloldás nélkül: a CUDA 12.9-es buildje újabb cuDNN-t kér, ami a __cuda_arch virtuális
# csomagot (valódi drivert) igényli. A Vulkanizer alatt a cuDNN amúgy is ki van kapcsolva.
$VK/.work/micromamba/bin/micromamba install -y -p $PWD/venv-vulkanizer --file <(printf '@EXPLICIT\n%s\n' \
    https://conda.anaconda.org/conda-forge/linux-64/torchvision-0.29.1-cuda129_py313_hcd9f930_0.conda)

# a ComfyUI többi függősége; a pip nem cserélheti le a torch/torchvision csomagot a csak SASS-t tartalmazó wheelekre
venv-vulkanizer/bin/pip install -r <(grep -vE '^(torch|torchvision)$' requirements.txt) -r manager_requirements.txt

# indítás
$VK/run_torch.sh --env venv-vulkanizer main.py --enable-manager --disable-cuda-malloc
```

Ezután nyisd meg a <http://127.0.0.1:8188> címet. A modellek a szokott módon a `models/` alá kerülnek. Megjegyzések:

- `--disable-cuda-malloc`: a ComfyUI `cudaMallocAsync` allokátor-backendje nincs használatban.
- A DynamicVRAM (comfy-aimdo) a Vulkanizer virtuális memóriakezelésén fut (VMM, sparse bufferek). Ha gondot okoz, a
  `--disable-dynamic-vram` visszavált a becslés alapú modellbetöltésre.
- A `tools/torch/sitecustomize.py` csak akkor alkalmazza a PyTorch-beállításokat, amikor a ComfyUI importálja a
  torch-ot, és a fúzionált attention backendeket akkor is kikapcsolva tartja, ha a ComfyUI visszakapcsolja őket: az
  attention a math úton fut.
- A ComfyUI-Manager „PyTorch is not installed” üzenete a hiányzó `torchaudio` miatt jön (a conda-forge-on nincs build
  a torch 2.14-hez); a képgeneráláshoz nem kell.
- Az első generálás lassú, amíg a kernelek SPIR-V-re fordulnak; a későbbi futások a gyorsítótárból töltik őket.
- A `$VK/run_torch.sh --log info --env venv-vulkanizer main.py ...` kiírja, melyik Vulkan GPU-t használja.
- A `python3 tests/torch/zimage_forward.py` a GPU-t a CPU-val veti össze: egy Z-Image Turbo forward 256×256-on, fp16-ban
  a GPU-n, fp32-ben és fp16-ban a CPU-n. A GPU legfeljebb annyira térhet el az fp32-től, mint maga az fp16 (a ComfyUI
  fp16-ban futtatja a modellt a GPU-n, ezért a teljes képek a finom részletekben eltérnek egy CPU-s fp32 futástól).

### Hasznos környezeti változók

| Változó | Hatás |
|---|---|
| `VULKANIZER_LOG=info\|debug\|trace` | Naplózás a stderr-re; a `trace` minden API-hívást kiír |
| `VULKANIZER_DEVICE=<név>` | Vulkan eszköz kiválasztása névrészlet alapján (`radv`, `llvmpipe`, ...) |
| `VULKANIZER_ALLOW_CPU=1` | CPU-s Vulkan eszközök (llvmpipe) engedélyezése |
| `VULKANIZER_PROFILE=1` | Kilépéskor kernelenkénti GPU-idő és API-hívásonkénti host-idő |
| `VULKANIZER_BATCH_MS=<ms>` | Egy queue-beküldés GPU-időkerete (alapból 50; `0` = nincs korlát). A hosszabb kernelek darabokra bomlanak, így nem éri el őket a kernel driver GPU-timeoutja (amdgpu: 2 s) |
| `CUDA_LAUNCH_BLOCKING=1` | Minden kernelindítás megvárja a GPU-t, így a hibák és lefagyások annál az indításnál jelentkeznek, amelyik okozta őket |
| `VULKANIZER_FAST_FMA=1` | Gyorsabb, kevésbé pontos f32 `fma` |
| `VULKANIZER_GEMM_F32=1` | A `COMPUTE_16F`-es cuBLAS GEMM-ek is f32 szorzással futnak |
| `VULKANIZER_CACHE_DIR=<dir>\|off` | A kernel-gyorsítótár helye, vagy kikapcsolása |
| `VULKANIZER_CC=86` | A jelentett compute capability |
| `VULKANIZER_DUMP=<dir>` | A generált SPIR-V kiírása (`<kernel>.spv`) |

Teljes lista: [plan.md, 4. szakasz](plan.md).

### Korlátok

- Nincs SASS-futtatás: csak PTX-et tartalmazó binárisok futnak.
- CUDA Graphs: a stream capture és a visszajátszás működik (a llama.cpp gráfjai, a PyTorch `CUDAGraph`,
  `torch.compile(mode="reduce-overhead")`); child graph, memória- és feltételes csomópont nincs.
- (Még) nem támogatott: textúrák, cuDNN, NCCL, több GPU.
- A managed, a pinned és a regisztrált memória host-memória, amit a GPU helyben, ugyanazon a címen használ; a managed
  memória sosem migrál a GPU-ra (integrált GPU-n ez amúgy is ugyanaz a memória).
- Amíg regisztrált host-memória (`cuMemHostRegister`) létezik, minden GPU-submit lassabb (a kernel újraellenőrzi az
  importált lapokat, ~4 ms / 256 MB); a pinned és a managed memóriát ez nem érinti.
- A llama.cpp flash attention helyesen fut (`tools/build_llama.sh --fa`), de tensor core-ok nélkül vele lassabb a
  prompt feldolgozása, ezért alapból ki van kapcsolva.
- PyTorch: nincsenek cuDNN-, NCCL- és fúzionált attention kernelek; egy eszközoldali `assert` hibája csak a szálat
  állítja meg (nincs `cudaErrorAssert`).
- ComfyUI: nincsenek a CUDA 13-ra (cu130) optimalizált utak.

## Szkriptek

| Parancs | Mit csinál |
|---|---|
| `./build.sh` | Konfigurál (ha kell) és fordít. `--release`, `--debug`, `--clean`, `--asan`/`--no-asan` |
| `./test.sh` | Fordítás + minden teszt. `--device llvmpipe`, `--all-devices`, `--log debug`, `-R <regex>` |
| `./run.sh <program>` | Programot futtat a Vulkanizer `libcuda.so`-jával. `--log trace`, `--device`, `--gdb`, `--validation` |
| `./clean.sh` | Törli a `build/` könyvtárat |
| `./ci.sh` | Teljes ellenőrzés: Debug+ASAN és Release, minden eszközön (a `build-ci/`-be fordít) |
| `tools/package.sh` | Kiadási csomagok: egy `.deb` és egy `.tar.gz` a `build-pkg/`-ben (`--check`: a `.deb`-et ki is próbálja) |
| `build/bin/vulkanizer-info` | Kilistázza a Vulkanizer által látott Vulkan eszközöket |
| `build/bin/vulkanizer-ptxc` | Offline PTX → SPIR-V: `vulkanizer-ptxc k.ptx --list`, `-e vecAdd -o k.spv --dis --val` |
| `build/bin/vulkanizer-bandwidth` | Másolási sávszélesség (HtoD/DtoH/DtoD, lapozható és pinned) és foglalási költség: `vulkanizer-bandwidth [MiB]` |

Példák:

```sh
./test.sh --all-devices            # RADV + llvmpipe
./run.sh --log trace device_query  # minden API-hívás naplózva
./run.sh ./some_cuda_app           # külső CUDA-program (-cudart shared-del fordítva)
```

## Könyvtárszerkezet

```
include/cuda.h        clean-room CUDA Driver API header (részhalmaz)
src/common            naplózás
src/vk                Vulkan instance/eszköz, bufferek, pipeline-ok
src/ptx               PTX lexer, modul- és utasításparser
src/compiler          CFG + strukturálás, SPIR-V builder, PTX -> SPIR-V fordító
src/driver            a libcuda.so API-felülete
src/runtime           libcudart.so.12 és .so.13 (a CUDA Runtime API a Driver API-ra építve, egy forrásból)
src/cublas            libcublas.so.12/.13 (GEMM/TRSM GLSL-kernelekkel), include/cublas/cublas_v2.h
tests/                Driver API tesztek + belső Vulkan teszt
tests/cuda            nvcc-vel fordított Runtime API programok
tests/cudart_abi      a clean-room runtime típusok ABI-egyezése a hivatalos headerekkel
tests/data            teszt-PTX fájlok, mellettük a .cu forrásaik; a gen_ptx.sh újragenerálja őket (clang)
tools/                vulkanizer-info (eszközlista), vulkanizer-ptxc (offline fordító), vulkanizer-bandwidth, setup_cuda.sh,
                      build_llama.sh (a llama.cpp CUDA-buildje a Vulkanizer ellen)
```
