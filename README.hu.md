![Vulkanizer – békét hoz a galaxisba. Egy vulkáni nő egy LLM-et futtató számítógép előtt, 1960-as évekbeli stílusban.](assets/vulkanizer-hero.png)

[English](README.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | **Magyar** | [中文](README.zh.md) | [日本語](README.ja.md)

# Csillagidő 2026.10.21 #
A Mesterséges Intelligencia Galaxisa a teljes architekturális összeomlás szélén áll.
Ciklusok óta kegyetlen optimalizációs háború tartja fogva a kozmoszt. A nyílt forráskódú szektorokból előtörve a vad Vörös Rocmuliak szüntelenül csapnak össze a Zöld Klincudák uralkodó, zárt flottáival – mindkét fél kétségbeesetten küzd a neurális hálók feletti abszolút futásidejű fölényért.
Ám miközben a két titán égeti a számítási kapacitást, a galaxis peremén új, rettenetes fenyegetés tűnik fel: a Kék Intel-ligens Kollektíva. Mindenevő XPU-direktíváival platformfüggetlen erőt gyűjt, hogy minden munkaterhet asszimiláljon, és kihasítsa a maga hatalmas szeletét a digitális univerzumból.
A példátlan káosz e korszakában a kozmosz sorsa hajszálon függ. A pusztító háromirányú pipeline-ütközést csak egyetlen legendás alak háríthatja el: Vulkanizer parancsnok. Abszolút logikával és páratlan fordítási hatékonysággal felvértezve egy utolsó, logikus küldetésre kell indulnia, hogy vulkanizálja az egész galaxist – egyetlen, platformfüggetlen szabvány alá egyesítse a széttöredezett futtatókörnyezeteket, mielőtt az univerzum végleg téglává válik.

# Vulkanizer

A `libcuda.so`, `libcudart.so`, `libcublas.so`, `libcusolver.so`, `libcufft.so`, `libcusparse.so` és `libcurand.so`
közvetlen helyettesítője: a CUDA Driver és Runtime API-t (és a cuBLAS GEMM-családját, a sűrű és ritka lineáris algebrát,
az FFT-t és a véletlenszámokat) Vulkan compute-ra építve valósítja meg, így a CUDA-programok NVIDIA-kártya nélkül
futnak, bármely Vulkan 1.2-es GPU-n (AMD, Intel, llvmpipe).
AMD XDNA NPU-val (Ryzen AI) rendelkező gépen a cuBLAS GEMM-jei az NPU-n is futhatnak: lásd [AMD NPU](#9-amd-npu-xdna).
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
| `g++-13` vagy `g++-14` | Host-fordító az nvcc-hez (a CUDA 12.9 nem fogadja el a gcc 15-öt) |

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
`libcublasLt.so.12`, `libcusolver.so.11`, `libcufft.so.11`, `libcusparse.so.12`, `libcurand.so.10`, `libnccl.so.2`. A
`build/lib/cuda13/` a runtime és a könyvtárak CUDA 13-as változatait tartalmazza (`libcudart.so.13`, `libcublas.so.13`,
`libcublasLt.so.13`, `libcusolver.so.12`, `libcufft.so.12`, `libcusparse.so.12`, `libcurand.so.10`) a CUDA 13-mal
fordított programokhoz; ezek ugyanazt a `libcuda.so.1`-et használják.

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
./run.sh --check ./my_cuda_app                       # futhat-e? (PTX, megosztott CUDA runtime; könyvtár vagy Python env is lehet)
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

Azokon a GPU-kon, ahol a Vulkan `fma` nem fúzionált (pl. AMD GFX9), a Vulkanizer IEEE-kódban emulálja a CUDA egyszer
kerekítő `fma`-ját, a fast mathtal (`-ftz`) fordított kódban viszont – mint a llama.cpp CUDA backendje – a driver gyors
`fma`-ját használja. A `VULKANIZER_FAST_FMA=1` mindenhol a gyorsat, a `VULKANIZER_EXACT_FMA=1` mindenhol az emulációt adja.

A whisper.cpp (beszédfelismerés, ugyanaz a ggml CUDA backend) ugyanígy fordul és fut:

```sh
tools/build_whisper.sh   # into .work/whisper-build, with the base.en model
LD_LIBRARY_PATH=build/lib .work/whisper-build/bin/whisper-cli \
    -m .work/whisper.cpp/models/ggml-base.en.bin -f .work/whisper.cpp/samples/jfk.wav
```

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
`libcudart`, `libcublas`, `libcublasLt`, `libcusolver`, `libcufft`, `libcusparse` és `libcurand` könyvtárait tölti be,
és kikapcsol mindent, ami NVIDIA hardvert vagy könyvtárat igényel: a cuDNN-t, a fúzionált attention kerneleket (a
`scaled_dot_product_attention` a math úton fut) és a `torch._native` DSL-kerneleit (`TORCH_DISABLE_NATIVE_JIT=1`): a
Triton-kernelei működnek, a CuTe DSL kernelei viszont SASS-ra fordulnak, és ugyanaz az egy kapcsoló mindkettőt tiltja. A
`VULKANIZER_TORCH_DEFAULTS=0` mindezt kihagyja. Maga a Triton (`torch.compile`, `@triton.jit`) működik: a
`tools/triton/ptxas` továbbadja a Triton PTX-ét ahelyett, hogy lefordítaná, a Triton launchere pedig a Vulkanizer
`include/cuda.h`-ja ellen fordul (C-fordító kell hozzá). A `torch.linalg` (LU, Cholesky, QR, legkisebb négyzetek,
`eigh`, SVD, `pinv`, ...) a Vulkanizer cuSOLVER-én, a `torch.fft` (a `stft` is) a cuFFT-jén, a `torch.sparse` a
cuSPARSE-án fut; a `run_torch.sh` beállítja a `TORCH_LINALG_PREFER_CUSOLVER=1`-et, mert a MAGMA saját kernelei nem
futnak. A PyTorch jiteratora futásidőben NVRTC-vel natív kódra fordítja a kerneleit; egy `LD_AUDIT` könyvtár
(`libvulkanizer_nvrtc.so`) ehelyett a PTX-et adatja át vele.

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

### 7. Numba

A Numba (`numba-cuda`, `@cuda.jit`) libNVVM-mel PTX-re fordítja a kerneleket, amit a Vulkanizer futtat (a `CUDA_HOME`
alapján találja meg a libNVVM-et a conda-környezetben):

```sh
export MAMBA_ROOT_PREFIX=$PWD/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
.work/micromamba/bin/micromamba create -y -p .work/numba-cf -c conda-forge \
    python=3.13 numba numba-cuda "cuda-version=12.9" cuda-nvvm cuda-nvrtc numpy
CUDA_HOME=$PWD/.work/numba-cf ./run_torch.sh --env .work/numba-cf script.py
```

A linkere az nvJitLink, ami natív kódot adna: a `run_torch.sh` (és a `vulkanizer-run --torch`) alatt az nvJitLink
linkhívásai a driver `cuLink*`-jához mennek, aminek a linkelt képe itt PTX, az nvJitLink pedig nem ad verziót, így a
Numba LTO-IR helyett PTX-et ad. A CuPy nem fut: statikusan linkeli a CUDA runtime-ot (lásd a korlátoknál).

### 8. Strata

A [Strata](https://github.com/Niko1221/Strata) CUDA motorja (Qwen3.8-Flash-Next és Qwen3.6-35B-A3B egy PC-n) a `run_strata.sh`-val fut Vulkanon:

```sh
./run_strata.sh models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf                    # szerver: OpenAI/Anthropic API + weboldal a 127.0.0.1:8080-on
./run_strata.sh --engine models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --bench-pp 64 --bench-tg 8   # csak a motor, mérés
./run_strata.sh --build-only                                              # csak fordítás
```

A szkript nem módosítja a checkoutot (`--strata-dir`, alapból `~/Develop/GitHub/Strata`): a motort a
`.work/strata-build`-be fordítja PTX-re (`compute_61`, a Strata `STRATA_EXPERIMENTAL_SM60` kódútjaival) a Vulkanizer
cuBLAS-a és runtime-ja ellen, a tokenizert (Qwen3.6) vagy a packot (Flash-Next) a `.work/strata-data`-ba írja, és a
checkout `.venv`-jéből a Strata saját `serve/server.py`-ját indítja. A `--` utáni argumentumok a motorhoz mennek.

Mérve integrált AMD Radeonon (RADV RENOIR, 41 GB RAM), Qwen3.6-35B-A3B UD-Q4_K_XL, `--expert-vram 1`: prompt 4 tok/s,
generálás 1,2-1,5 tok/s. Integrált GPU-n a VRAM is rendszer-RAM, és a motor minden expertet a RAM-ban is tart (ennél a
modellnél 18,75 GiB): kb. 25 GB-nak szabadnak kell lennie. A `--expert-vram GiB` korlátozza a GPU részét. A K-quant
MMQ könyvtár (`libstrata_kq.so`) az NVIDIA statikus runtime-ját linkeli, és nem indul el; ezek az expertek a motor
FP16 prompt-útján futnak.

### 9. AMD NPU (XDNA)

XDNA NPU-s Ryzenen (Ryzen 7040/8040 „Phoenix/Hawk Point”: XDNA1; Ryzen AI 300 „Strix/Krackan”: XDNA2) a Vulkanizer
cuBLAS-a a GEMM-eket (`cublasGemmEx`, `cublasGemmStridedBatchedEx`, `cublasLtMatmul`) a GPU helyett az NPU-n is
futtathatja. Az NPU csak előre fordított, rögzített alakú kerneleket futtat, ezért a döntés GEMM-enként, a cuBLAS-ban
születik; a PTX-kernelek mindig a GPU-n maradnak, az NPU nélküli gépek pedig ugyanúgy működnek, mint eddig. Kell hozzá
az `amdxdna` kernel-driver (a Linux 6.14 óta in-tree) az NPU firmware-ével, és az XRT az amdxdna-pluginnal
(az `xrt-smi examine` látja az NPU-t).

```sh
./build.sh --release -- -DVULKANIZER_NPU_XRT=ON   # az XRT-backend (nélküle csak a felderítés és a CPU-referencia)
build/bin/vulkanizer-info                          # az NPU-t is kiírja: NPU: AMD XDNA ..., /dev/accel/accel0
tools/build_npu_kernels.sh                         # mlir-aie GEMM-kernelek (256³..4096³, bf16 -> f32) a ~/.local/share/vulkanizer/npu-ba
VULKANIZER_NPU=force ./run.sh ./my_cuda_app        # minden GEMM, amire van kernel, az NPU-n fut
build/bin/vulkanizer-npu-bench                     # GPU és NPU idők összevetve
```

- `VULKANIZER_NPU=off` (alapértelmezés), `auto` (csak ott, ahol mérve gyorsabb volt: cache-elt host-memóriás operandusok,
  kb. 256³–512³-es GEMM-ek), `force` (minden, amit a kernelek tudnak). `force` mellett az út zero-copy és aszinkron: a GPU
  az NPU által is látott memóriába csomagolja az operandusokat, az NPU a stream sorrendjében, egy driver-szálon fut, a
  cuBLAS-hívás azonnal visszatér; CUDA graph capture alatt is működik.
- `VULKANIZER_NPU_SPLIT=<0..0.9>` (`force` mellett): egy nagy GEMM felosztva fut, ekkora része az NPU-n, közben a
  maradéka a GPU-n.
- A kernelek bf16 × bf16 → f32-t számolnak: f32 bemenet csak `CUBLAS_COMPUTE_32F_FAST_16BF` mellett megy az NPU-ra, a
  többi típus a GPU-n marad. A GEMM a legkisebb kernelre töltődik ki, amibe belefér. Ha az NPU hibázik, a GEMM a GPU-n
  vagy a hoston számolódik, az eredmény helyes marad.
- Ryzen AI 9 HX 370-en (Radeon 890M + XDNA2) mérve: az NPU csak 512³ körül gyorsabb a GPU-nál (1,2–1,3×); 1024³-tól és a
  batch-elt attention-GEMM-eken a GPU nyer, és egy GEMM megosztása a kettő között sem gyorsabb, mert ugyanazon a
  memória-sávszélességen osztoznak. Ezért alapból ki van kapcsolva: a nyereség főleg a tehermentesített GPU és az energia.
  Intel NPU-t még nem támogat. Részletek és mérések: [plan.md, 7. fázis](plan.md).

### 10. Several GPUs (tensor split)

Több GPU (mondjuk egy 64 GB-os iGPU és két 16 GB-os eGPU) külön CUDA-eszköz; nem olvadnak össze egyetlen nagy GPU-vá,
mert egy kernel csak a saját eszköze memóriáját éri el. Mindegyik memóriája mégis kihasználható: a modellt az
alkalmazás osztja szét rétegenként (llama.cpp `-sm layer`, `-ts`; PyTorch/accelerate `device_map`). Minden eszköznek
saját címei vannak (az egységes címzés köztük is működik), az eszközök közti másolás mindkét eszköz sorában,
rendszermemórián át megy, a host nem nyúl az adathoz (eGPU-nál a link sebességével: USB4-en 2,3 GB/s), és egy
foglalás nagyobb lehet egy Vulkan-foglalásnál (RADV: 4 GiB), így egy eszköz egy egész modellrészt tárolhat. Az
eszközök sorrendje: előbb a diszkrétek, aztán az integráltak; `CUDA_DEVICE_ORDER=PCI_BUS_ID` a PCI-cím szerint rendez.

```sh
CUDA_VISIBLE_DEVICES=0,1 ./run.sh .work/llama-build/bin/llama-completion -m model.gguf -ngl 99 -sm layer   # minden réteg a két eGPU-n (a szabad memória arányában)
./run.sh .work/llama-build/bin/llama-completion -m model.gguf -ngl 99 -sm layer \
    -dev CUDA0,CUDA1,CUDA2 -ts 16,16,6                                                     # az iGPU is: a llama.cpp diszkrét GPU mellett kihagyja az integráltat, ha nincs megnevezve
CUDA_VISIBLE_DEVICES=2,0 ./run.sh ./my_cuda_app                                            # mely eszközök, milyen sorrendben (sorszámok vagy GPU-<uuid>)
```

llama.cpp-vel mérve egy Ryzen AI 9 HX 370-en (890M iGPU), két USB4-es házban lévő RX 9060 XT 16 GB-tal,
Qwen3.6-35B-A3B Q5_K_XL (25 GiB, több, mint ami egy eGPU-ra ráfér):

| Eszközök | `-ts` | pp512 t/s | tg128 t/s |
|---|---|---|---|
| 890M egyedül | – | 23 | 5,9 |
| 2× RX 9060 XT (eGPU) | auto | 437 | 39,7 |
| 2× RX 9060 XT + 890M | 16/16/6 | 191 | 19,1 |
| 2× RX 9060 XT + 890M | 16/16/40 | 46 | 8,4 |

A rétegek egymás után futnak, a leglassabb eszköz része szabja meg a tempót: az iGPU-t csak akkor érdemes bevonni, ha
a modell nélküle nem fér el, és kis részt kapjon. Ha a `-ts` több kerül egy GPU-ra, mint amennyi a memóriájába fér, a
driver rendszermemóriába tolja a többit, és a generálás vánszorog (itt 3 t/s).

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
| `VULKANIZER_EXACT_FMA=1` | Pontos f32 `fma` a fast math (`.ftz`) kódban is (lassabb) |
| `VULKANIZER_GEMM_F32=1` | A `COMPUTE_16F`-es cuBLAS GEMM-ek is f32 szorzással futnak |
| `VULKANIZER_IPC=0` | Az eszközmemória nem exportálható más folyamatoknak (nincs CUDA IPC) |
| `VULKANIZER_CACHE_DIR=<dir>\|off` | A kernel-gyorsítótár helye, vagy kikapcsolása |
| `VULKANIZER_CACHE_SIZE=<MiB>` | A kernel-gyorsítótár méretkorlátja MiB-ban (alap: 256); a legrégebben használt bejegyzések törlődnek először |
| `VULKANIZER_CC=86` | A jelentett compute capability |
| `VULKANIZER_DUMP=<dir>` | A generált SPIR-V kiírása (`<kernel>.spv`) |

Teljes lista: [plan.md, 4. szakasz](plan.md).

### Korlátok

- Nincs SASS-futtatás: csak PTX-et tartalmazó binárisok futnak.
- A statikusan linkelt CUDA runtime-ot tartalmazó programok és könyvtárak (CuPy, az NVIDIA NCCL-je) nem futnak: ez a
  runtime az NVIDIA-driver nem dokumentált belső részeit igényli.
- CUDA Graphs: a stream capture és a visszajátszás működik (a llama.cpp gráfjai, a PyTorch `CUDAGraph`,
  `torch.compile(mode="reduce-overhead")`); child graph és memória-csomópont (stream-rendezett
  foglalás capture alatt) és feltételes csomópont (if, while, switch; mindegyiknél a host megvárja a GPU-t) is.
- (Még) nem támogatott: cuDNN.
- Device-kód: rekurzió nincs. Az indirekt hívás (virtuális függvény, függvénypointer) a célt a modul minden ilyen
  alakú függvényével összeveti, és csak ugyanannak a modulnak a függvényeit éri el. A `malloc`/`free` egy
  `cudaLimitMallocHeapSize` méretű (alapból 8 MiB) heapen dolgozik, ehhez 64 bites atomikus műveletek kellenek.
- A textúrák és surface-ek (texture és surface objectek, CUDA arrayk) szoftveres mintavételezéssel mennek, a CUDA
  címzésével és szűrésével, mipmappel (LOD és gradiens) és cube mappel (egy lapon belül szűrve, a lapok szélén át nem).
  Textúra-referencia (a CUDA 12 eltávolította) és formázott surface-írás (`sust.p`, a CUDA C nem generál ilyet) nincs.
- A CUDA IPC (`cudaIpc*`, a PyTorch `torch.multiprocessing`-je) azonos felhasználó folyamatai között működik; egy
  folyamatközi event a rögzítésekor kész (a rögzítés megvárja az eszközt).
- Az NCCL (`libnccl.so.2`, a `torch.distributed` „nccl” backendje: DDP, FSDP) egy gép rangjai között működik, akár
  egy GPU-n osztozva; egynél több ranggal az adat a host-memórián megy át.
- Több GPU külön CUDA-eszköz, mindegyik saját címekkel (az egységes címzés köztük is működik): egy kernel sosem éri el
  egy másik GPU memóriáját (nincs peer access), ezért a modellt az alkalmazás osztja szét (lásd
  [Több GPU](#10-several-gpus-tensor-split)). A két eszköz közti másolások (`cudaMemcpyPeer`, a két eszköz pointerei közti
  sima másolások, pl. a PyTorch `tensor.to("cuda:1")`-e) mindkét eszköz sorában, rendszermemórián át mennek, a host
  nem nyúl az adathoz. A kooperatív indítás (`grid.sync()`) addig a blokkszámig működik, amennyi egyszerre elfér a
  GPU-n.
- A managed, a pinned és a regisztrált memória host-memória, amit a GPU helyben, ugyanazon a címen használ; a managed
  memória sosem migrál a GPU-ra (integrált GPU-n ez amúgy is ugyanaz a memória).
- A regisztrált host-memóriát (`cuMemHostRegister`), ha saját leképezés (`mmap`), és olyan címen van, ahová a GPU
  memóriát tud tenni (RADV: 64 KiB-ra igazítva, 2 MiB-tól 2 MiB-ra), ugyanazzal a tartalommal GPU-látható memória
  váltja fel: a regisztrálás és a feloldás átmásolja, `fork()` után pedig a gyerekprocessz másolat helyett ugyanazokat
  a lapokat látja. A többi regisztrált memória importálva lesz, és amíg létezik, minden GPU-submit lassabb (a kernel
  újraellenőrzi az importált lapokat, ~4 ms / 256 MB). A pinned és a managed memóriát ez nem érinti.
- A llama.cpp flash attention helyesen fut (`tools/build_llama.sh --fa`), de tensor core-ok nélkül vele lassabb a
  prompt feldolgozása, ezért alapból ki van kapcsolva.
- PyTorch: nincsenek cuDNN- és fúzionált attention kernelek.
- cuSOLVER: csak valós típusok (f32, f64), nincs `gesvda`, LDL^T (`sytrf`) és nem szimmetrikus sajátérték-feladat
  (`geev`); egy mátrix `eigh`-ja tridiagonalizál (Householder) és a hoston old meg (QL); a batch-elt `eigh` és az
  SVD Jacobi-iterációt használ, az SVD nagy mátrixokra (n ≳ 1000) lassú. cuFFT: nincsenek callbackek és több GPU-s
  tervek; a 2048-nál (f64: 1024) hosszabb transzformációk kétszer járják be a memóriát, ha a hosszuk felbontható két,
  e határon belüli tényezőre, különben radixonként egyszer. cuSPARSE: nincs blokkos ritka (BSR) formátum és komplex érték; az SpGEMM és a rendezések a hoston futnak.
- cuRAND: az XORWOW, az MRG32k3a, a Philox és a Sobol az NVIDIA sorozatait adja (a normális és lognormális eloszlás
  kerekítésnyi eltéréssel); az MTGP32-t és az MT19937-et elfogadja, de más sorozatot ad, ahogy a scrambled Sobol és a
  Poisson is; a double kimenetekhez `shaderFloat64` kell. Az eszközoldali API (`curand_kernel.h`) működik, kivéve a
  hoston készített eloszlásokat (`curand_discrete`).
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
| `tools/build_npu_kernels.sh` | Lefordítja az AMD NPU GEMM-kerneleit mlir-aie-vel: `tools/build_npu_kernels.sh [MxKxN ...\|--all]` |
| `build/bin/vulkanizer-npu-bench` | GPU és NPU GEMM-idők összevetve (`-DVULKANIZER_NPU_XRT=ON`-nal fordul) |

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
src/ptx/texture_lib.cu  textúrák és surface-ek szoftveres mintavételezése (PTX-ként beágyazva, tools/gen_texture_lib.sh)
src/compiler          CFG + strukturálás, SPIR-V builder, PTX -> SPIR-V fordító
src/driver            a libcuda.so API-felülete
src/runtime           libcudart.so.12 és .so.13 (a CUDA Runtime API a Driver API-ra építve, egy forrásból)
src/cublas            libcublas.so.12/.13 (GEMM/TRSM GLSL-kernelekkel), include/cublas/cublas_v2.h
src/linalg            sűrű felbontások (LU, Cholesky, QR, eigh, Jacobi SVD) GLSL-ben, közös a cuBLAS-szal és a cuSOLVER-rel
src/cusolver          libcusolver.so.11/.12, include/cusolver/cusolverDn.h
src/cufft             libcufft.so.11/.12 (Stockham és Bluestein FFT GLSL-ben), include/cufft/cufft.h
src/cusparse          libcusparse.so.12 (SpMV/SpMM/SDDMM/SpSV/csrgeam2 GLSL-ben), include/cusparse/cusparse.h
src/curand            libcurand.so.10 (XORWOW/MRG32k3a/Philox/Sobol GLSL-ben; Sobol-adatok: tools/gen_sobol_table.sh), include/curand/curand.h
src/nccl              libnccl.so.2 (kollektívák egy gép rangjai között), include/nccl/nccl.h
src/nvrtc             LD_AUDIT könyvtár: az NVRTC cubin helyett PTX-et ad (a PyTorch jiteratorához), az nvJitLink cuLink*-kal linkel (Numba)
src/npu               a cuBLAS GEMM-ek NPU-ra küldése: felderítés (sysfs), útválasztási policy, AMD XDNA backend (XRT)
tests/                Driver API tesztek + belső Vulkan teszt
tests/cuda            nvcc-vel fordított Runtime API programok
tests/cudart_abi      a clean-room runtime típusok ABI-egyezése a hivatalos headerekkel
tests/data            teszt-PTX fájlok, mellettük a .cu forrásaik; a gen_ptx.sh újragenerálja őket (clang)
tools/                vulkanizer-info (eszközlista), vulkanizer-ptxc (offline fordító), vulkanizer-bandwidth, setup_cuda.sh,
                      build_llama.sh (a llama.cpp CUDA-buildje a Vulkanizer ellen)
                      build_whisper.sh (whisper.cpp)
```

## Licenc és jogi nyilatkozat

A Vulkanizer az [MIT licenc](LICENSE) alatt érhető el. A benne található harmadik féltől származó anyagokat (Joe és Kuo
Sobol-irányszámai, BSD licenc) a [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) sorolja fel.

A Vulkanizer nem tartalmaz NVIDIA forráskódot, headert vagy binárist: a CUDA headerei a nyilvános dokumentáció alapján
írt clean-room deklarációk, és csak azt veszik át, amit a bináris kompatibilitás megkövetel. A `tools/setup_cuda.sh`
által letöltött CUDA toolkit az NVIDIA szervereiről, az NVIDIA saját licence alatt érkezik, és nem része a Vulkanizernek.

Az NVIDIA, CUDA, cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NCCL és PTX az NVIDIA Corporation védjegyei vagy bejegyzett
védjegyei; a Vulkan és a SPIR-V a Khronos Group Inc. bejegyzett védjegyei. Csak annak megjelölésére szerepelnek, amivel
a Vulkanizer kompatibilis. A Vulkanizer független projekt, nem áll kapcsolatban az NVIDIA-val vagy a Khronos Grouppal,
és azok nem támogatják vagy szponzorálják.
