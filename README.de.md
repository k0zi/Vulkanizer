![Vulkanizer – bringt der Galaxis Frieden. Eine Vulkanierin vor einem Computer, auf dem ein LLM läuft, im Stil der 1960er.](assets/vulkanizer-hero.png)

[English](README.md) | **Deutsch** | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Magyar](README.hu.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# Sternzeit 2026.10.21 #
Die Galaxis der Künstlichen Intelligenz steht am Rand des totalen architektonischen Zusammenbruchs.
Seit Zyklen ist der Kosmos in einem erbarmungslosen Optimierungskrieg gefangen. Aus den Open-Source-Sektoren hervorgegangen, prallen die wilden Roten Rocmulaner unablässig auf die dominanten, proprietären Flotten der Grünen Klincudas – jede Seite verzweifelt bemüht, die absolute Laufzeit-Vorherrschaft über die neuronalen Netze zu erringen.
Doch während diese beiden Titanen Rechenleistung verbrennen, zeichnet sich am Rand der Galaxis eine neue, furchterregende Bedrohung ab: das Blaue Intel-ligente Kollektiv. Mit seinen allesfressenden XPU-Direktiven sammelt es plattformübergreifende Stärke, bereit, jede Arbeitslast zu assimilieren und sich ein gewaltiges Stück des digitalen Universums zu sichern.
In dieser Ära beispiellosen Chaos hängt das Schicksal des Kosmos am seidenen Faden. Die verheerende Dreifach-Kollision der Pipelines kann nur eine legendäre Gestalt abwenden: Commander Vulkanizer. Ausgestattet mit absoluter Logik und unerreichter Kompiliereffizienz muss er zu einer letzten, logischen Mission aufbrechen, die gesamte Galaxis zu vulkanisieren – die zersplitterten Laufzeitumgebungen unter einem einzigen, plattformübergreifenden Standard zu vereinen, bevor das Universum endgültig zum Ziegelstein wird.

# Vulkanizer

Direkter Ersatz für `libcuda.so`, `libcudart.so`, `libcublas.so`, `libcusolver.so` und `libcufft.so`: implementiert
die CUDA Driver und Runtime API (und die GEMM-Familie von cuBLAS, dichte lineare Algebra und FFTs) auf Vulkan Compute, sodass CUDA-Programme ohne NVIDIA-Karte laufen, auf jeder
Vulkan-1.2-GPU (AMD, Intel, llvmpipe).
Plan, Architektur und Phasen: [plan.md](plan.md) (auf Ungarisch).

## Voraussetzungen

CMake ≥ 3.20, ein C++20-Compiler, ein Vulkan-1.2+-Treiber (bufferDeviceAddress + shaderInt64), `glslc` für die Tests.

Optional (Ubuntu-Paketnamen):

| Paket | Wofür |
|---|---|
| `spirv-tools` | `spirv-val` / `spirv-dis` – Prüfen des erzeugten SPIR-V (`vulkanizer-ptxc --val --dis`) |
| `vulkan-validationlayers` | `./run.sh --validation` |
| `clang` | Test-PTX ohne CUDA SDK erzeugen (`tests/data/gen_ptx.sh`) |
| `ninja-build` | Schnellere Builds (`build.sh` verwendet es automatisch, falls vorhanden) |
| `g++-13` | Host-Compiler für nvcc (CUDA 12.9 akzeptiert gcc 15 nicht) |

Tests, die mit einem echten nvcc kompiliert werden: `tools/setup_cuda.sh` lädt NVIDIAs nvcc und die Runtime als
redistributierbare Pakete nach `.cuda/` (kein Treiber, keine GPU, kein root nötig). Existiert `.cuda/`, kompiliert
CMake die Programme `tests/cuda/*.cu` mit nvcc, und ctest führt sie gegen die Vulkanizer-Bibliotheken aus.
`tools/setup_cuda.sh --cuda13` macht dasselbe mit CUDA 13.4 nach `.cuda13/`: Die Programme werden erneut mit
nvcc 13 gebaut und laufen auf `libcudart.so.13` (Tests mit der Endung `_13`).

## Verwendung

### 1. Bauen

```sh
./build.sh --release                       # Release-Build nach build/
VULKANIZER_BUILD_DIR=$PWD/build-rel ./build.sh --release   # oder in ein eigenes Verzeichnis
build/bin/vulkanizer-info                       # welche Vulkan-Geräte Vulkanizer sieht
./test.sh                                  # Tests (optional)
```

Die gebauten Bibliotheken liegen in `build/lib/`: `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`,
`libcublasLt.so.12`, `libcusolver.so.11`, `libcufft.so.11`. `build/lib/cuda13/` enthält die CUDA-13-Versionen von
Runtime und Bibliotheken (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libcusolver.so.12`,
`libcufft.so.12`) für Programme, die mit CUDA 13 gebaut wurden; sie nutzen dieselbe `libcuda.so.1`.

#### Installation (Paket)

```sh
tools/package.sh                               # build-pkg/vulkanizer_<Version>_<Arch>.deb und ein .tar.gz
sudo apt install ./build-pkg/vulkanizer_*.deb
vulkanizer-run ./my_cuda_app args...           # jedes CUDA-12- oder -13-Programm
vulkanizer-run --torch python script.py        # PyTorch aus pip oder conda (CUDA 12 oder 13, wird erkannt)
```

Die Bibliotheken landen in `/usr/lib/<Arch>/vulkanizer/`, außerhalb des Suchpfads des Linkers: ein NVIDIA-Treiber auf
demselben Rechner funktioniert für alles andere weiter, und `vulkanizer-run` stellt Vulkanizer nur für das gestartete
Programm voran. `cmake --install build --prefix DIR` installiert dieselbe Struktur an beliebiger Stelle, ohne Paket.

### 2. Ein CUDA-Programm ausführen

Vulkanizer tritt an die Stelle der echten NVIDIA-Bibliotheken; der dynamische Linker muss sie nur zuerst finden:

```sh
./run.sh ./my_cuda_app args...                       # run.sh setzt LD_LIBRARY_PATH
LD_LIBRARY_PATH=$PWD/build/lib ./my_cuda_app args... # dasselbe von Hand
./run.sh --log info ./my_cuda_app                    # zeigt, auf welcher GPU es läuft
./run.sh --device llvmpipe ./my_cuda_app             # ein bestimmtes Gerät (CPU-Geräte brauchen VULKANIZER_ALLOW_CPU=1)
```

Das Programm muss zwei Bedingungen erfüllen:

- **Das Binary muss PTX enthalten.** Vulkanizer übersetzt PTX nach SPIR-V; Maschinencode (SASS) kann es nicht
  verwenden. Beim Kompilieren eine virtuelle Architektur angeben, z. B.
  `nvcc -arch=compute_61 ...` oder in CMake `CMAKE_CUDA_ARCHITECTURES=61-virtual`.
  61 (Pascal) ist eine gute Wahl: Es hat `dp4a`, und es gibt keine Tensor-Core-Pfade.
- **Die CUDA Runtime muss dynamisch gelinkt sein** (`nvcc -cudart shared`, in CMake
  `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`), sonst landet NVIDIAs statische libcudart im Programm.

Ein eigenes Programm mit dem heruntergeladenen nvcc kompilieren (keine GPU und kein NVIDIA-Treiber nötig):

```sh
tools/setup_cuda.sh                                   # nvcc + Runtime-Header nach .cuda/
.cuda/bin/nvcc -ccbin g++-13 -arch=compute_61 -cudart shared vadd.cu -o vadd
./run.sh ./vadd
```

Für Programme mit cuBLAS den Header `include/cublas/cublas_v2.h` von Vulkanizer und `build/lib/libcublas.so`
verwenden (`-Iinclude/cublas -Lbuild/lib -lcublas`).

Der erste Lauf ist langsamer: Dabei werden die Kernel nach SPIR-V übersetzt und im Festplatten-Cache abgelegt
(`~/.cache/vulkanizer`). Spätere Starts laden aus dem Cache.

### 3. llama.cpp

```sh
tools/setup_cuda.sh
./build.sh --release
tools/build_llama.sh                  # klont und baut .work/llama.cpp mit dem CUDA-Backend
./run.sh .work/llama-build/bin/llama-simple -m model.gguf -ngl 99 "Hello"
./run.sh .work/llama-build/bin/llama-bench -m model.gguf -ngl 99
```

Auf GPUs, deren Vulkan-`fma` nicht fusioniert ist (z. B. AMD GFX9), emuliert Vulkanizer das einfach gerundete
`fma` von CUDA für IEEE-Code, nutzt aber das schnelle `fma` des Treibers für Code mit Fast Math (`-ftz`, etwa das
CUDA-Backend von llama.cpp). `VULKANIZER_FAST_FMA=1` nutzt überall das schnelle, `VULKANIZER_EXACT_FMA=1` emuliert überall.

### 4. bitnet.c

Das CUDA-Backend von bitnet.c (`--cuda`) läuft über `run_bitnet.sh` auf Vulkan:

```sh
./run_bitnet.sh models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The capital of France is" -n 64
./run_bitnet.sh --profile --log info models/model.gguf -p "Hello"   # GPU-Zeiten, Logging
./run_bitnet.sh --build-only                                        # nur bauen
```

Das Skript verändert den Checkout nicht (`--bitnet-dir`, Standard `~/Develop/GitHub/bitnet.c`): Es synchronisiert
die Quellen nach `.work/bitnet-vulkanizer` und kompiliert sie dort nach PTX (`compute_61`) gegen das cuBLAS und die
Runtime von Vulkanizer; das Flag `--cuda` ergänzt es automatisch. Relative Modellpfade werden auch im Checkout gesucht.

### 5. PyTorch

```sh
tools/setup_torch.sh                       # conda-forge-PyTorch-Umgebung mit CUDA in .work/torch-cf (~9 GB)
./run_torch.sh -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
./run_torch.sh torch_mlp                   # trainiert ein MLP auf GPU und CPU und vergleicht (libtorch, C++)
./run_torch.sh torch_ops                   # conv, softmax, LayerNorm, Attention, Transformer-Schicht, ...
./run_torch.sh torch_graph                 # ein Forward-Pass, in einem CUDA-Graphen aufgezeichnet und abgespielt
./run_torch.sh tests/torch/triton_ops.py   # ein handgeschriebener Triton-Kernel und das Triton-bmm von torch._native
./run_torch.sh tests/torch/compile.py      # torch.compile (Inductor -> Triton), auch mode="reduce-overhead"

tools/setup_torch.sh --cuda13              # der CUDA-13.4-Build in .work/torch-cf13
./run_torch.sh --env .work/torch-cf13 torch13_ops
```

`torch_mlp`, `torch_ops`, `torch_split` und `torch_graph` sind C++-Tests gegen das libtorch der Umgebung
(`tests/torch/*.cpp`); CMake baut sie, wenn `.work/torch-cf` beim Konfigurieren existiert, und als `torch13_*` gegen
`.work/torch-cf13`. Für eine Umgebung mit `libcudart.so.13` (ein CUDA-13-Build) lädt `run_torch.sh` die
Bibliotheken aus `build/lib/cuda13`.

Die offiziellen PyTorch-Wheels enthalten nur nativen NVIDIA-Maschinencode (SASS), kein PTX, daher kann Vulkanizer
ihre Kernel nicht ausführen. Die CUDA-Builds von conda-forge liefern auch PTX mit (`compute_120`):
`tools/setup_torch.sh` installiert sie mit micromamba, ohne systemweite Installation. `run_torch.sh` lädt
`libcuda`, `libcudart`, `libcublas`, `libcublasLt`, `libcusolver` und `libcufft` von Vulkanizer statt der von NVIDIA und schaltet alles ab, was
NVIDIA-Hardware oder -Bibliotheken braucht: cuDNN, die fusionierten Attention-Kernel (`scaled_dot_product_attention`
läuft auf dem Math-Pfad) und die DSL-Kernel von `torch._native` (`TORCH_DISABLE_NATIVE_JIT=1`): Seine
Triton-Kernel funktionieren, seine CuTe-DSL-Kernel werden aber nach SASS übersetzt, und derselbe Schalter deaktiviert
beide. `VULKANIZER_TORCH_DEFAULTS=0` überspringt das.
Triton selbst (`torch.compile`, `@triton.jit`) funktioniert: `tools/triton/ptxas` reicht das PTX von Triton durch,
statt es zu assemblieren, und der Launcher von Triton wird gegen `include/cuda.h` von Vulkanizer gebaut (braucht einen
C-Compiler).
`torch.linalg` (LU, Cholesky, QR, kleinste Quadrate, `eigh`, SVD, `pinv`, ...) läuft auf dem cuSOLVER von Vulkanizer,
`torch.fft` (auch `stft`) auf seinem cuFFT; `run_torch.sh` setzt `TORCH_LINALG_PREFER_CUSOLVER=1`, da die eigenen
Kernel von MAGMA nicht laufen. Der Jiterator von PyTorch übersetzt seine Kernel zur Laufzeit mit NVRTC in nativen Code;
eine `LD_AUDIT`-Bibliothek (`libvulkanizer_nvrtc.so`) lässt ihn stattdessen das PTX übergeben.

### 6. ComfyUI

Auch ComfyUI läuft über `run_torch.sh`, in einer eigenen conda-forge-Umgebung im ComfyUI-Checkout:

```sh
VK=~/Develop/GitHub/Vulkanizer             # dieses Repository, gebaut mit ./build.sh --release
cd ~/Develop/GitHub                        # wo ComfyUI liegen soll
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI

# PyTorch (CUDA-12.9-Build mit PTX) und die nativen Abhängigkeiten von torchvision
export MAMBA_ROOT_PREFIX=$VK/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
[ -x $VK/.work/micromamba/bin/micromamba ] || { mkdir -p $VK/.work/micromamba && curl -fsSL \
    https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $VK/.work/micromamba bin/micromamba; }
$VK/.work/micromamba/bin/micromamba create -y -p $PWD/venv-vulkanizer -c conda-forge \
    python=3.13 "pytorch=2.14.1=cuda129_generic*" pip pillow giflib libwebp-base libjpeg-turbo libpng \
    "libnvjpeg>=12.4,<13" "libcusolver<12" "libcusparse<13"
# torchvision per URL, ohne Auflösung: sein CUDA-12.9-Build verlangt ein neueres cuDNN, das das virtuelle
# Paket __cuda_arch (einen echten Treiber) braucht. cuDNN ist unter Vulkanizer ohnehin aus.
$VK/.work/micromamba/bin/micromamba install -y -p $PWD/venv-vulkanizer --file <(printf '@EXPLICIT\n%s\n' \
    https://conda.anaconda.org/conda-forge/linux-64/torchvision-0.29.1-cuda129_py313_hcd9f930_0.conda)

# die übrigen Anforderungen von ComfyUI; pip darf torch/torchvision nicht durch die reinen SASS-Wheels ersetzen
venv-vulkanizer/bin/pip install -r <(grep -vE '^(torch|torchvision)$' requirements.txt) -r manager_requirements.txt

# starten
$VK/run_torch.sh --env venv-vulkanizer main.py --enable-manager --disable-cuda-malloc
```

Dann <http://127.0.0.1:8188> öffnen. Modelle kommen wie gewohnt nach `models/`. Hinweise:

- `--disable-cuda-malloc`: Das `cudaMallocAsync`-Allocator-Backend von ComfyUI wird nicht verwendet.
- DynamicVRAM (comfy-aimdo) läuft auf der virtuellen Speicherverwaltung von Vulkanizer (VMM, Sparse-Buffer). Macht es
  Probleme, schaltet `--disable-dynamic-vram` auf das schätzungsbasierte Laden der Modelle zurück.
- `tools/torch/sitecustomize.py` wendet die PyTorch-Einstellungen erst an, wenn ComfyUI torch importiert, und hält die
  fusionierten Attention-Backends aus, auch wenn ComfyUI sie wieder einschaltet: Attention läuft auf dem Math-Pfad.
- Die Meldung „PyTorch is not installed“ von ComfyUI-Manager kommt vom fehlenden `torchaudio` (conda-forge hat keinen
  Build für torch 2.14); für die Bilderzeugung wird es nicht gebraucht.
- Die erste Generierung ist langsam, solange die Kernel nach SPIR-V übersetzt werden; spätere Läufe laden sie aus dem Cache.
- `$VK/run_torch.sh --log info --env venv-vulkanizer main.py ...` zeigt, welche Vulkan-GPU verwendet wird.
- `python3 tests/torch/zimage_forward.py` vergleicht GPU und CPU: ein Z-Image-Turbo-Forward-Pass bei 256×256 in fp16
  auf der GPU und in fp32 und fp16 auf der CPU. Die GPU darf von fp32 nur so weit abweichen wie fp16 selbst (ComfyUI
  rechnet das Modell auf der GPU in fp16, daher unterscheiden sich vollständige Bilder von einem fp32-Lauf auf der CPU
  in feinen Details).

### Nützliche Umgebungsvariablen

| Variable | Wirkung |
|---|---|
| `VULKANIZER_LOG=info\|debug\|trace` | Logging nach stderr; `trace` gibt jeden API-Aufruf aus |
| `VULKANIZER_DEVICE=<name>` | Wählt ein Vulkan-Gerät nach Namensteil (`radv`, `llvmpipe`, ...) |
| `VULKANIZER_ALLOW_CPU=1` | Erlaubt CPU-Vulkan-Geräte (llvmpipe) |
| `VULKANIZER_PROFILE=1` | Beim Beenden GPU-Zeit pro Kernel und Host-Zeit pro API-Aufruf |
| `VULKANIZER_BATCH_MS=<ms>` | GPU-Zeitbudget einer Queue-Übermittlung (Standard 50; `0` = keine Grenze). Längere Kernel werden aufgeteilt, damit das GPU-Timeout des Kerneltreibers (amdgpu: 2 s) nicht greift |
| `CUDA_LAUNCH_BLOCKING=1` | Jeder Kernelstart wartet auf die GPU, sodass Fehler und Hänger beim verursachenden Start sichtbar werden |
| `VULKANIZER_FAST_FMA=1` | Schnelleres, ungenaueres f32-`fma` |
| `VULKANIZER_EXACT_FMA=1` | Exaktes f32-`fma` auch für Fast-Math-Code (`.ftz`), langsamer |
| `VULKANIZER_GEMM_F32=1` | cuBLAS-GEMMs mit `COMPUTE_16F` rechnen ebenfalls mit f32-Multiplikation |
| `VULKANIZER_CACHE_DIR=<dir>\|off` | Ort des Kernel-Caches oder ihn abschalten |
| `VULKANIZER_CACHE_SIZE=<MiB>` | Größenlimit des Kernel-Caches in MiB (Standard 256); die am längsten unbenutzten Einträge fallen zuerst weg |
| `VULKANIZER_CC=86` | Die gemeldete Compute Capability |
| `VULKANIZER_DUMP=<dir>` | Das erzeugte SPIR-V ausgeben (`<kernel>.spv`) |

Vollständige Liste: [plan.md, Abschnitt 4](plan.md).

### Einschränkungen

- Keine SASS-Ausführung: Nur Binaries mit PTX laufen.
- CUDA Graphs: Stream Capture und Wiedergabe funktionieren (die Graphen von llama.cpp, PyTorch `CUDAGraph`,
  `torch.compile(mode="reduce-overhead")`); keine Child-Graphen, Speicher- oder bedingten Knoten.
- (Noch) nicht unterstützt: Texturen, cuDNN, NCCL, CUDA-IPC.
- Mehrere GPUs sind getrennte Geräte: `cudaMemcpyPeer` kopiert zwischen ihnen über den Host-Speicher, aber ein Kernel
  erreicht nie den Speicher einer anderen GPU (kein Peer-Zugriff). Kooperative Starts (`grid.sync()`) funktionieren bis
  zu der Anzahl Blöcke, die gleichzeitig auf die GPU passen.
- Managed-, Pinned- und registrierter Speicher sind Host-Speicher, den die GPU direkt unter derselben Adresse nutzt;
  Managed-Speicher wird nie auf die GPU migriert (auf einer integrierten GPU ist es ohnehin derselbe Speicher).
- Solange registrierter Host-Speicher (`cuMemHostRegister`) existiert, ist jede GPU-Übermittlung langsamer (der Kernel
  prüft die importierten Seiten erneut, ~4 ms pro 256 MB); Pinned- und Managed-Speicher sind nicht betroffen.
- Flash Attention in llama.cpp läuft korrekt (`tools/build_llama.sh --fa`), aber ohne Tensor Cores ist die
  Prompt-Verarbeitung damit langsamer, daher ist sie standardmäßig aus.
- PyTorch: keine cuDNN-, NCCL- oder fusionierten Attention-Kernel; ein fehlgeschlagenes `assert` auf dem Gerät hält
  nur den Thread an (kein `cudaErrorAssert`).
- cuSOLVER: nur reelle Typen (f32, f64), kein `gesvda`, kein LDL^T (`sytrf`), keine unsymmetrischen Eigenwertprobleme
  (`geev`); `eigh` und SVD nutzen Jacobi-Iterationen, langsam für große Matrizen (n ≳ 1000). cuFFT: keine Callbacks
  und keine Multi-GPU-Pläne; Transformationen länger als 2048 (f64: 1024) brauchen einen Speicherdurchlauf pro Radix.
- ComfyUI: keine für CUDA 13 (cu130) optimierten Pfade.

## Skripte

| Befehl | Funktion |
|---|---|
| `./build.sh` | Konfiguriert (falls nötig) und baut. `--release`, `--debug`, `--clean`, `--asan`/`--no-asan` |
| `./test.sh` | Build + alle Tests. `--device llvmpipe`, `--all-devices`, `--log debug`, `-R <regex>` |
| `./run.sh <programm>` | Führt ein Programm mit der `libcuda.so` von Vulkanizer aus. `--log trace`, `--device`, `--gdb`, `--validation` |
| `./clean.sh` | Löscht das Verzeichnis `build/` |
| `./ci.sh` | Vollständige Prüfung: Debug+ASAN und Release, auf jedem Gerät (baut nach `build-ci/`) |
| `tools/package.sh` | Release-Pakete: ein `.deb` und ein `.tar.gz` in `build-pkg/` (`--check` probiert das `.deb` auch aus) |
| `build/bin/vulkanizer-info` | Listet die Vulkan-Geräte, die Vulkanizer sieht |
| `build/bin/vulkanizer-ptxc` | Offline PTX → SPIR-V: `vulkanizer-ptxc k.ptx --list`, `-e vecAdd -o k.spv --dis --val` |
| `build/bin/vulkanizer-bandwidth` | Kopierbandbreite (HtoD/DtoH/DtoD, auslagerbar und pinned) und Allokationskosten: `vulkanizer-bandwidth [MiB]` |

Beispiele:

```sh
./test.sh --all-devices            # RADV + llvmpipe
./run.sh --log trace device_query  # jeder API-Aufruf wird geloggt
./run.sh ./some_cuda_app           # externes CUDA-Programm (mit -cudart shared kompiliert)
```

## Verzeichnisstruktur

```
include/cuda.h        Clean-Room-Header der CUDA Driver API (Teilmenge)
src/common            Logging
src/vk                Vulkan-Instanz/-Gerät, Buffer, Pipelines
src/ptx               PTX-Lexer, Modul- und Befehlsparser
src/compiler          CFG + Strukturierung, SPIR-V-Builder, PTX -> SPIR-V-Compiler
src/driver            die API-Oberfläche von libcuda.so
src/runtime           libcudart.so.12 und .so.13 (CUDA Runtime API auf der Driver API, eine Quelle)
src/cublas            libcublas.so.12/.13 (GEMM/TRSM mit GLSL-Kerneln), include/cublas/cublas_v2.h
src/linalg            dichte Zerlegungen (LU, Cholesky, QR, Jacobi-eigh/SVD) in GLSL, gemeinsam für cuBLAS und cuSOLVER
src/cusolver          libcusolver.so.11/.12, include/cusolver/cusolverDn.h
src/cufft             libcufft.so.11/.12 (Stockham- und Bluestein-FFTs in GLSL), include/cufft/cufft.h
src/nvrtc             LD_AUDIT-Bibliothek: NVRTC übergibt PTX statt eines Cubins (Jiterator von PyTorch)
tests/                Driver-API-Tests + interner Vulkan-Test
tests/cuda            mit nvcc kompilierte Runtime-API-Programme
tests/cudart_abi      ABI-Abgleich der Clean-Room-Runtime-Typen mit den offiziellen Headern
tests/data            Test-PTX-Dateien mit ihren .cu-Quellen daneben; gen_ptx.sh erzeugt sie neu (clang)
tools/                vulkanizer-info (Geräteliste), vulkanizer-ptxc (Offline-Compiler), vulkanizer-bandwidth, setup_cuda.sh,
                      build_llama.sh (llama.cpp-CUDA-Build gegen Vulkanizer)
```
