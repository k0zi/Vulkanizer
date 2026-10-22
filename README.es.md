![Vulkanizer – trae la paz a la galaxia. Una vulcana frente a un ordenador que ejecuta un LLM, al estilo de los años sesenta.](assets/vulkanizer-hero.png)

[English](README.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | **Español** | [Português](README.pt.md) | [Magyar](README.hu.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# Fecha estelar 2026.10.21 #
La Galaxia de la Inteligencia Artificial está al borde del colapso arquitectónico total.
Durante ciclos, el cosmos ha estado atrapado en una brutal guerra de optimización. Surgidos de los sectores de código abierto, los feroces Rocmulanos Rojos chocan sin descanso contra las flotas dominantes y propietarias de los Klincudas Verdes, cada bando desesperado por conquistar la supremacía absoluta de ejecución sobre las redes neuronales.
Sin embargo, mientras estos dos titanes queman potencia de cálculo, una nueva y aterradora amenaza se cierne sobre el borde galáctico: el Colectivo Intel-igente Azul. Acumulando fuerza multiplataforma con sus directivas XPU omnívoras, se prepara para asimilar cada carga de trabajo y hacerse con su propia y enorme porción del universo digital.
En esta era de caos sin precedentes, el destino del cosmos pende de un hilo. La devastadora colisión de tres pipelines solo puede evitarla una figura legendaria: el Comandante Vulkanizer. Armado con lógica absoluta y una eficiencia de compilación sin igual, debe emprender una última y lógica misión para vulcanizar toda la galaxia: unificar los entornos de ejecución fragmentados bajo un único estándar multiplataforma antes de que el universo quede inutilizado para siempre.

# Vulkanizer

Sustituto directo de `libcuda.so`, `libcudart.so` y `libcublas.so`: implementa las API CUDA Driver y Runtime (y la
familia GEMM de cuBLAS) sobre Vulkan Compute, de modo que los programas CUDA se ejecutan sin tarjeta NVIDIA, en
cualquier GPU con Vulkan 1.2 (AMD, Intel, llvmpipe).
Plan, arquitectura y fases: [plan.md](plan.md) (en húngaro).

## Requisitos

CMake ≥ 3.20, un compilador de C++20, un controlador Vulkan 1.2+ (bufferDeviceAddress + shaderInt64), `glslc` para las
pruebas.

Opcional (nombres de paquetes de Ubuntu):

| Paquete | Para qué |
|---|---|
| `spirv-tools` | `spirv-val` / `spirv-dis` – validar el SPIR-V generado (`vulkanizer-ptxc --val --dis`) |
| `vulkan-validationlayers` | `./run.sh --validation` |
| `clang` | Generar el PTX de prueba sin el SDK de CUDA (`tests/data/gen_ptx.sh`) |
| `ninja-build` | Compilaciones más rápidas (`build.sh` lo usa automáticamente si está presente) |
| `g++-13` | Compilador anfitrión para nvcc (CUDA 12.9 no acepta gcc 15) |

Pruebas compiladas con un nvcc real: `tools/setup_cuda.sh` descarga el nvcc y el runtime de NVIDIA, como paquetes
redistribuibles, en `.cuda/` (no hace falta controlador, GPU ni root). Si `.cuda/` existe, CMake compila los programas
`tests/cuda/*.cu` con nvcc y ctest los ejecuta con las bibliotecas de Vulkanizer.
`tools/setup_cuda.sh --cuda13` hace lo mismo con CUDA 13.4 en `.cuda13/`: los programas se vuelven a compilar con
nvcc 13 y se ejecutan sobre `libcudart.so.13` (pruebas que terminan en `_13`).

## Uso

### 1. Compilación

```sh
./build.sh --release                       # compilación Release en build/
VULKANIZER_BUILD_DIR=$PWD/build-rel ./build.sh --release   # o en un directorio aparte
build/bin/vulkanizer-info                       # qué dispositivos Vulkan ve Vulkanizer
./test.sh                                  # pruebas (opcional)
```

Las bibliotecas compiladas están en `build/lib/`: `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`,
`libcublasLt.so.12`. `build/lib/cuda13/` contiene las versiones CUDA 13 del runtime y de cuBLAS (`libcudart.so.13`,
`libcublas.so.13`, `libcublasLt.so.13`), para programas compilados con CUDA 13; usan la misma `libcuda.so.1`.

#### Instalación (paquete)

```sh
tools/package.sh                               # build-pkg/vulkanizer_<versión>_<arq>.deb y un .tar.gz
sudo apt install ./build-pkg/vulkanizer_*.deb
vulkanizer-run ./my_cuda_app args...           # cualquier programa CUDA 12 o 13
vulkanizer-run --torch python script.py        # PyTorch de pip o conda (CUDA 12 o 13, se detecta)
```

Las bibliotecas van a `/usr/lib/<arq>/vulkanizer/`, fuera de la ruta de búsqueda del enlazador: un controlador NVIDIA
en la misma máquina sigue funcionando para todo lo demás, y `vulkanizer-run` antepone Vulkanizer solo para el programa
que inicia. `cmake --install build --prefix DIR` instala la misma estructura en cualquier sitio, sin paquete.

### 2. Ejecutar un programa CUDA

Vulkanizer ocupa el lugar de las bibliotecas reales de NVIDIA; basta con que el enlazador dinámico las encuentre primero:

```sh
./run.sh ./my_cuda_app args...                       # run.sh define LD_LIBRARY_PATH
LD_LIBRARY_PATH=$PWD/build/lib ./my_cuda_app args... # lo mismo, a mano
./run.sh --log info ./my_cuda_app                    # muestra en qué GPU se ejecuta
./run.sh --device llvmpipe ./my_cuda_app             # un dispositivo concreto (los de CPU requieren VULKANIZER_ALLOW_CPU=1)
```

El programa debe cumplir dos requisitos:

- **El binario debe contener PTX.** Vulkanizer compila PTX a SPIR-V; no puede usar código máquina (SASS). Indique una
  arquitectura virtual al compilar, p. ej.
  `nvcc -arch=compute_61 ...` o en CMake `CMAKE_CUDA_ARCHITECTURES=61-virtual`.
  61 (Pascal) es una buena opción: tiene `dp4a` y no hay rutas de código para tensor cores.
- **El runtime de CUDA debe enlazarse dinámicamente** (`nvcc -cudart shared`, en CMake
  `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`); si no, la libcudart estática de NVIDIA acaba dentro del programa.

Compilar un programa propio con el nvcc descargado (sin GPU ni controlador de NVIDIA):

```sh
tools/setup_cuda.sh                                   # nvcc + cabeceras del runtime en .cuda/
.cuda/bin/nvcc -ccbin g++-13 -arch=compute_61 -cudart shared vadd.cu -o vadd
./run.sh ./vadd
```

Para programas que usan cuBLAS, utilice la cabecera `include/cublas/cublas_v2.h` de Vulkanizer y
`build/lib/libcublas.so` (`-Iinclude/cublas -Lbuild/lib -lcublas`).

La primera ejecución es más lenta: en ella los kernels se compilan a SPIR-V y se guardan en la caché en disco
(`~/.cache/vulkanizer`). Los arranques posteriores los cargan desde la caché.

### 3. llama.cpp

```sh
tools/setup_cuda.sh
./build.sh --release
tools/build_llama.sh                  # clona y compila .work/llama.cpp con el backend CUDA
./run.sh .work/llama-build/bin/llama-simple -m model.gguf -ngl 99 "Hello"
./run.sh .work/llama-build/bin/llama-bench -m model.gguf -ngl 99
```

`VULKANIZER_FAST_FMA=1` acelera la generación de tokens en las GPU cuyo `fma` de Vulkan no está fusionado
(p. ej. AMD GFX9): por defecto, Vulkanizer emula allí un fma exacto; esta opción usa en su lugar la versión más rápida
pero menos precisa del controlador. Para los LLM la diferencia es despreciable.

### 4. bitnet.c

El backend CUDA de bitnet.c (`--cuda`) funciona sobre Vulkan mediante `run_bitnet.sh`:

```sh
./run_bitnet.sh models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The capital of France is" -n 64
./run_bitnet.sh --profile --log info models/model.gguf -p "Hello"   # tiempos de GPU, registro
./run_bitnet.sh --build-only                                        # solo compilar
```

El script no modifica el checkout (`--bitnet-dir`, por defecto `~/Develop/GitHub/bitnet.c`): sincroniza las fuentes en
`.work/bitnet-vulkanizer` y las compila allí a PTX (`compute_61`) contra el cuBLAS y el runtime de Vulkanizer; añade
automáticamente la opción `--cuda`. Las rutas relativas de los modelos también se buscan en el checkout.

### 5. PyTorch

```sh
tools/setup_torch.sh                       # entorno PyTorch CUDA de conda-forge en .work/torch-cf (~9 GB)
./run_torch.sh -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
./run_torch.sh torch_mlp                   # entrena un MLP en GPU y CPU y los compara (libtorch, C++)
./run_torch.sh torch_ops                   # conv, softmax, LayerNorm, attention, capa Transformer, ...
./run_torch.sh torch_graph                 # una pasada hacia delante capturada en un grafo CUDA y reproducida
./run_torch.sh tests/torch/triton_ops.py   # un kernel Triton escrito a mano y el bmm Triton de torch._native
./run_torch.sh tests/torch/compile.py      # torch.compile (Inductor -> Triton), también mode="reduce-overhead"

tools/setup_torch.sh --cuda13              # la versión CUDA 13.4 en .work/torch-cf13
./run_torch.sh --env .work/torch-cf13 torch13_ops
```

`torch_mlp`, `torch_ops`, `torch_split` y `torch_graph` son pruebas en C++ contra la libtorch del entorno
(`tests/torch/*.cpp`); CMake las compila si `.work/torch-cf` existe al configurar, y como `torch13_*` contra
`.work/torch-cf13`. Para un entorno con `libcudart.so.13` (una versión CUDA 13), `run_torch.sh` carga las bibliotecas
desde `build/lib/cuda13`.

Los wheels oficiales de PyTorch solo contienen código máquina nativo de NVIDIA (SASS), sin PTX, así que Vulkanizer no
puede ejecutar sus kernels. Las versiones CUDA de conda-forge incluyen también PTX (`compute_120`):
`tools/setup_torch.sh` las instala con micromamba, sin instalación a nivel de sistema. `run_torch.sh` carga
`libcuda`, `libcudart`, `libcublas` y `libcublasLt` de Vulkanizer en lugar de las de NVIDIA y desactiva todo lo que
requiere hardware o bibliotecas de NVIDIA: cuDNN, los kernels de atención fusionados
(`scaled_dot_product_attention` usa la ruta matemática) y los kernels DSL de `torch._native`
(`TORCH_DISABLE_NATIVE_JIT=1`): sus kernels Triton funcionan, pero sus kernels CuTe DSL se compilan a SASS, y el mismo
interruptor apaga ambos. `VULKANIZER_TORCH_DEFAULTS=0` omite todo esto.
Triton en sí (`torch.compile`, `@triton.jit`) funciona: `tools/triton/ptxas` deja pasar el PTX de Triton en lugar de
ensamblarlo, y el lanzador de Triton se compila contra `include/cuda.h` de Vulkanizer (requiere un compilador de C).

### 6. ComfyUI

ComfyUI también funciona mediante `run_torch.sh`, en su propio entorno de conda-forge dentro del checkout de ComfyUI:

```sh
VK=~/Develop/GitHub/Vulkanizer             # este repositorio, compilado con ./build.sh --release
cd ~/Develop/GitHub                        # donde deba quedar ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI

# PyTorch (versión CUDA 12.9 con PTX) y las dependencias nativas de torchvision
export MAMBA_ROOT_PREFIX=$VK/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
[ -x $VK/.work/micromamba/bin/micromamba ] || { mkdir -p $VK/.work/micromamba && curl -fsSL \
    https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $VK/.work/micromamba bin/micromamba; }
$VK/.work/micromamba/bin/micromamba create -y -p $PWD/venv-vulkanizer -c conda-forge \
    python=3.13 "pytorch=2.14.1=cuda129_generic*" pip pillow giflib libwebp-base libjpeg-turbo libpng \
    "libnvjpeg>=12.4,<13" "libcusolver<12" "libcusparse<13"
# torchvision por URL, sin resolución: su versión CUDA 12.9 pide un cuDNN más reciente que necesita el paquete
# virtual __cuda_arch (un controlador real). Bajo Vulkanizer, cuDNN está desactivado de todos modos.
$VK/.work/micromamba/bin/micromamba install -y -p $PWD/venv-vulkanizer --file <(printf '@EXPLICIT\n%s\n' \
    https://conda.anaconda.org/conda-forge/linux-64/torchvision-0.29.1-cuda129_py313_hcd9f930_0.conda)

# el resto de requisitos de ComfyUI; pip no debe sustituir torch/torchvision por los wheels solo con SASS
venv-vulkanizer/bin/pip install -r <(grep -vE '^(torch|torchvision)$' requirements.txt) -r manager_requirements.txt

# arranque
$VK/run_torch.sh --env venv-vulkanizer main.py --enable-manager --disable-cuda-malloc
```

Después abra <http://127.0.0.1:8188>. Los modelos van en `models/`, como siempre. Notas:

- `--disable-cuda-malloc`: no se usa el backend de asignación `cudaMallocAsync` de ComfyUI.
- DynamicVRAM (comfy-aimdo) funciona sobre la gestión de memoria virtual de Vulkanizer (VMM, buffers sparse). Si da
  problemas, `--disable-dynamic-vram` vuelve a la carga de modelos basada en estimaciones.
- `tools/torch/sitecustomize.py` aplica los ajustes de PyTorch solo cuando ComfyUI importa torch, y mantiene apagados
  los backends de atención fusionados aunque ComfyUI los vuelva a activar: la atención usa la ruta matemática.
- El mensaje «PyTorch is not installed» de ComfyUI-Manager se debe a que falta `torchaudio` (conda-forge no tiene una
  versión para torch 2.14); la generación de imágenes no lo necesita.
- La primera generación es lenta mientras los kernels se compilan a SPIR-V; las siguientes los cargan desde la caché.
- `$VK/run_torch.sh --log info --env venv-vulkanizer main.py ...` muestra qué GPU Vulkan se usa.
- `python3 tests/torch/zimage_forward.py` compara la GPU con la CPU: una pasada hacia delante de Z-Image Turbo a
  256×256 en fp16 en la GPU y en fp32 y fp16 en la CPU. La GPU no debe alejarse de fp32 más que el propio fp16 (ComfyUI
  ejecuta el modelo en fp16 en la GPU, por lo que las imágenes completas difieren de una ejecución fp32 en CPU en los
  detalles finos).

### Variables de entorno útiles

| Variable | Efecto |
|---|---|
| `VULKANIZER_LOG=info\|debug\|trace` | Registro en stderr; `trace` muestra cada llamada a la API |
| `VULKANIZER_DEVICE=<nombre>` | Elige un dispositivo Vulkan por fragmento de nombre (`radv`, `llvmpipe`, ...) |
| `VULKANIZER_ALLOW_CPU=1` | Permite dispositivos Vulkan de CPU (llvmpipe) |
| `VULKANIZER_PROFILE=1` | Al salir, tiempo de GPU por kernel y tiempo de anfitrión por llamada a la API |
| `VULKANIZER_BATCH_MS=<ms>` | Presupuesto de tiempo de GPU de un envío a la cola (por defecto 50; `0` = sin límite). Los kernels más largos se dividen para no alcanzar el tiempo límite de GPU del controlador del kernel (amdgpu: 2 s) |
| `CUDA_LAUNCH_BLOCKING=1` | Cada lanzamiento de kernel espera a la GPU, así los errores y bloqueos aparecen en el lanzamiento que los causó |
| `VULKANIZER_FAST_FMA=1` | `fma` f32 más rápido y menos preciso |
| `VULKANIZER_GEMM_F32=1` | Los GEMM de cuBLAS con `COMPUTE_16F` también multiplican en f32 |
| `VULKANIZER_CACHE_DIR=<dir>\|off` | Ubicación de la caché de kernels, o desactivarla |
| `VULKANIZER_CC=86` | La compute capability que se informa |
| `VULKANIZER_DUMP=<dir>` | Vuelca el SPIR-V generado (`<kernel>.spv`) |

Lista completa: [plan.md, sección 4](plan.md).

### Limitaciones

- No se ejecuta SASS: solo funcionan los binarios que contienen PTX.
- CUDA Graphs: la captura de streams y la reproducción funcionan (los grafos de llama.cpp, `CUDAGraph` de PyTorch,
  `torch.compile(mode="reduce-overhead")`); no hay grafos hijos, nodos de memoria ni nodos condicionales.
- (Aún) no compatible: texturas, cuDNN, NCCL, varias GPU.
- La memoria managed, fijada (pinned) y registrada es memoria del anfitrión que la GPU usa en su sitio, en la misma
  dirección; la memoria managed nunca migra a la GPU (en una GPU integrada es la misma memoria de todos modos).
- Mientras exista memoria del anfitrión registrada (`cuMemHostRegister`), cada envío a la GPU es más lento (el kernel
  revalida las páginas importadas, ~4 ms por 256 MB); la memoria fijada y la managed no se ven afectadas.
- La flash attention de llama.cpp funciona correctamente (`tools/build_llama.sh --fa`), pero sin tensor cores el
  procesamiento del prompt es más lento con ella, por eso está desactivada por defecto.
- PyTorch: sin kernels de cuDNN, NCCL ni de atención fusionada; un `assert` fallido en el dispositivo solo detiene el
  hilo (no hay `cudaErrorAssert`).
- ComfyUI: sin las rutas optimizadas para CUDA 13 (cu130).

## Scripts

| Comando | Qué hace |
|---|---|
| `./build.sh` | Configura (si hace falta) y compila. `--release`, `--debug`, `--clean`, `--asan`/`--no-asan` |
| `./test.sh` | Compilación + todas las pruebas. `--device llvmpipe`, `--all-devices`, `--log debug`, `-R <regex>` |
| `./run.sh <programa>` | Ejecuta un programa con la `libcuda.so` de Vulkanizer. `--log trace`, `--device`, `--gdb`, `--validation` |
| `./clean.sh` | Borra el directorio `build/` |
| `./ci.sh` | Comprobación completa: Debug+ASAN y Release, en cada dispositivo (compila en `build-ci/`) |
| `tools/package.sh` | Paquetes de versión: un `.deb` y un `.tar.gz` en `build-pkg/` (`--check` también prueba el `.deb`) |
| `build/bin/vulkanizer-info` | Lista los dispositivos Vulkan que ve Vulkanizer |
| `build/bin/vulkanizer-ptxc` | PTX → SPIR-V sin conexión: `vulkanizer-ptxc k.ptx --list`, `-e vecAdd -o k.spv --dis --val` |
| `build/bin/vulkanizer-bandwidth` | Ancho de banda de copia (HtoD/DtoH/DtoD, paginable y fijada) y coste de asignación: `vulkanizer-bandwidth [MiB]` |

Ejemplos:

```sh
./test.sh --all-devices            # RADV + llvmpipe
./run.sh --log trace device_query  # se registra cada llamada a la API
./run.sh ./some_cuda_app           # programa CUDA externo (compilado con -cudart shared)
```

## Estructura de directorios

```
include/cuda.h        cabecera clean-room de la API CUDA Driver (subconjunto)
src/common            registro
src/vk                instancia/dispositivo Vulkan, buffers, pipelines
src/ptx               lexer de PTX, analizador de módulos e instrucciones
src/compiler          CFG + estructuración, generador de SPIR-V, compilador PTX -> SPIR-V
src/driver            la superficie de la API de libcuda.so
src/runtime           libcudart.so.12 y .so.13 (API CUDA Runtime sobre la API Driver, una sola fuente)
src/cublas            libcublas.so.12/.13 (GEMM/TRSM con kernels GLSL), include/cublas/cublas_v2.h
tests/                pruebas de la API Driver + prueba interna de Vulkan
tests/cuda            programas de la API Runtime compilados con nvcc
tests/cudart_abi      conformidad ABI de los tipos clean-room del runtime con las cabeceras oficiales
tests/data            archivos PTX de prueba con sus fuentes .cu al lado; gen_ptx.sh los regenera (clang)
tools/                vulkanizer-info (lista de dispositivos), vulkanizer-ptxc (compilador sin conexión), vulkanizer-bandwidth, setup_cuda.sh,
                      build_llama.sh (compilación CUDA de llama.cpp contra Vulkanizer)
```
