![Vulkanizer – apporte la paix à la galaxie. Une Vulcaine devant un ordinateur faisant tourner un LLM, dans le style des années 1960.](assets/vulkanizer-hero.png)

[English](README.md) | [Deutsch](README.de.md) | **Français** | [Español](README.es.md) | [Português](README.pt.md) | [Magyar](README.hu.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# Date stellaire 2026.10.21 #
La Galaxie de l’Intelligence Artificielle est au bord de l’effondrement architectural total.
Depuis des cycles, le cosmos est enlisé dans une guerre d’optimisation impitoyable. Issus des secteurs open source, les farouches Rocmuliens Rouges s’opposent sans relâche aux flottes dominantes et propriétaires des Klincudas Verts, chacun cherchant désespérément la suprématie absolue d’exécution sur les réseaux de neurones.
Mais tandis que ces deux titans consument leur puissance de calcul, une nouvelle menace terrifiante se profile aux confins de la galaxie : le Collectif Intel-ligent Bleu. Rassemblant une force multiplateforme grâce à ses directives XPU omnivores, il se prépare à assimiler toutes les charges de travail et à se tailler une part immense de l’univers numérique.
En cette ère de chaos sans précédent, le destin du cosmos ne tient qu’à un fil. Seule une figure légendaire peut empêcher la collision dévastatrice des trois pipelines : le Commandant Vulkanizer. Armé d’une logique absolue et d’une efficacité de compilation inégalée, il doit entreprendre une ultime mission, logique, pour vulkaniser la galaxie entière — unifier les environnements d’exécution fragmentés sous un standard unique et multiplateforme, avant que l’univers ne soit définitivement briqué.

# Vulkanizer

Remplacement direct de `libcuda.so`, `libcudart.so`, `libcublas.so`, `libcusolver.so`, `libcufft.so`, `libcusparse.so`
et `libcurand.so` : implémente les API CUDA Driver et Runtime (ainsi que la famille GEMM de cuBLAS, l’algèbre linéaire
dense et creuse, les FFT et les nombres aléatoires) sur Vulkan Compute, pour que les programmes CUDA tournent sans carte
NVIDIA, sur n’importe quel GPU Vulkan 1.2 (AMD, Intel, llvmpipe).
Plan, architecture et phases : [plan.md](plan.md) (en hongrois).

## Prérequis

CMake ≥ 3.20, un compilateur C++20, un pilote Vulkan 1.2+ (bufferDeviceAddress + shaderInt64), `glslc` pour les tests.

Facultatif (noms des paquets Ubuntu) :

| Paquet | Utilisé pour |
|---|---|
| `spirv-tools` | `spirv-val` / `spirv-dis` – valider le SPIR-V généré (`vulkanizer-ptxc --val --dis`) |
| `vulkan-validationlayers` | `./run.sh --validation` |
| `clang` | Générer le PTX de test sans le SDK CUDA (`tests/data/gen_ptx.sh`) |
| `ninja-build` | Compilations plus rapides (`build.sh` l’utilise automatiquement s’il est présent) |
| `g++-13` | Compilateur hôte pour nvcc (CUDA 12.9 n’accepte pas gcc 15) |

Tests compilés avec un vrai nvcc : `tools/setup_cuda.sh` télécharge nvcc et le runtime de NVIDIA, sous forme de
paquets redistribuables, dans `.cuda/` (ni pilote, ni GPU, ni root nécessaires). Si `.cuda/` existe, CMake compile les
programmes `tests/cuda/*.cu` avec nvcc et ctest les exécute avec les bibliothèques de Vulkanizer.
`tools/setup_cuda.sh --cuda13` fait de même avec CUDA 13.4 dans `.cuda13/` : les programmes sont recompilés avec
nvcc 13 et s’exécutent sur `libcudart.so.13` (tests se terminant par `_13`).

## Utilisation

### 1. Compilation

```sh
./build.sh --release                       # compilation Release dans build/
VULKANIZER_BUILD_DIR=$PWD/build-rel ./build.sh --release   # ou dans un répertoire séparé
build/bin/vulkanizer-info                       # les périphériques Vulkan que Vulkanizer voit
./test.sh                                  # tests (facultatif)
```

Les bibliothèques compilées se trouvent dans `build/lib/` : `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`,
`libcublasLt.so.12`, `libcusolver.so.11`, `libcufft.so.11`, `libcusparse.so.12`, `libcurand.so.10`, `libnccl.so.2`.
`build/lib/cuda13/` contient les versions CUDA 13 du runtime et des bibliothèques (`libcudart.so.13`, `libcublas.so.13`,
`libcublasLt.so.13`, `libcusolver.so.12`, `libcufft.so.12`, `libcusparse.so.12`, `libcurand.so.10`), pour les programmes
compilés avec CUDA 13 ; elles utilisent la même `libcuda.so.1`.

#### Installation (paquet)

```sh
tools/package.sh                               # build-pkg/vulkanizer_<version>_<arch>.deb et un .tar.gz
sudo apt install ./build-pkg/vulkanizer_*.deb
vulkanizer-run ./my_cuda_app args...           # tout programme CUDA 12 ou 13
vulkanizer-run --torch python script.py        # PyTorch de pip ou conda (CUDA 12 ou 13, détecté)
```

Les bibliothèques vont dans `/usr/lib/<arch>/vulkanizer/`, hors du chemin de recherche de l’éditeur de liens : un
pilote NVIDIA sur la même machine continue de servir tout le reste, et `vulkanizer-run` ne place Vulkanizer devant que
pour le programme qu’il lance. `cmake --install build --prefix DIR` installe la même arborescence n’importe où, sans paquet.

### 2. Exécuter un programme CUDA

Vulkanizer prend la place des vraies bibliothèques NVIDIA ; il suffit que l’éditeur de liens dynamique les trouve en
premier :

```sh
./run.sh ./my_cuda_app args...                       # run.sh définit LD_LIBRARY_PATH
LD_LIBRARY_PATH=$PWD/build/lib ./my_cuda_app args... # la même chose, à la main
./run.sh --log info ./my_cuda_app                    # affiche le GPU utilisé
./run.sh --device llvmpipe ./my_cuda_app             # un périphérique précis (les CPU demandent VULKANIZER_ALLOW_CPU=1)
./run.sh --check ./my_cuda_app                       # peut-il tourner ? (PTX, runtime CUDA partagé ; aussi un répertoire ou un env Python)
```

Le programme doit remplir deux conditions :

- **Le binaire doit contenir du PTX.** Vulkanizer compile le PTX en SPIR-V ; il ne peut pas utiliser de code machine
  (SASS). Indiquez une architecture virtuelle à la compilation, par ex.
  `nvcc -arch=compute_61 ...` ou dans CMake `CMAKE_CUDA_ARCHITECTURES=61-virtual`.
  61 (Pascal) est un bon choix : elle dispose de `dp4a` et n’a pas de chemins de code pour les tensor cores.
- **Le runtime CUDA doit être lié dynamiquement** (`nvcc -cudart shared`, dans CMake
  `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`), sinon la libcudart statique de NVIDIA se retrouve dans le programme.

Compiler son propre programme avec le nvcc téléchargé (sans GPU ni pilote NVIDIA) :

```sh
tools/setup_cuda.sh                                   # nvcc + en-têtes du runtime dans .cuda/
.cuda/bin/nvcc -ccbin g++-13 -arch=compute_61 -cudart shared vadd.cu -o vadd
./run.sh ./vadd
```

Pour les programmes utilisant cuBLAS, utilisez l’en-tête `include/cublas/cublas_v2.h` de Vulkanizer et
`build/lib/libcublas.so` (`-Iinclude/cublas -Lbuild/lib -lcublas`).

La première exécution est plus lente : c’est là que les kernels sont compilés en SPIR-V et stockés dans le cache sur
disque (`~/.cache/vulkanizer`). Les lancements suivants les chargent depuis le cache.

### 3. llama.cpp

```sh
tools/setup_cuda.sh
./build.sh --release
tools/build_llama.sh                  # clone et compile .work/llama.cpp avec le backend CUDA
./run.sh .work/llama-build/bin/llama-simple -m model.gguf -ngl 99 "Hello"
./run.sh .work/llama-build/bin/llama-bench -m model.gguf -ngl 99
```

Sur les GPU dont le `fma` Vulkan n’est pas fusionné (p. ex. AMD GFX9), Vulkanizer émule le `fma` à arrondi unique de
CUDA pour le code IEEE, mais utilise le `fma` rapide du pilote pour le code compilé en fast math (`-ftz`, comme le
backend CUDA de llama.cpp). `VULKANIZER_FAST_FMA=1` utilise partout le rapide, `VULKANIZER_EXACT_FMA=1` émule partout.

whisper.cpp (reconnaissance vocale, le même backend CUDA de ggml) se compile et s’exécute de la même façon :

```sh
tools/build_whisper.sh   # into .work/whisper-build, with the base.en model
LD_LIBRARY_PATH=build/lib .work/whisper-build/bin/whisper-cli \
    -m .work/whisper.cpp/models/ggml-base.en.bin -f .work/whisper.cpp/samples/jfk.wav
```

### 4. bitnet.c

Le backend CUDA de bitnet.c (`--cuda`) fonctionne sur Vulkan via `run_bitnet.sh` :

```sh
./run_bitnet.sh models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The capital of France is" -n 64
./run_bitnet.sh --profile --log info models/model.gguf -p "Hello"   # temps GPU, journalisation
./run_bitnet.sh --build-only                                        # compilation seule
```

Le script ne modifie pas le checkout (`--bitnet-dir`, par défaut `~/Develop/GitHub/bitnet.c`) : il synchronise les
sources dans `.work/bitnet-vulkanizer` et les y compile en PTX (`compute_61`) avec le cuBLAS et le runtime de
Vulkanizer ; il ajoute automatiquement l’option `--cuda`. Les chemins de modèles relatifs sont aussi cherchés dans le
checkout.

### 5. PyTorch

```sh
tools/setup_torch.sh                       # environnement PyTorch CUDA conda-forge dans .work/torch-cf (~9 Go)
./run_torch.sh -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
./run_torch.sh torch_mlp                   # entraîne un MLP sur GPU et CPU et compare (libtorch, C++)
./run_torch.sh torch_ops                   # conv, softmax, LayerNorm, attention, couche Transformer, ...
./run_torch.sh torch_graph                 # une passe avant capturée dans un graphe CUDA puis rejouée
./run_torch.sh tests/torch/triton_ops.py   # un kernel Triton écrit à la main et le bmm Triton de torch._native
./run_torch.sh tests/torch/compile.py      # torch.compile (Inductor -> Triton), aussi mode="reduce-overhead"

tools/setup_torch.sh --cuda13              # la version CUDA 13.4 dans .work/torch-cf13
./run_torch.sh --env .work/torch-cf13 torch13_ops
```

`torch_mlp`, `torch_ops`, `torch_split` et `torch_graph` sont des tests C++ utilisant la libtorch de l’environnement
(`tests/torch/*.cpp`) ; CMake les compile si `.work/torch-cf` existe au moment de la configuration, et sous le nom
`torch13_*` avec `.work/torch-cf13`. Pour un environnement contenant `libcudart.so.13` (une version CUDA 13),
`run_torch.sh` charge les bibliothèques depuis `build/lib/cuda13`.

Les wheels officiels de PyTorch ne contiennent que du code machine NVIDIA natif (SASS), sans PTX : Vulkanizer ne peut
donc pas exécuter leurs kernels. Les versions CUDA de conda-forge contiennent aussi du PTX (`compute_120`) :
`tools/setup_torch.sh` les installe avec micromamba, sans installation à l’échelle du système. `run_torch.sh` charge
`libcuda`, `libcudart`, `libcublas`, `libcublasLt`, `libcusolver`, `libcufft`, `libcusparse` et `libcurand` de
Vulkanizer à la place de celles de NVIDIA, et désactive tout ce qui exige du matériel ou des bibliothèques NVIDIA :
cuDNN, les kernels d’attention fusionnés (`scaled_dot_product_attention` passe par le chemin mathématique) et les
kernels DSL de `torch._native` (`TORCH_DISABLE_NATIVE_JIT=1`) : ses kernels Triton fonctionnent, mais ses kernels CuTe
DSL se compilent en SASS, et le même interrupteur désactive les deux. `VULKANIZER_TORCH_DEFAULTS=0` désactive ces
réglages. Triton lui-même (`torch.compile`, `@triton.jit`) fonctionne : `tools/triton/ptxas` transmet le PTX de Triton
au lieu de l’assembler, et le lanceur de Triton est compilé avec `include/cuda.h` de Vulkanizer (il faut un compilateur
C). `torch.linalg` (LU, Cholesky, QR, moindres carrés, `eigh`, SVD, `pinv`, ...) tourne sur le cuSOLVER de Vulkanizer et
`torch.fft` (y compris `stft`) sur son cuFFT, `torch.sparse` sur son cuSPARSE ; `run_torch.sh` définit
`TORCH_LINALG_PREFER_CUSOLVER=1`, car les kernels propres à MAGMA ne tournent pas. Le jiterator de PyTorch compile ses
kernels à l’exécution avec NVRTC en code natif ; une bibliothèque `LD_AUDIT` (`libvulkanizer_nvrtc.so`) lui fait
transmettre le PTX à la place.

### 6. ComfyUI

ComfyUI passe aussi par `run_torch.sh`, dans son propre environnement conda-forge à l’intérieur du checkout de ComfyUI :

```sh
VK=~/Develop/GitHub/Vulkanizer             # ce dépôt, compilé avec ./build.sh --release
cd ~/Develop/GitHub                        # là où ComfyUI doit être installé
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI

# PyTorch (version CUDA 12.9 avec PTX) et les dépendances natives de torchvision
export MAMBA_ROOT_PREFIX=$VK/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
[ -x $VK/.work/micromamba/bin/micromamba ] || { mkdir -p $VK/.work/micromamba && curl -fsSL \
    https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $VK/.work/micromamba bin/micromamba; }
$VK/.work/micromamba/bin/micromamba create -y -p $PWD/venv-vulkanizer -c conda-forge \
    python=3.13 "pytorch=2.14.1=cuda129_generic*" pip pillow giflib libwebp-base libjpeg-turbo libpng \
    "libnvjpeg>=12.4,<13" "libcusolver<12" "libcusparse<13"
# torchvision par URL, sans résolution : sa version CUDA 12.9 demande un cuDNN plus récent qui exige le paquet
# virtuel __cuda_arch (un vrai pilote). De toute façon, cuDNN est désactivé sous Vulkanizer.
$VK/.work/micromamba/bin/micromamba install -y -p $PWD/venv-vulkanizer --file <(printf '@EXPLICIT\n%s\n' \
    https://conda.anaconda.org/conda-forge/linux-64/torchvision-0.29.1-cuda129_py313_hcd9f930_0.conda)

# le reste des dépendances de ComfyUI ; pip ne doit pas remplacer torch/torchvision par les wheels SASS uniquement
venv-vulkanizer/bin/pip install -r <(grep -vE '^(torch|torchvision)$' requirements.txt) -r manager_requirements.txt

# démarrage
$VK/run_torch.sh --env venv-vulkanizer main.py --enable-manager --disable-cuda-malloc
```

Ouvrez ensuite <http://127.0.0.1:8188>. Les modèles vont dans `models/`, comme d’habitude. Remarques :

- `--disable-cuda-malloc` : le backend d’allocation `cudaMallocAsync` de ComfyUI n’est pas utilisé.
- DynamicVRAM (comfy-aimdo) fonctionne sur la gestion de mémoire virtuelle de Vulkanizer (VMM, buffers sparse). En cas
  de problème, `--disable-dynamic-vram` revient au chargement des modèles fondé sur des estimations.
- `tools/torch/sitecustomize.py` n’applique les réglages PyTorch qu’au moment où ComfyUI importe torch, et maintient
  désactivés les backends d’attention fusionnés même si ComfyUI les réactive : l’attention passe par le chemin
  mathématique.
- Le message « PyTorch is not installed » de ComfyUI-Manager vient de l’absence de `torchaudio` (conda-forge n’a pas de
  version pour torch 2.14) ; la génération d’images n’en a pas besoin.
- La première génération est lente, le temps de compiler les kernels en SPIR-V ; les suivantes les chargent depuis le cache.
- `$VK/run_torch.sh --log info --env venv-vulkanizer main.py ...` indique le GPU Vulkan utilisé.
- `python3 tests/torch/zimage_forward.py` compare le GPU au CPU : une passe avant de Z-Image Turbo en 256×256, en fp16
  sur le GPU et en fp32 et fp16 sur le CPU. Le GPU doit s’écarter du fp32 autant que le fp16 lui-même, pas plus
  (ComfyUI exécute le modèle en fp16 sur le GPU, si bien que les images complètes diffèrent d’une exécution fp32 sur CPU
  dans les détails fins).

### 7. Numba

Numba (`numba-cuda`, `@cuda.jit`) compile les kernels avec libNVVM en PTX, que Vulkanizer exécute (`CUDA_HOME` lui
permet de trouver libNVVM dans un environnement conda) :

```sh
export MAMBA_ROOT_PREFIX=$PWD/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
.work/micromamba/bin/micromamba create -y -p .work/numba-cf -c conda-forge \
    python=3.13 numba numba-cuda "cuda-version=12.9" cuda-nvvm cuda-nvrtc numpy
CUDA_HOME=$PWD/.work/numba-cf ./run_torch.sh --env .work/numba-cf script.py
```

Son éditeur de liens est nvJitLink, qui produirait du code natif : sous `run_torch.sh` (et `vulkanizer-run --torch`),
les appels d’édition de liens de nvJitLink vont au `cuLink*` du pilote, dont l’image liée est ici du PTX, et nvJitLink
ne rapporte aucune version, si bien que Numba produit du PTX plutôt que du LTO-IR. CuPy ne fonctionne pas : il lie le
runtime CUDA statiquement (voir Limitations).

### Variables d’environnement utiles

| Variable | Effet |
|---|---|
| `VULKANIZER_LOG=info\|debug\|trace` | Journal sur stderr ; `trace` affiche chaque appel d’API |
| `VULKANIZER_DEVICE=<nom>` | Choisit un périphérique Vulkan par fragment de nom (`radv`, `llvmpipe`, ...) |
| `VULKANIZER_ALLOW_CPU=1` | Autorise les périphériques Vulkan CPU (llvmpipe) |
| `VULKANIZER_PROFILE=1` | À la fin, temps GPU par kernel et temps hôte par appel d’API |
| `VULKANIZER_BATCH_MS=<ms>` | Budget de temps GPU d’une soumission à la file (50 par défaut ; `0` = sans limite). Les kernels plus longs sont découpés, pour ne pas déclencher le délai d’attente GPU du pilote noyau (amdgpu : 2 s) |
| `CUDA_LAUNCH_BLOCKING=1` | Chaque lancement de kernel attend le GPU, si bien que les erreurs et blocages apparaissent au lancement fautif |
| `VULKANIZER_FAST_FMA=1` | `fma` f32 plus rapide mais moins précis |
| `VULKANIZER_EXACT_FMA=1` | `fma` f32 exact aussi pour le code fast math (`.ftz`), plus lent |
| `VULKANIZER_GEMM_F32=1` | Les GEMM cuBLAS en `COMPUTE_16F` multiplient aussi en f32 |
| `VULKANIZER_IPC=0` | La mémoire du périphérique n’est pas exportable vers d’autres processus (pas d’IPC CUDA) |
| `VULKANIZER_CACHE_DIR=<rép>\|off` | Emplacement du cache de kernels, ou sa désactivation |
| `VULKANIZER_CACHE_SIZE=<MiB>` | Taille maximale du cache de kernels en Mio (256 par défaut) ; les entrées les moins récemment utilisées partent d’abord |
| `VULKANIZER_CC=86` | La compute capability annoncée |
| `VULKANIZER_DUMP=<rép>` | Écrit le SPIR-V généré (`<kernel>.spv`) |

Liste complète : [plan.md, section 4](plan.md).

### Limitations

- Pas d’exécution de SASS : seuls les binaires contenant du PTX fonctionnent.
- Les programmes et bibliothèques qui lient le runtime CUDA statiquement (CuPy, le NCCL de NVIDIA) ne fonctionnent
  pas : ce runtime a besoin d’éléments internes non documentés du pilote NVIDIA.
- CUDA Graphs : la capture de stream et la relecture fonctionnent (les graphes de llama.cpp, `CUDAGraph` de PyTorch,
  `torch.compile(mode="reduce-overhead")`) ; aussi les graphes enfants et les nœuds
  mémoire (allocation ordonnée par flux pendant la capture) et les nœuds conditionnels (if, while, switch ; à chacun,
  l’hôte attend le GPU).
- Pas (encore) pris en charge : cuDNN.
- Les textures et surfaces (objets texture et surface, tableaux CUDA) sont échantillonnées en logiciel, avec
  l’adressage et le filtrage de CUDA ; pas de mipmaps, de cubemaps, de références de texture ni d’écritures de surface
  formatées.
- L’IPC CUDA (`cudaIpc*`, `torch.multiprocessing` de PyTorch) fonctionne entre processus du même utilisateur ; un
  événement interprocessus est terminé dès son enregistrement (l’enregistrement attend le périphérique).
- NCCL (`libnccl.so.2`, le backend « nccl » de `torch.distributed` : DDP, FSDP) fonctionne entre les rangs d’une
  machine, qui peuvent partager un GPU ; avec plus d’un rang, les données passent par la mémoire de l’hôte.
- Plusieurs GPU sont des périphériques distincts : `cudaMemcpyPeer` et les copies ordinaires entre pointeurs de deux
  périphériques (adressage unifié, p. ex. `tensor.to("cuda:1")` de PyTorch) passent par la mémoire hôte, mais un
  kernel n’accède jamais à la mémoire d’un autre GPU (pas d’accès pair). Les lancements coopératifs (`grid.sync()`)
  fonctionnent jusqu’au nombre de blocs qui tiennent simultanément sur le GPU.
- La mémoire managed, épinglée et enregistrée est de la mémoire hôte que le GPU utilise sur place, à la même adresse ;
  la mémoire managed ne migre jamais vers le GPU (sur un GPU intégré, c’est de toute façon la même mémoire).
- Tant que de la mémoire hôte enregistrée (`cuMemHostRegister`) existe, chaque soumission au GPU est plus lente (le noyau
  revalide les pages importées, ~4 ms pour 256 Mo) ; la mémoire épinglée et managed n’est pas concernée.
- La flash attention de llama.cpp fonctionne correctement (`tools/build_llama.sh --fa`), mais sans tensor cores le
  traitement du prompt est plus lent avec elle ; elle est donc désactivée par défaut.
- PyTorch : pas de kernels cuDNN ni d’attention fusionnée.
- cuSOLVER : types réels (f32, f64) uniquement, pas de `gesvda`, de LDL^T (`sytrf`) ni de problèmes aux valeurs
  propres non symétriques (`geev`) ; `eigh` et la SVD utilisent des itérations de Jacobi, lentes pour les grandes
  matrices (n ≳ 1000). cuFFT : pas de callbacks ni de plans multi-GPU ; les transformées de plus de 2048 points (f64 :
  1024) font un passage en mémoire par radix. cuSPARSE : pas de matrices creuses par blocs (BSR) ni de valeurs
  complexes ; SpGEMM et les routines de tri tournent sur l’hôte.
- cuRAND : XORWOW, MRG32k3a, Philox et Sobol donnent les suites de NVIDIA (lois normale et log-normale à l’arrondi
  près) ; MTGP32 et MT19937 sont acceptés mais donnent d’autres suites, de même que Sobol brouillé et Poisson ; les
  sorties en double exigent `shaderFloat64`. L’API device (`curand_kernel.h`) fonctionne, sauf les distributions
  créées sur l’hôte (`curand_discrete`).
- ComfyUI : pas de chemins optimisés pour CUDA 13 (cu130).

## Scripts

| Commande | Rôle |
|---|---|
| `./build.sh` | Configure (si nécessaire) et compile. `--release`, `--debug`, `--clean`, `--asan`/`--no-asan` |
| `./test.sh` | Compilation + tous les tests. `--device llvmpipe`, `--all-devices`, `--log debug`, `-R <regex>` |
| `./run.sh <programme>` | Exécute un programme avec la `libcuda.so` de Vulkanizer. `--log trace`, `--device`, `--gdb`, `--validation` |
| `./clean.sh` | Supprime le répertoire `build/` |
| `./ci.sh` | Vérification complète : Debug+ASAN et Release, sur chaque périphérique (compile dans `build-ci/`) |
| `tools/package.sh` | Paquets de version : un `.deb` et un `.tar.gz` dans `build-pkg/` (`--check` essaie aussi le `.deb`) |
| `build/bin/vulkanizer-info` | Liste les périphériques Vulkan que Vulkanizer voit |
| `build/bin/vulkanizer-ptxc` | PTX → SPIR-V hors ligne : `vulkanizer-ptxc k.ptx --list`, `-e vecAdd -o k.spv --dis --val` |
| `build/bin/vulkanizer-bandwidth` | Débit des copies (HtoD/DtoH/DtoD, mémoire paginable et épinglée) et coût des allocations : `vulkanizer-bandwidth [MiB]` |

Exemples :

```sh
./test.sh --all-devices            # RADV + llvmpipe
./run.sh --log trace device_query  # chaque appel d’API est journalisé
./run.sh ./some_cuda_app           # programme CUDA externe (compilé avec -cudart shared)
```

## Organisation des répertoires

```
include/cuda.h        en-tête clean-room de l’API CUDA Driver (sous-ensemble)
src/common            journalisation
src/vk                instance/périphérique Vulkan, buffers, pipelines
src/ptx               lexer PTX, analyseur de modules et d’instructions
src/ptx/texture_lib.cu  échantillonnage logiciel des textures et surfaces (intégré en PTX, tools/gen_texture_lib.sh)
src/compiler          CFG + structuration, générateur SPIR-V, compilateur PTX -> SPIR-V
src/driver            la surface d’API de libcuda.so
src/runtime           libcudart.so.12 et .so.13 (API CUDA Runtime sur l’API Driver, une seule source)
src/cublas            libcublas.so.12/.13 (GEMM/TRSM avec des kernels GLSL), include/cublas/cublas_v2.h
src/linalg            factorisations denses (LU, Cholesky, QR, eigh/SVD de Jacobi) en GLSL, communes à cuBLAS et cuSOLVER
src/cusolver          libcusolver.so.11/.12, include/cusolver/cusolverDn.h
src/cufft             libcufft.so.11/.12 (FFT de Stockham et de Bluestein en GLSL), include/cufft/cufft.h
src/cusparse          libcusparse.so.12 (SpMV/SpMM/SDDMM/SpSV/csrgeam2 en GLSL), include/cusparse/cusparse.h
src/curand            libcurand.so.10 (XORWOW/MRG32k3a/Philox/Sobol en GLSL ; données Sobol : tools/gen_sobol_table.sh), include/curand/curand.h
src/nccl              libnccl.so.2 (collectives entre les rangs d’une machine), include/nccl/nccl.h
src/nvrtc             bibliothèque LD_AUDIT : NVRTC transmet le PTX au lieu d’un cubin (jiterator de PyTorch), nvJitLink lie via cuLink* (Numba)
tests/                tests de l’API Driver + test Vulkan interne
tests/cuda            programmes de l’API Runtime compilés avec nvcc
tests/cudart_abi      conformité ABI des types clean-room du runtime avec les en-têtes officiels
tests/data            fichiers PTX de test avec leurs sources .cu ; gen_ptx.sh les régénère (clang)
tools/                vulkanizer-info (liste des périphériques), vulkanizer-ptxc (compilateur hors ligne), vulkanizer-bandwidth, setup_cuda.sh,
                      build_llama.sh (compilation CUDA de llama.cpp avec Vulkanizer)
                      build_whisper.sh (whisper.cpp)
```

## Licence et mentions légales

Vulkanizer est distribué sous [licence MIT](LICENSE). Les éléments tiers qu’il inclut (les nombres directeurs de Sobol
de Joe et Kuo, licence BSD) sont listés dans [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Vulkanizer ne contient aucun code source, en-tête ni binaire de NVIDIA : ses en-têtes CUDA sont des déclarations
clean-room rédigées à partir de la documentation publique et ne reprennent que ce qu’exige la compatibilité binaire. Le
toolkit CUDA téléchargé par `tools/setup_cuda.sh` provient des serveurs de NVIDIA, sous la propre licence de NVIDIA, et
ne fait pas partie de Vulkanizer.

NVIDIA, CUDA, cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NCCL et PTX sont des marques ou des marques déposées de NVIDIA
Corporation ; Vulkan et SPIR-V sont des marques déposées de The Khronos Group Inc. Elles ne servent qu’à indiquer ce
avec quoi Vulkanizer est compatible. Vulkanizer est un projet indépendant, sans lien avec NVIDIA ni le Khronos Group, et
n’est ni approuvé ni parrainé par eux.
