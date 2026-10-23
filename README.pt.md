![Vulkanizer – traz a paz à galáxia. Uma vulcana diante de um computador que executa um LLM, no estilo dos anos 1960.](assets/vulkanizer-hero.png)

[English](README.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Español](README.es.md) | **Português** | [Magyar](README.hu.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# Data estelar 2026.10.21 #
A Galáxia da Inteligência Artificial está à beira do colapso arquitetural total.
Há ciclos, o cosmos está preso numa guerra de otimização brutal. Vindos dos setores de código aberto, os ferozes Rocmulanos Vermelhos chocam-se incessantemente com as frotas dominantes e proprietárias dos Klincudas Verdes, cada lado desesperado por conquistar a supremacia absoluta de execução sobre as redes neurais.
Mas, enquanto estes dois titãs queimam poder de processamento, uma nova e aterradora ameaça surge na orla galáctica: o Coletivo Intel-igente Azul. Reunindo força multiplataforma com suas diretivas XPU onívoras, prepara-se para assimilar todas as cargas de trabalho e conquistar seu próprio e enorme pedaço do universo digital.
Nesta era de caos sem precedentes, o destino do cosmos está por um fio. A devastadora colisão tripla de pipelines só pode ser evitada por uma figura lendária: o Comandante Vulkanizer. Armado com lógica absoluta e uma eficiência de compilação inigualável, ele precisa partir numa missão final e lógica para vulcanizar a galáxia inteira — unificar os ambientes de execução fragmentados sob um único padrão multiplataforma antes que o universo fique inutilizado para sempre.

# Vulkanizer

Substituto direto de `libcuda.so`, `libcudart.so`, `libcublas.so`, `libcusolver.so`, `libcufft.so` e `libcusparse.so`:
implementa as APIs CUDA Driver e Runtime (e a família GEMM do cuBLAS, a álgebra linear densa e esparsa e as FFTs)
sobre Vulkan Compute, para que programas CUDA rodem sem placa NVIDIA, em qualquer GPU com Vulkan 1.2 (AMD, Intel,
llvmpipe).
Plano, arquitetura e fases: [plan.md](plan.md) (em húngaro).

## Requisitos

CMake ≥ 3.20, um compilador C++20, um driver Vulkan 1.2+ (bufferDeviceAddress + shaderInt64), `glslc` para os testes.

Opcional (nomes de pacotes do Ubuntu):

| Pacote | Para quê |
|---|---|
| `spirv-tools` | `spirv-val` / `spirv-dis` – validar o SPIR-V gerado (`vulkanizer-ptxc --val --dis`) |
| `vulkan-validationlayers` | `./run.sh --validation` |
| `clang` | Gerar o PTX de teste sem o SDK do CUDA (`tests/data/gen_ptx.sh`) |
| `ninja-build` | Compilações mais rápidas (`build.sh` o usa automaticamente se estiver presente) |
| `g++-13` | Compilador host do nvcc (o CUDA 12.9 não aceita o gcc 15) |

Testes compilados com um nvcc real: `tools/setup_cuda.sh` baixa o nvcc e o runtime da NVIDIA, como pacotes
redistribuíveis, para `.cuda/` (sem driver, sem GPU, sem root). Se `.cuda/` existir, o CMake compila os programas
`tests/cuda/*.cu` com o nvcc e o ctest os executa com as bibliotecas do Vulkanizer.
`tools/setup_cuda.sh --cuda13` faz o mesmo com o CUDA 13.4 em `.cuda13/`: os programas são recompilados com o nvcc 13
e rodam sobre `libcudart.so.13` (testes terminados em `_13`).

## Uso

### 1. Compilação

```sh
./build.sh --release                       # compilação Release em build/
VULKANIZER_BUILD_DIR=$PWD/build-rel ./build.sh --release   # ou num diretório separado
build/bin/vulkanizer-info                       # quais dispositivos Vulkan o Vulkanizer enxerga
./test.sh                                  # testes (opcional)
```

As bibliotecas compiladas ficam em `build/lib/`: `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`,
`libcublasLt.so.12`, `libcusolver.so.11`, `libcufft.so.11`, `libcusparse.so.12`, `libnccl.so.2`. `build/lib/cuda13/`
contém as versões CUDA 13 do runtime e das bibliotecas (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`,
`libcusolver.so.12`, `libcufft.so.12`, `libcusparse.so.12`), para programas compilados com CUDA 13; elas usam a mesma
`libcuda.so.1`.

#### Instalação (pacote)

```sh
tools/package.sh                               # build-pkg/vulkanizer_<versão>_<arq>.deb e um .tar.gz
sudo apt install ./build-pkg/vulkanizer_*.deb
vulkanizer-run ./my_cuda_app args...           # qualquer programa CUDA 12 ou 13
vulkanizer-run --torch python script.py        # PyTorch do pip ou conda (CUDA 12 ou 13, detectado)
```

As bibliotecas vão para `/usr/lib/<arq>/vulkanizer/`, fora do caminho de busca do linker: um driver NVIDIA na mesma
máquina continua funcionando para todo o resto, e o `vulkanizer-run` coloca o Vulkanizer à frente só para o programa
que inicia. `cmake --install build --prefix DIR` instala a mesma estrutura em qualquer lugar, sem pacote.

### 2. Executar um programa CUDA

O Vulkanizer ocupa o lugar das bibliotecas reais da NVIDIA; basta que o linker dinâmico as encontre primeiro:

```sh
./run.sh ./my_cuda_app args...                       # o run.sh define LD_LIBRARY_PATH
LD_LIBRARY_PATH=$PWD/build/lib ./my_cuda_app args... # o mesmo, à mão
./run.sh --log info ./my_cuda_app                    # mostra em qual GPU roda
./run.sh --device llvmpipe ./my_cuda_app             # um dispositivo específico (os de CPU exigem VULKANIZER_ALLOW_CPU=1)
```

O programa precisa atender a dois requisitos:

- **O binário precisa conter PTX.** O Vulkanizer compila PTX para SPIR-V; não consegue usar código de máquina (SASS).
  Indique uma arquitetura virtual ao compilar, p. ex.
  `nvcc -arch=compute_61 ...` ou no CMake `CMAKE_CUDA_ARCHITECTURES=61-virtual`.
  61 (Pascal) é uma boa escolha: tem `dp4a` e não há caminhos de código para tensor cores.
- **O runtime do CUDA precisa ser vinculado dinamicamente** (`nvcc -cudart shared`, no CMake
  `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`); caso contrário, a libcudart estática da NVIDIA acaba dentro do programa.

Compilar um programa próprio com o nvcc baixado (sem GPU nem driver da NVIDIA):

```sh
tools/setup_cuda.sh                                   # nvcc + cabeçalhos do runtime em .cuda/
.cuda/bin/nvcc -ccbin g++-13 -arch=compute_61 -cudart shared vadd.cu -o vadd
./run.sh ./vadd
```

Para programas que usam cuBLAS, use o cabeçalho `include/cublas/cublas_v2.h` do Vulkanizer e `build/lib/libcublas.so`
(`-Iinclude/cublas -Lbuild/lib -lcublas`).

A primeira execução é mais lenta: nela os kernels são compilados para SPIR-V e guardados no cache em disco
(`~/.cache/vulkanizer`). As execuções seguintes carregam do cache.

### 3. llama.cpp

```sh
tools/setup_cuda.sh
./build.sh --release
tools/build_llama.sh                  # clona e compila .work/llama.cpp com o backend CUDA
./run.sh .work/llama-build/bin/llama-simple -m model.gguf -ngl 99 "Hello"
./run.sh .work/llama-build/bin/llama-bench -m model.gguf -ngl 99
```

Em GPUs cujo `fma` do Vulkan não é fundido (p. ex. AMD GFX9), o Vulkanizer emula o `fma` de arredondamento único do
CUDA para código IEEE, mas usa o `fma` rápido do driver para código compilado com fast math (`-ftz`, como o backend CUDA
do llama.cpp). `VULKANIZER_FAST_FMA=1` usa o rápido em tudo, `VULKANIZER_EXACT_FMA=1` emula em tudo.

### 4. bitnet.c

O backend CUDA do bitnet.c (`--cuda`) roda sobre Vulkan via `run_bitnet.sh`:

```sh
./run_bitnet.sh models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The capital of France is" -n 64
./run_bitnet.sh --profile --log info models/model.gguf -p "Hello"   # tempos de GPU, log
./run_bitnet.sh --build-only                                        # só compilar
```

O script não altera o checkout (`--bitnet-dir`, padrão `~/Develop/GitHub/bitnet.c`): sincroniza os fontes em
`.work/bitnet-vulkanizer` e os compila ali para PTX (`compute_61`) contra o cuBLAS e o runtime do Vulkanizer; adiciona
automaticamente a opção `--cuda`. Caminhos relativos de modelos também são procurados no checkout.

### 5. PyTorch

```sh
tools/setup_torch.sh                       # ambiente PyTorch CUDA do conda-forge em .work/torch-cf (~9 GB)
./run_torch.sh -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
./run_torch.sh torch_mlp                   # treina um MLP na GPU e na CPU e compara (libtorch, C++)
./run_torch.sh torch_ops                   # conv, softmax, LayerNorm, attention, camada Transformer, ...
./run_torch.sh torch_graph                 # uma passada direta capturada num grafo CUDA e reproduzida
./run_torch.sh tests/torch/triton_ops.py   # um kernel Triton escrito à mão e o bmm Triton do torch._native
./run_torch.sh tests/torch/compile.py      # torch.compile (Inductor -> Triton), também mode="reduce-overhead"

tools/setup_torch.sh --cuda13              # a versão CUDA 13.4 em .work/torch-cf13
./run_torch.sh --env .work/torch-cf13 torch13_ops
```

`torch_mlp`, `torch_ops`, `torch_split` e `torch_graph` são testes em C++ contra a libtorch do ambiente
(`tests/torch/*.cpp`); o CMake os compila se `.work/torch-cf` existir na configuração, e como `torch13_*` contra
`.work/torch-cf13`. Para um ambiente com `libcudart.so.13` (uma versão CUDA 13), o `run_torch.sh` carrega as
bibliotecas de `build/lib/cuda13`.

Os wheels oficiais do PyTorch contêm apenas código de máquina nativo da NVIDIA (SASS), sem PTX, então o Vulkanizer não
consegue executar seus kernels. As versões CUDA do conda-forge trazem também PTX (`compute_120`):
`tools/setup_torch.sh` as instala com o micromamba, sem instalação no sistema. O `run_torch.sh` carrega `libcuda`,
`libcudart`, `libcublas`, `libcublasLt`, `libcusolver`, `libcufft` e `libcusparse` do Vulkanizer no lugar das da
NVIDIA e desativa tudo o que exige hardware ou bibliotecas da NVIDIA: o cuDNN, os kernels de atenção fundidos
(`scaled_dot_product_attention` usa o caminho matemático) e os kernels DSL do `torch._native`
(`TORCH_DISABLE_NATIVE_JIT=1`): seus kernels Triton funcionam, mas os kernels CuTe DSL são compilados para SASS, e o
mesmo interruptor desliga os dois. `VULKANIZER_TORCH_DEFAULTS=0` pula tudo isso. O próprio Triton (`torch.compile`,
`@triton.jit`) funciona: `tools/triton/ptxas` repassa o PTX do Triton em vez de montá-lo, e o lançador do Triton é
compilado contra o `include/cuda.h` do Vulkanizer (requer um compilador C). `torch.linalg` (LU, Cholesky, QR, mínimos
quadrados, `eigh`, SVD, `pinv`, ...) roda sobre o cuSOLVER do Vulkanizer e `torch.fft` (inclusive `stft`) sobre o seu
cuFFT e `torch.sparse` sobre o seu cuSPARSE; o `run_torch.sh` define `TORCH_LINALG_PREFER_CUSOLVER=1`, pois os kernels
próprios do MAGMA não rodam. O jiterator do PyTorch compila seus kernels em tempo de execução com o NVRTC para código
nativo; uma biblioteca `LD_AUDIT` (`libvulkanizer_nvrtc.so`) faz com que ele entregue o PTX no lugar.

### 6. ComfyUI

O ComfyUI também roda via `run_torch.sh`, num ambiente próprio do conda-forge dentro do checkout do ComfyUI:

```sh
VK=~/Develop/GitHub/Vulkanizer             # este repositório, compilado com ./build.sh --release
cd ~/Develop/GitHub                        # onde o ComfyUI deve ficar
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI

# PyTorch (versão CUDA 12.9 com PTX) e as dependências nativas do torchvision
export MAMBA_ROOT_PREFIX=$VK/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
[ -x $VK/.work/micromamba/bin/micromamba ] || { mkdir -p $VK/.work/micromamba && curl -fsSL \
    https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $VK/.work/micromamba bin/micromamba; }
$VK/.work/micromamba/bin/micromamba create -y -p $PWD/venv-vulkanizer -c conda-forge \
    python=3.13 "pytorch=2.14.1=cuda129_generic*" pip pillow giflib libwebp-base libjpeg-turbo libpng \
    "libnvjpeg>=12.4,<13" "libcusolver<12" "libcusparse<13"
# torchvision por URL, sem resolução: sua versão CUDA 12.9 pede um cuDNN mais novo que exige o pacote virtual
# __cuda_arch (um driver real). No Vulkanizer, o cuDNN fica desativado de qualquer forma.
$VK/.work/micromamba/bin/micromamba install -y -p $PWD/venv-vulkanizer --file <(printf '@EXPLICIT\n%s\n' \
    https://conda.anaconda.org/conda-forge/linux-64/torchvision-0.29.1-cuda129_py313_hcd9f930_0.conda)

# o restante dos requisitos do ComfyUI; o pip não deve trocar torch/torchvision pelos wheels só com SASS
venv-vulkanizer/bin/pip install -r <(grep -vE '^(torch|torchvision)$' requirements.txt) -r manager_requirements.txt

# iniciar
$VK/run_torch.sh --env venv-vulkanizer main.py --enable-manager --disable-cuda-malloc
```

Depois abra <http://127.0.0.1:8188>. Os modelos vão em `models/`, como de costume. Observações:

- `--disable-cuda-malloc`: o backend de alocação `cudaMallocAsync` do ComfyUI não é usado.
- O DynamicVRAM (comfy-aimdo) roda sobre o gerenciamento de memória virtual do Vulkanizer (VMM, buffers sparse). Se
  causar problemas, `--disable-dynamic-vram` volta ao carregamento de modelos baseado em estimativas.
- `tools/torch/sitecustomize.py` aplica as configurações do PyTorch só quando o ComfyUI importa o torch, e mantém
  desligados os backends de atenção fundidos mesmo que o ComfyUI os religue: a atenção usa o caminho matemático.
- A mensagem "PyTorch is not installed" do ComfyUI-Manager vem da falta do `torchaudio` (o conda-forge não tem versão
  para o torch 2.14); a geração de imagens não precisa dele.
- A primeira geração é lenta enquanto os kernels são compilados para SPIR-V; as seguintes os carregam do cache.
- `$VK/run_torch.sh --log info --env venv-vulkanizer main.py ...` mostra qual GPU Vulkan é usada.
- `python3 tests/torch/zimage_forward.py` compara a GPU com a CPU: uma passada direta do Z-Image Turbo em 256×256, em
  fp16 na GPU e em fp32 e fp16 na CPU. A GPU não deve se afastar do fp32 mais do que o próprio fp16 (o ComfyUI executa o
  modelo em fp16 na GPU, por isso as imagens completas diferem de uma execução fp32 na CPU nos detalhes finos).

### Variáveis de ambiente úteis

| Variável | Efeito |
|---|---|
| `VULKANIZER_LOG=info\|debug\|trace` | Log em stderr; `trace` mostra cada chamada de API |
| `VULKANIZER_DEVICE=<nome>` | Escolhe um dispositivo Vulkan por trecho do nome (`radv`, `llvmpipe`, ...) |
| `VULKANIZER_ALLOW_CPU=1` | Permite dispositivos Vulkan de CPU (llvmpipe) |
| `VULKANIZER_PROFILE=1` | Ao sair, tempo de GPU por kernel e tempo de host por chamada de API |
| `VULKANIZER_BATCH_MS=<ms>` | Orçamento de tempo de GPU de um envio à fila (padrão 50; `0` = sem limite). Kernels mais longos são divididos para não atingir o timeout de GPU do driver do kernel (amdgpu: 2 s) |
| `CUDA_LAUNCH_BLOCKING=1` | Cada lançamento de kernel espera a GPU, assim erros e travamentos aparecem no lançamento que os causou |
| `VULKANIZER_FAST_FMA=1` | `fma` f32 mais rápido e menos preciso |
| `VULKANIZER_EXACT_FMA=1` | `fma` f32 exato também para código fast math (`.ftz`), mais lento |
| `VULKANIZER_GEMM_F32=1` | GEMMs do cuBLAS com `COMPUTE_16F` também multiplicam em f32 |
| `VULKANIZER_IPC=0` | A memória do dispositivo não é exportável para outros processos (sem IPC do CUDA) |
| `VULKANIZER_CACHE_DIR=<dir>\|off` | Local do cache de kernels, ou desativá-lo |
| `VULKANIZER_CACHE_SIZE=<MiB>` | Limite de tamanho do cache de kernels em MiB (padrão 256); as entradas usadas há mais tempo saem primeiro |
| `VULKANIZER_CC=86` | A compute capability informada |
| `VULKANIZER_DUMP=<dir>` | Grava o SPIR-V gerado (`<kernel>.spv`) |

Lista completa: [plan.md, seção 4](plan.md).

### Limitações

- Sem execução de SASS: só rodam binários que contêm PTX.
- CUDA Graphs: a captura de streams e a reprodução funcionam (os grafos do llama.cpp, `CUDAGraph` do PyTorch,
  `torch.compile(mode="reduce-overhead")`); sem grafos filhos, nós de memória ou nós condicionais.
- (Ainda) não suportado: texturas, cuDNN.
- O IPC do CUDA (`cudaIpc*`, `torch.multiprocessing` do PyTorch) funciona entre processos do mesmo usuário; um evento
  entre processos está concluído assim que é registrado (registrá-lo espera pelo dispositivo).
- NCCL (`libnccl.so.2`, o backend "nccl" do `torch.distributed`: DDP, FSDP) funciona entre os ranks de uma máquina,
  que podem compartilhar uma GPU; com mais de um rank os dados passam pela memória do host.
- Várias GPUs são dispositivos separados: `cudaMemcpyPeer` e as cópias comuns entre ponteiros de dois dispositivos
  (endereçamento unificado, p. ex. `tensor.to("cuda:1")` do PyTorch) passam pela memória do host, mas um kernel nunca
  acessa a memória de outra GPU (sem acesso entre pares). Lançamentos cooperativos (`grid.sync()`) funcionam até o
  número de blocos que cabem ao mesmo tempo na GPU.
- A memória managed, fixada (pinned) e registrada é memória do host que a GPU usa no lugar, no mesmo endereço;
  a memória managed nunca migra para a GPU (numa GPU integrada é a mesma memória de qualquer forma).
- Enquanto existir memória do host registrada (`cuMemHostRegister`), cada submissão à GPU fica mais lenta (o kernel
  revalida as páginas importadas, ~4 ms por 256 MB); a memória fixada e a managed não são afetadas.
- A flash attention do llama.cpp funciona corretamente (`tools/build_llama.sh --fa`), mas sem tensor cores o
  processamento do prompt fica mais lento com ela, por isso vem desativada.
- PyTorch: sem kernels de cuDNN ou atenção fundida; um `assert` que falha no dispositivo só para a thread
  (não há `cudaErrorAssert`).
- cuSOLVER: apenas tipos reais (f32, f64), sem `gesvda`, LDL^T (`sytrf`) nem problemas de autovalores não simétricos
  (`geev`); `eigh` e a SVD usam iterações de Jacobi, lentas para matrizes grandes (n ≳ 1000). cuFFT: sem callbacks nem
  planos multi-GPU; transformadas com mais de 2048 pontos (f64: 1024) fazem uma passada pela memória por radix.
  cuSPARSE: sem matrizes esparsas por blocos (BSR) nem valores complexos; o SpGEMM e as rotinas de ordenação rodam no
  host.
- ComfyUI: sem os caminhos otimizados para CUDA 13 (cu130).

## Scripts

| Comando | O que faz |
|---|---|
| `./build.sh` | Configura (se necessário) e compila. `--release`, `--debug`, `--clean`, `--asan`/`--no-asan` |
| `./test.sh` | Compilação + todos os testes. `--device llvmpipe`, `--all-devices`, `--log debug`, `-R <regex>` |
| `./run.sh <programa>` | Executa um programa com a `libcuda.so` do Vulkanizer. `--log trace`, `--device`, `--gdb`, `--validation` |
| `./clean.sh` | Apaga o diretório `build/` |
| `./ci.sh` | Verificação completa: Debug+ASAN e Release, em todos os dispositivos (compila em `build-ci/`) |
| `tools/package.sh` | Pacotes de versão: um `.deb` e um `.tar.gz` em `build-pkg/` (`--check` também testa o `.deb`) |
| `build/bin/vulkanizer-info` | Lista os dispositivos Vulkan que o Vulkanizer enxerga |
| `build/bin/vulkanizer-ptxc` | PTX → SPIR-V offline: `vulkanizer-ptxc k.ptx --list`, `-e vecAdd -o k.spv --dis --val` |
| `build/bin/vulkanizer-bandwidth` | Largura de banda de cópia (HtoD/DtoH/DtoD, paginável e fixada) e custo de alocação: `vulkanizer-bandwidth [MiB]` |

Exemplos:

```sh
./test.sh --all-devices            # RADV + llvmpipe
./run.sh --log trace device_query  # cada chamada de API vai para o log
./run.sh ./some_cuda_app           # programa CUDA externo (compilado com -cudart shared)
```

## Estrutura de diretórios

```
include/cuda.h        cabeçalho clean-room da API CUDA Driver (subconjunto)
src/common            log
src/vk                instância/dispositivo Vulkan, buffers, pipelines
src/ptx               lexer de PTX, analisador de módulos e instruções
src/compiler          CFG + estruturação, gerador de SPIR-V, compilador PTX -> SPIR-V
src/driver            a superfície de API da libcuda.so
src/runtime           libcudart.so.12 e .so.13 (API CUDA Runtime sobre a API Driver, um só fonte)
src/cublas            libcublas.so.12/.13 (GEMM/TRSM com kernels GLSL), include/cublas/cublas_v2.h
src/linalg            fatorações densas (LU, Cholesky, QR, eigh/SVD de Jacobi) em GLSL, compartilhadas por cuBLAS e cuSOLVER
src/cusolver          libcusolver.so.11/.12, include/cusolver/cusolverDn.h
src/cufft             libcufft.so.11/.12 (FFTs de Stockham e de Bluestein em GLSL), include/cufft/cufft.h
src/cusparse          libcusparse.so.12 (SpMV/SpMM/SDDMM/SpSV/csrgeam2 em GLSL), include/cusparse/cusparse.h
src/nccl              libnccl.so.2 (coletivas entre os ranks de uma máquina), include/nccl/nccl.h
src/nvrtc             biblioteca LD_AUDIT: o NVRTC entrega PTX em vez de um cubin (jiterator do PyTorch)
tests/                testes da API Driver + teste interno de Vulkan
tests/cuda            programas da API Runtime compilados com nvcc
tests/cudart_abi      conformidade ABI dos tipos clean-room do runtime com os cabeçalhos oficiais
tests/data            arquivos PTX de teste com seus fontes .cu ao lado; gen_ptx.sh os regenera (clang)
tools/                vulkanizer-info (lista de dispositivos), vulkanizer-ptxc (compilador offline), vulkanizer-bandwidth, setup_cuda.sh,
                      build_llama.sh (compilação CUDA do llama.cpp contra o Vulkanizer)
```
