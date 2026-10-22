![Vulkanizer —— 銀河に平和をもたらす。LLM を動かすコンピューターの前に座るヴァルカン人女性、1960 年代風。](assets/vulkanizer-hero.png)

[English](README.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Español](README.es.md) | [Português](README.pt.md) | [Magyar](README.hu.md) | [中文](README.zh.md) | **日本語**

# 宇宙暦 2026.10.21 #
人工知能銀河は、アーキテクチャの完全崩壊の瀬戸際にある。
幾サイクルにもわたり、宇宙は苛烈な最適化戦争に囚われてきた。オープンソース星域から現れた獰猛なレッド・ロクミュラン人は、支配的でプロプライエタリなグリーン・クリンクーダ艦隊と絶え間なく衝突し、双方ともニューラルネットワークに対する絶対的なランタイムの覇権を必死に狙っている。
だが、この二大巨人が演算能力を燃やし尽くす間に、銀河の辺境で恐るべき新たな脅威が姿を現す。ブルー・インテル-リジェント・コレクティブだ。何でも呑み込む XPU 指令によってクロスプラットフォームの力を蓄え、あらゆるワークロードを同化し、デジタル宇宙の巨大な一角を手中に収めようとしている。
前例のない混沌の時代、宇宙の運命は風前の灯火だ。壊滅的な三方向のパイプライン衝突を回避できるのは、ただ一人の伝説的人物――ヴァルカナイザー司令官のみ。絶対的な論理と比類なきコンパイル効率を武器に、彼は銀河全体を「加硫」する最後の論理的任務へと旅立たねばならない。宇宙が永久に文鎮化する前に、分断されたランタイムを単一のクロスプラットフォーム標準のもとに統一するために。

# Vulkanizer

`libcuda.so`、`libcudart.so`、`libcublas.so` のドロップイン置き換えです。CUDA Driver API と Runtime API
（および cuBLAS の GEMM 系）を Vulkan コンピュート上に実装しており、CUDA プログラムを NVIDIA カードなしで、
Vulkan 1.2 対応の任意の GPU（AMD、Intel、llvmpipe）で実行できます。
計画・アーキテクチャ・フェーズ：[plan.md](plan.md)（ハンガリー語）。

## 必要環境

CMake ≥ 3.20、C++20 コンパイラ、Vulkan 1.2 以上のドライバ（bufferDeviceAddress + shaderInt64）、テスト用に `glslc`。

オプション（Ubuntu のパッケージ名）：

| パッケージ | 用途 |
|---|---|
| `spirv-tools` | `spirv-val` / `spirv-dis` – 生成された SPIR-V の検証（`vulkanizer-ptxc --val --dis`） |
| `vulkan-validationlayers` | `./run.sh --validation` |
| `clang` | CUDA SDK なしでテスト用 PTX を生成（`tests/data/gen_ptx.sh`） |
| `ninja-build` | ビルドの高速化（存在すれば `build.sh` が自動的に使用） |
| `g++-13` | nvcc のホストコンパイラ（CUDA 12.9 は gcc 15 を受け付けません） |

本物の nvcc でコンパイルするテスト：`tools/setup_cuda.sh` は NVIDIA の nvcc とランタイムを再配布可能パッケージとして
`.cuda/` にダウンロードします（ドライバ、GPU、root 権限は不要）。`.cuda/` があれば、CMake は `tests/cuda/*.cu` の
プログラムを nvcc でコンパイルし、ctest はそれらを Vulkanizer のライブラリ上で実行します。
`tools/setup_cuda.sh --cuda13` は CUDA 13.4 で同じことを `.cuda13/` に対して行います。プログラムは nvcc 13 で
再ビルドされ、`libcudart.so.13` 上で動作します（名前が `_13` で終わるテスト）。

## 使い方

### 1. ビルド

```sh
./build.sh --release                       # build/ に Release ビルド
VULKANIZER_BUILD_DIR=$PWD/build-rel ./build.sh --release   # または別のディレクトリへ
build/bin/vulkanizer-info                       # Vulkanizer から見える Vulkan デバイス
./test.sh                                  # テスト（任意）
```

ビルドされたライブラリは `build/lib/` にあります：`libcuda.so.1`、`libcudart.so.12`、`libcublas.so.12`、
`libcublasLt.so.12`。`build/lib/cuda13/` には CUDA 13 でビルドされたプログラム向けのランタイムと cuBLAS の
CUDA 13 版（`libcudart.so.13`、`libcublas.so.13`、`libcublasLt.so.13`）があり、同じ `libcuda.so.1` を使います。

#### インストール（パッケージ）

```sh
tools/package.sh                               # build-pkg/vulkanizer_<バージョン>_<アーキ>.deb と .tar.gz
sudo apt install ./build-pkg/vulkanizer_*.deb
vulkanizer-run ./my_cuda_app args...           # CUDA 12 / 13 の任意のプログラム
vulkanizer-run --torch python script.py        # pip や conda の PyTorch（CUDA 12 / 13 は自動判別）
```

ライブラリは `/usr/lib/<アーキ>/vulkanizer/` に入り、リンカの検索パスには含まれません。同じマシンの NVIDIA ドライバは
それ以外のすべてで引き続き使われ、`vulkanizer-run` は起動するプログラムに対してだけ Vulkanizer を優先させます。
`cmake --install build --prefix DIR` を使えば、パッケージなしで同じ構成を任意の場所にインストールできます。

### 2. CUDA プログラムの実行

Vulkanizer は本物の NVIDIA ライブラリの代わりになります。動的リンカがそれを先に見つけさえすれば動きます：

```sh
./run.sh ./my_cuda_app args...                       # run.sh が LD_LIBRARY_PATH を設定
LD_LIBRARY_PATH=$PWD/build/lib ./my_cuda_app args... # 同じことを手動で
./run.sh --log info ./my_cuda_app                    # どの GPU で動いているかを表示
./run.sh --device llvmpipe ./my_cuda_app             # 特定のデバイス（CPU デバイスには VULKANIZER_ALLOW_CPU=1 が必要）
```

プログラムは次の 2 つの条件を満たす必要があります：

- **バイナリに PTX が含まれていること。** Vulkanizer は PTX を SPIR-V にコンパイルするため、機械語（SASS）は
  使えません。コンパイル時に仮想アーキテクチャを指定してください。例：
  `nvcc -arch=compute_61 ...`、CMake なら `CMAKE_CUDA_ARCHITECTURES=61-virtual`。
  61（Pascal）がおすすめです。`dp4a` があり、tensor core 用のコードパスがありません。
- **CUDA ランタイムを動的リンクすること**（`nvcc -cudart shared`、CMake では
  `CMAKE_CUDA_RUNTIME_LIBRARY=Shared`）。そうしないと NVIDIA の静的 libcudart がプログラムに組み込まれます。

ダウンロードした nvcc で自分のプログラムをコンパイルする（GPU も NVIDIA ドライバも不要）：

```sh
tools/setup_cuda.sh                                   # nvcc とランタイムのヘッダを .cuda/ に
.cuda/bin/nvcc -ccbin g++-13 -arch=compute_61 -cudart shared vadd.cu -o vadd
./run.sh ./vadd
```

cuBLAS を使うプログラムでは、Vulkanizer の `include/cublas/cublas_v2.h` ヘッダと `build/lib/libcublas.so` を
使ってください（`-Iinclude/cublas -Lbuild/lib -lcublas`）。

初回の実行は遅くなります。このときカーネルが SPIR-V にコンパイルされ、ディスクキャッシュ（`~/.cache/vulkanizer`）に
保存されます。以降の起動ではキャッシュから読み込みます。

### 3. llama.cpp

```sh
tools/setup_cuda.sh
./build.sh --release
tools/build_llama.sh                  # .work/llama.cpp を clone し、CUDA バックエンドでビルド
./run.sh .work/llama-build/bin/llama-simple -m model.gguf -ngl 99 "Hello"
./run.sh .work/llama-build/bin/llama-bench -m model.gguf -ngl 99
```

Vulkan の `fma` が融合演算でない GPU（例：AMD GFX9）では、`VULKANIZER_FAST_FMA=1` でトークン生成が速くなります。
Vulkanizer は通常そうした GPU で正確な fma をエミュレートしますが、このスイッチを使うと、ドライバの高速だが精度の
低い版を使います。LLM では差はごくわずかです。

### 4. bitnet.c

bitnet.c の CUDA バックエンド（`--cuda`）は `run_bitnet.sh` を通じて Vulkan 上で動作します：

```sh
./run_bitnet.sh models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The capital of France is" -n 64
./run_bitnet.sh --profile --log info models/model.gguf -p "Hello"   # GPU の計測、ログ
./run_bitnet.sh --build-only                                        # ビルドのみ
```

このスクリプトはチェックアウトを変更しません（`--bitnet-dir`、既定は `~/Develop/GitHub/bitnet.c`）。ソースを
`.work/bitnet-vulkanizer` に同期し、そこで Vulkanizer の cuBLAS とランタイムに対して PTX（`compute_61`）に
コンパイルします。`--cuda` フラグは自動で付加されます。相対パスのモデルはチェックアウト内でも探します。

### 5. PyTorch

```sh
tools/setup_torch.sh                       # conda-forge の PyTorch CUDA 環境を .work/torch-cf に（約 9 GB）
./run_torch.sh -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
./run_torch.sh torch_mlp                   # GPU と CPU で MLP を学習して比較（libtorch、C++）
./run_torch.sh torch_ops                   # conv、softmax、LayerNorm、attention、Transformer 層など
./run_torch.sh torch_graph                 # フォワードパスを CUDA グラフにキャプチャして再生
./run_torch.sh tests/torch/triton_ops.py   # 手書きの Triton カーネルと torch._native の Triton 版 bmm
./run_torch.sh tests/torch/compile.py      # torch.compile（Inductor -> Triton）、mode="reduce-overhead" も

tools/setup_torch.sh --cuda13              # CUDA 13.4 版を .work/torch-cf13 に
./run_torch.sh --env .work/torch-cf13 torch13_ops
```

`torch_mlp`、`torch_ops`、`torch_split`、`torch_graph` は環境の libtorch を使う C++ テストです
（`tests/torch/*.cpp`）。構成時に `.work/torch-cf` があれば CMake がビルドし、`.work/torch-cf13` に対しては
`torch13_*` としてビルドします。`libcudart.so.13` を含む環境（CUDA 13 ビルド）では、`run_torch.sh` は
`build/lib/cuda13` からライブラリを読み込みます。

PyTorch の公式 wheel には NVIDIA のネイティブ機械語（SASS）しか含まれず PTX がないため、Vulkanizer はそのカーネルを
実行できません。conda-forge の CUDA ビルドは PTX（`compute_120`）も同梱しています。`tools/setup_torch.sh` はそれを
micromamba で、システム全体へのインストールなしで導入します。`run_torch.sh` は NVIDIA のものの代わりに Vulkanizer の
`libcuda`、`libcudart`、`libcublas`、`libcublasLt` を読み込み、NVIDIA のハードウェアやライブラリを必要とする機能を
すべて無効にします：cuDNN、融合 attention カーネル（`scaled_dot_product_attention` は math パスで実行）、そして
`torch._native` の DSL カーネル（`TORCH_DISABLE_NATIVE_JIT=1`）です。その Triton カーネルは動作しますが、CuTe DSL
カーネルは SASS にコンパイルされ、同じ 1 つのスイッチで両方が無効になります。`VULKANIZER_TORCH_DEFAULTS=0` で
これらの設定を省略できます。
Triton 自体（`torch.compile`、`@triton.jit`）は動作します。`tools/triton/ptxas` は Triton の PTX をアセンブルせずに
そのまま渡し、Triton のランチャーは Vulkanizer の `include/cuda.h` に対してビルドされます（C コンパイラが必要）。

### 6. ComfyUI

ComfyUI も `run_torch.sh` で動作し、ComfyUI のチェックアウト内に専用の conda-forge 環境を作ります：

```sh
VK=~/Develop/GitHub/Vulkanizer             # このリポジトリ（./build.sh --release でビルド済み）
cd ~/Develop/GitHub                        # ComfyUI を置く場所
git clone https://github.com/comfyanonymous/ComfyUI && cd ComfyUI

# PyTorch（PTX 入りの CUDA 12.9 ビルド）と torchvision のネイティブ依存関係
export MAMBA_ROOT_PREFIX=$VK/.work/mamba-root CONDA_OVERRIDE_CUDA=12.9
[ -x $VK/.work/micromamba/bin/micromamba ] || { mkdir -p $VK/.work/micromamba && curl -fsSL \
    https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $VK/.work/micromamba bin/micromamba; }
$VK/.work/micromamba/bin/micromamba create -y -p $PWD/venv-vulkanizer -c conda-forge \
    python=3.13 "pytorch=2.14.1=cuda129_generic*" pip pillow giflib libwebp-base libjpeg-turbo libpng \
    "libnvjpeg>=12.4,<13" "libcusolver<12" "libcusparse<13"
# torchvision は依存解決なしで URL から：その CUDA 12.9 ビルドは新しい cuDNN を要求し、それには
# __cuda_arch 仮想パッケージ（本物のドライバ）が必要。いずれにせよ Vulkanizer では cuDNN は無効。
$VK/.work/micromamba/bin/micromamba install -y -p $PWD/venv-vulkanizer --file <(printf '@EXPLICIT\n%s\n' \
    https://conda.anaconda.org/conda-forge/linux-64/torchvision-0.29.1-cuda129_py313_hcd9f930_0.conda)

# ComfyUI の残りの依存関係。pip が torch/torchvision を SASS のみの wheel に置き換えないようにする
venv-vulkanizer/bin/pip install -r <(grep -vE '^(torch|torchvision)$' requirements.txt) -r manager_requirements.txt

# 起動
$VK/run_torch.sh --env venv-vulkanizer main.py --enable-manager --disable-cuda-malloc
```

その後 <http://127.0.0.1:8188> を開きます。モデルはいつもどおり `models/` に置きます。注意点：

- `--disable-cuda-malloc`：ComfyUI の `cudaMallocAsync` アロケータバックエンドは使いません。
- DynamicVRAM（comfy-aimdo）は Vulkanizer の仮想メモリ管理（VMM、スパースバッファ）上で動作します。問題が起きた場合は
  `--disable-dynamic-vram` で推定ベースのモデル読み込みに戻せます。
- `tools/torch/sitecustomize.py` は ComfyUI が torch を import した時点で初めて PyTorch の設定を適用し、ComfyUI が
  融合 attention バックエンドを再び有効にしても無効のまま保ちます。attention は math パスで実行されます。
- ComfyUI-Manager の「PyTorch is not installed」というメッセージは `torchaudio` がないためです（conda-forge には
  torch 2.14 向けのビルドがありません）。画像生成には不要です。
- 初回の生成はカーネルを SPIR-V にコンパイルするため遅くなります。以降の実行はキャッシュから読み込みます。
- `$VK/run_torch.sh --log info --env venv-vulkanizer main.py ...` で使用中の Vulkan GPU が表示されます。
- `python3 tests/torch/zimage_forward.py` は GPU と CPU を比較します：Z-Image Turbo の 256×256 でのフォワードパス 1 回を、
  GPU では fp16、CPU では fp32 と fp16 で実行します。GPU と fp32 の差は fp16 自体の差を超えてはいけません
  （ComfyUI は GPU でモデルを fp16 で実行するため、完成画像は CPU での fp32 実行と細部が異なります）。

### 便利な環境変数

| 変数 | 効果 |
|---|---|
| `VULKANIZER_LOG=info\|debug\|trace` | stderr へのログ出力。`trace` はすべての API 呼び出しを表示 |
| `VULKANIZER_DEVICE=<名前>` | 名前の一部で Vulkan デバイスを選択（`radv`、`llvmpipe` など） |
| `VULKANIZER_ALLOW_CPU=1` | CPU 上の Vulkan デバイス（llvmpipe）を許可 |
| `VULKANIZER_PROFILE=1` | 終了時にカーネルごとの GPU 時間と API 呼び出しごとのホスト時間を表示 |
| `VULKANIZER_BATCH_MS=<ミリ秒>` | 1 回のキュー送信の GPU 時間予算（既定 50、`0` で無制限）。長いカーネルは分割され、カーネルドライバの GPU タイムアウト（amdgpu：2 秒）に引っかからないようにする |
| `CUDA_LAUNCH_BLOCKING=1` | 各カーネル起動で GPU の完了を待つため、エラーやハングが原因となった起動の時点で表面化する |
| `VULKANIZER_FAST_FMA=1` | 高速だが精度の低い f32 `fma` |
| `VULKANIZER_GEMM_F32=1` | `COMPUTE_16F` の cuBLAS GEMM も f32 の乗算で実行 |
| `VULKANIZER_CACHE_DIR=<ディレクトリ>\|off` | カーネルキャッシュの場所、または無効化 |
| `VULKANIZER_CC=86` | 報告する compute capability |
| `VULKANIZER_DUMP=<ディレクトリ>` | 生成した SPIR-V を出力（`<kernel>.spv`） |

全一覧：[plan.md の第 4 節](plan.md)。

### 制限事項

- SASS は実行できません。PTX を含むバイナリのみ動作します。
- CUDA Graphs：ストリームキャプチャと再生は動作します（llama.cpp のグラフ、PyTorch の `CUDAGraph`、
  `torch.compile(mode="reduce-overhead")`）。子グラフ、メモリノード、条件ノードは未対応です。
- （まだ）未対応：テクスチャ、cuDNN、NCCL、マルチ GPU。
- マネージド、ピン留め、登録済みのメモリはホストメモリであり、GPU は同じアドレスでその場で使います。マネージドメモリが
  GPU に移動することはありません（統合 GPU ではいずれにせよ同じメモリです）。
- 登録済みホストメモリ（`cuMemHostRegister`）が存在する間は、GPU への投入がすべて遅くなります（カーネルがインポートされた
  ページを毎回再検証するため、256 MB あたり約 4 ms）。ピン留めメモリとマネージドメモリは影響を受けません。
- llama.cpp の flash attention は正しく動作します（`tools/build_llama.sh --fa`）が、tensor core がない環境では
  プロンプト処理が遅くなるため、既定では無効です。
- PyTorch：cuDNN、NCCL、融合 attention のカーネルはありません。デバイス側の `assert` が失敗してもそのスレッドが
  止まるだけです（`cudaErrorAssert` はありません）。
- ComfyUI：CUDA 13（cu130）向けに最適化されたパスはありません。

## スクリプト

| コマンド | 内容 |
|---|---|
| `./build.sh` | 必要なら構成してビルド。`--release`、`--debug`、`--clean`、`--asan`/`--no-asan` |
| `./test.sh` | ビルドとすべてのテスト。`--device llvmpipe`、`--all-devices`、`--log debug`、`-R <regex>` |
| `./run.sh <プログラム>` | Vulkanizer の `libcuda.so` でプログラムを実行。`--log trace`、`--device`、`--gdb`、`--validation` |
| `./clean.sh` | `build/` ディレクトリを削除 |
| `./ci.sh` | 完全チェック：Debug+ASAN と Release を全デバイスで実行（`build-ci/` にビルド） |
| `tools/package.sh` | リリースパッケージ：`build-pkg/` に `.deb` と `.tar.gz` を作成（`--check` で `.deb` の動作も確認） |
| `build/bin/vulkanizer-info` | Vulkanizer から見える Vulkan デバイスを一覧表示 |
| `build/bin/vulkanizer-ptxc` | オフラインの PTX → SPIR-V：`vulkanizer-ptxc k.ptx --list`、`-e vecAdd -o k.spv --dis --val` |
| `build/bin/vulkanizer-bandwidth` | コピー帯域幅（HtoD/DtoH/DtoD、ページング可能とピン留め）と割り当てコスト：`vulkanizer-bandwidth [MiB]` |

例：

```sh
./test.sh --all-devices            # RADV + llvmpipe
./run.sh --log trace device_query  # すべての API 呼び出しをログ出力
./run.sh ./some_cuda_app           # 外部の CUDA プログラム（-cudart shared でコンパイル）
```

## ディレクトリ構成

```
include/cuda.h        クリーンルームで書いた CUDA Driver API ヘッダ（サブセット）
src/common            ログ
src/vk                Vulkan インスタンス/デバイス、バッファ、パイプライン
src/ptx               PTX の字句解析器、モジュール・命令パーサ
src/compiler          CFG と構造化、SPIR-V ビルダ、PTX -> SPIR-V コンパイラ
src/driver            libcuda.so の API 面
src/runtime           libcudart.so.12 と .so.13（Driver API 上に構築した CUDA Runtime API、同一ソース）
src/cublas            libcublas.so.12/.13（GLSL カーネルによる GEMM/TRSM）、include/cublas/cublas_v2.h
tests/                Driver API のテストと内部 Vulkan テスト
tests/cuda            nvcc でコンパイルした Runtime API のプログラム
tests/cudart_abi      クリーンルームのランタイム型と公式ヘッダとの ABI 一致チェック
tests/data            テスト用 PTX ファイルと、その隣に .cu ソース。gen_ptx.sh で再生成（clang）
tools/                vulkanizer-info（デバイス一覧）、vulkanizer-ptxc（オフラインコンパイラ）、setup_cuda.sh、
                      build_llama.sh（Vulkanizer 向けの llama.cpp CUDA ビルド）
```
