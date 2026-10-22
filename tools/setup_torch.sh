#!/usr/bin/env bash
# Creates a PyTorch environment for Vulkanizer in .work/torch-cf (plan.md 4.5).
#
#   tools/setup_torch.sh                create the environment (skipped if present)
#   tools/setup_torch.sh --force        recreate it
#   tools/setup_torch.sh --version V    PyTorch version (default 2.14.1)
#   tools/setup_torch.sh --cuda13       the CUDA 13.4 build in .work/torch-cf13 (libcudart.so.13)
#
# The official PyTorch wheels ship only native GPU code (SASS), which Vulkanizer cannot run;
# conda-forge's CUDA builds also carry PTX (compute_120), which Vulkanizer translates. The
# environment comes from conda-forge through micromamba (a static binary, fetched into
# .work/micromamba); nothing is installed system-wide. About 9 GB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="$ROOT/.work/torch-cf"
MAMBA="$ROOT/.work/micromamba/bin/micromamba"
VERSION=2.14.1
CUDA=12.9
BUILD_STRING=cuda129_generic
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --version) VERSION="$2"; shift ;;
    --cuda13) CUDA=13.4; BUILD_STRING=cuda134_generic; ENV="$ROOT/.work/torch-cf13" ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

if [[ $FORCE -eq 1 ]]; then rm -rf "$ENV"; fi
if [[ -x "$ENV/bin/python" ]]; then echo "already installed: $ENV"; exit 0; fi

if [[ ! -x "$MAMBA" ]]; then
  echo "==> micromamba"
  mkdir -p "$ROOT/.work/micromamba"
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$ROOT/.work/micromamba" bin/micromamba
fi

echo "==> conda-forge pytorch $VERSION (CUDA $CUDA build) into $ENV"
# The CUDA builds require the __cuda virtual package, i.e. a CUDA driver: Vulkanizer is it.
export MAMBA_ROOT_PREFIX="$ROOT/.work/mamba-root" CONDA_OVERRIDE_CUDA=$CUDA
"$MAMBA" create -y -p "$ENV" -c conda-forge python=3.13 "pytorch=$VERSION=$BUILD_STRING*" numpy
echo "==> done: ./run_torch.sh script.py"
