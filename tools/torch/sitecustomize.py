# Loaded by run_torch.sh (PYTHONPATH): PyTorch settings for running on Vulkanizer.
#  - cuDNN is NVIDIA's own library and needs NVIDIA's driver: off, PyTorch's own kernels are used.
#  - The fused attention kernels (flash, memory-efficient, cuDNN) are built per GPU generation
#    around tensor cores and refuse to run here: scaled_dot_product_attention uses its math path.
#  - torch._native routes some ops (e.g. bmm outer products) to Triton/CuTe DSL kernels, which are
#    compiled to SASS: TORCH_DISABLE_NATIVE_JIT=1 keeps them on PyTorch's own (PTX) kernels.
# The settings are applied when the program imports torch, not here: some programs (ComfyUI) set
# up their environment before importing torch and check that it has not been imported yet.
# VULKANIZER_TORCH_DEFAULTS=0 skips this.
import importlib.abc
import importlib.util
import os
import sys


def _apply(torch):
    torch.backends.cudnn.enabled = False
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
        if hasattr(torch.backends.cuda, name):
            getattr(torch.backends.cuda, name)(False)


class _TorchImportHook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "torch":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_apply(module):
            exec_module(module)
            _apply(module)

        spec.loader.exec_module = exec_and_apply
        return spec


if os.environ.get("VULKANIZER_TORCH_DEFAULTS", "1") != "0":
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    sys.meta_path.insert(0, _TorchImportHook())
