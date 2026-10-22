# Loaded by run_torch.sh (PYTHONPATH): PyTorch settings for running on Vulkanizer.
#  - cuDNN is NVIDIA's own library and needs NVIDIA's driver: off, PyTorch's own kernels are used.
#  - The fused attention kernels (flash, memory-efficient, cuDNN) are built per GPU generation
#    around tensor cores and refuse to run here: scaled_dot_product_attention uses its math path,
#    even where the program turns the other backends back on.
#  - torch._native routes some ops to Triton (bmm outer products) and CuTe DSL kernels. The Triton
#    ones work, the CuTe DSL ones compile to SASS, and one switch covers both:
#    TORCH_DISABLE_NATIVE_JIT=1 keeps them on PyTorch's own (PTX) kernels.
#  - Triton (torch.compile) runs with tools/triton/ptxas, which hands the PTX through instead of
#    assembling it, and builds its launcher against Vulkanizer's include/cuda.h.
# The settings are applied when the program imports torch, not here: some programs (ComfyUI) set
# up their environment before importing torch and check that it has not been imported yet.
# VULKANIZER_TORCH_DEFAULTS=0 skips this.
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _apply(torch):
    torch.backends.cudnn.enabled = False
    # Programs re-enable backends (ComfyUI: sdpa_kernel([FLASH, CUDNN, EFFICIENT, MATH])), and
    # every path ends in these setters, so they keep the fused backends off for good.
    for name in ("_set_sdp_use_flash", "_set_sdp_use_mem_efficient", "_set_sdp_use_cudnn"):
        setter = getattr(torch._C, name, None)
        if setter is not None:
            setter(False)
            setattr(torch._C, name, lambda enabled, _setter=setter: _setter(False))


def _apply_triton_driver(driver):
    # Triton compiles its launcher (driver.c) against cuda.h, which a CUDA-less environment lacks.
    driver.include_dirs.append(str(_ROOT / "include"))


# Runs `apply` on a module right after it is executed. Stays installed until then: programs look
# torch up without importing it first (ComfyUI's cuda_malloc.py calls importlib.util.find_spec("torch")).
class _PostImportHook(importlib.abc.MetaPathFinder):
    def __init__(self, module, apply):
        self._module = module
        self._apply = apply
        self._searching = False

    def find_spec(self, name, path, target=None):
        if name != self._module or self._searching:
            return None
        self._searching = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._searching = False
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_apply(module):
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            exec_module(module)
            self._apply(module)

        spec.loader.exec_module = exec_and_apply
        return spec


if os.environ.get("VULKANIZER_TORCH_DEFAULTS", "1") != "0":
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    sys.meta_path.insert(0, _PostImportHook("torch", _apply))
    ptxas = str(_ROOT / "tools" / "triton" / "ptxas")
    os.environ.setdefault("TRITON_PTXAS_PATH", ptxas)
    os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", ptxas)
    sys.meta_path.insert(0, _PostImportHook("triton.backends.nvidia.driver", _apply_triton_driver))
