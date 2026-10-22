# Loaded by run_torch.sh (PYTHONPATH) unless VULKANIZER_TORCH_DEFAULTS=0; the rest of the PyTorch
# settings are environment variables there. This turns off what only Python can:
#  - cuDNN is NVIDIA's own library and needs NVIDIA's driver: PyTorch's own kernels are used.
#  - The fused attention kernels (flash, memory-efficient, cuDNN) are built per GPU generation around
#    tensor cores and refuse to run here: scaled_dot_product_attention uses its math path, even where
#    the program turns the other backends back on (ComfyUI: sdpa_kernel([FLASH, CUDNN, EFFICIENT, MATH])).
# Applied right after torch is imported, not here: some programs (ComfyUI) set up their environment
# first and check that torch has not been imported yet, or look it up with importlib.util.find_spec.
import importlib.abc
import importlib.util
import sys


def _apply(torch):
    torch.backends.cudnn.enabled = False
    for name in ("_set_sdp_use_flash", "_set_sdp_use_mem_efficient", "_set_sdp_use_cudnn"):
        setter = getattr(torch._C, name, None)
        if setter is not None:  # every way of enabling a backend ends in these setters
            setter(False)
            setattr(torch._C, name, lambda enabled, _setter=setter: _setter(False))


# Stays installed until torch is executed: a find_spec lookup alone must not use it up.
class _TorchHook(importlib.abc.MetaPathFinder):
    searching = False

    def find_spec(self, name, path, target=None):
        if name != "torch" or self.searching:
            return None
        self.searching = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.searching = False
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_apply(module):
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            exec_module(module)
            _apply(module)

        spec.loader.exec_module = exec_and_apply
        return spec


sys.meta_path.insert(0, _TorchHook())
