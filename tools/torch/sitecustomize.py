# Loaded by run_torch.sh (PYTHONPATH) unless VULKANIZER_TORCH_DEFAULTS=0; the rest of the PyTorch
# settings are environment variables there. This turns off what only Python can:
#  - cuDNN is NVIDIA's own library and needs NVIDIA's driver: PyTorch's own kernels are used.
#  - scaled_dot_product_attention: the memory-efficient backend stays on, its forward kernels run as
#    Vulkanizer's own fused attention (src/driver/attention.h) where that is fast (_route_attention;
#    the backward kernels do not: below sm_80 their PTX is a stub). bf16 is left to the math path by
#    PyTorch itself below compute capability 8.0. The flash and cuDNN backends are off, even where the
#    program turns them back on (ComfyUI: sdpa_kernel([FLASH, CUDNN, EFFICIENT, MATH])).
#    VULKANIZER_TORCH_SDPA lists the backends to keep instead: "math" for none, "mem_efficient,flash"
#    adds the flash kernels (sm_80 ones with emulated tensor core instructions; run_torch.sh then
#    reports compute capability 8.0). cuDNN's always stays off. Without the fused attention
#    (VULKANIZER_FUSED_ATTENTION=0) the default is none.
# Applied right after torch is imported, not here: some programs (ComfyUI) set up their environment
# first and check that torch has not been imported yet, or look it up with importlib.util.find_spec.
import importlib.abc
import importlib.util
import os
import sys


def _apply(torch):
    torch.backends.cudnn.enabled = False
    default = "math" if os.environ.get("VULKANIZER_FUSED_ATTENTION") == "0" else "mem_efficient"
    keep = {b.strip() for b in os.environ.get("VULKANIZER_TORCH_SDPA", default).split(",")} - {"cudnn"}
    for backend, name in (("flash", "_set_sdp_use_flash"), ("mem_efficient", "_set_sdp_use_mem_efficient"),
                          ("cudnn", "_set_sdp_use_cudnn")):
        if backend in keep:
            continue
        setter = getattr(torch._C, name, None)
        if setter is not None:  # every way of enabling a backend ends in these setters
            setter(False)
            setattr(torch._C, name, lambda enabled, _setter=setter: _setter(False))
    if "mem_efficient" in keep and "flash" not in keep:
        _route_attention(torch)


# The fused attention runs scaled_dot_product_attention where it is fast: forward only (with gradients
# the backward kernels would be needed), f16 (bf16 where the device has bf16 cooperative matrices) with
# head dimensions up to 128, on cooperative matrices. Everything else takes the math path (PyTorch's
# GEMMs), which is faster there than the fused attention's scalar kernel.
def _route_attention(torch):
    import ctypes
    import functools
    import importlib

    importlib.import_module("torch.nn.attention")
    functional = torch.nn.functional
    sdpa = functional.scaled_dot_product_attention
    math_only = torch.nn.attention.SDPBackend.MATH
    features = {}  # device index -> (f16, bf16) cooperative matrices of 16 x 16

    def cooperative(device):
        if device.index not in features:
            f16 = bf16 = ctypes.c_int(0)
            try:
                get = ctypes.CDLL("libcuda.so.1").vulkanizerGetFeature
                with torch.cuda.device(device):
                    f16, bf16 = ctypes.c_int(0), ctypes.c_int(0)
                    get(3, ctypes.byref(f16))  # VULKANIZER_FEATURE_COOPERATIVE_MATRIX_F16: the size
                    get(5, ctypes.byref(bf16))  # VULKANIZER_FEATURE_COOPERATIVE_MATRIX_BF16
            except (OSError, AttributeError):
                pass
            features[device.index] = (f16.value == 16, f16.value == 16 and bf16.value != 0)
        return features[device.index]

    def fused(query, key, value):
        if query.device.type != "cuda" or query.shape[-1] > 128 or value.shape[-1] > 128:
            return False
        if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
            return False
        f16, bf16 = cooperative(query.device)
        return (query.dtype == torch.float16 and f16) or (query.dtype == torch.bfloat16 and bf16)

    @functools.wraps(sdpa)
    def scaled_dot_product_attention(query, key, value, *args, **kwargs):
        if fused(query, key, value):
            return sdpa(query, key, value, *args, **kwargs)
        with torch.nn.attention.sdpa_kernel([math_only]):
            return sdpa(query, key, value, *args, **kwargs)

    functional.scaled_dot_product_attention = scaled_dot_product_attention


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
