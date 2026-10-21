# Loaded by run_torch.sh (PYTHONPATH): PyTorch settings for running on Vulkanizer.
#  - cuDNN is NVIDIA's own library and needs NVIDIA's driver: off, PyTorch's own kernels are used.
#  - The fused attention kernels (flash, memory-efficient, cuDNN) are built per GPU generation
#    around tensor cores and refuse to run here: scaled_dot_product_attention uses its math path.
# VULKANIZER_TORCH_DEFAULTS=0 skips this.
import os

if os.environ.get("VULKANIZER_TORCH_DEFAULTS", "1") != "0":
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.backends.cudnn.enabled = False
        for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
            if hasattr(torch.backends.cuda, name):
                getattr(torch.backends.cuda, name)(False)
