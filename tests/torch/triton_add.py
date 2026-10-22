# A minimal Triton kernel on the Vulkanizer GPU: ./run_torch.sh tests/torch/triton_add.py
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


n = 10000
x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
err = (out - (x + y)).abs().max().item()
print(f"{'OK' if err == 0 else 'DIFF'} triton add: max err {err:.2e}")
sys.exit(0 if err == 0 else 1)
