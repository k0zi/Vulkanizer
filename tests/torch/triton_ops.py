# Triton on the Vulkanizer GPU: a hand-written kernel, and torch._native's Triton kernel for bmm
# outer products (run_torch.sh turns torch._native off by default): ./run_torch.sh tests/torch/triton_ops.py
import os
import sys

os.environ["TORCH_DISABLE_NATIVE_JIT"] = "0"  # read when torch is imported
import torch
import triton
import triton.language as tl
import torch._native.ops.bmm_outer_product.triton_kernels as native_bmm


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


failures = 0


def report(name, err, tol):
    global failures
    failures += err > tol
    print(f"{'OK  ' if err <= tol else 'DIFF'} {name}: max err {err:.2e}", flush=True)


n = 10000
x, y = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
out = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
report("triton add", (out - (x + y)).abs().max().item(), 0)

calls = 0
bmm_outer_product = native_bmm.bmm_outer_product


def counted(*args, **kwargs):
    global calls
    calls += 1
    return bmm_outer_product(*args, **kwargs)


native_bmm.bmm_outer_product = counted
a, b = torch.randn(6, 40, 1), torch.randn(6, 1, 24)
report("bmm outer product f32", (torch.bmm(a.cuda(), b.cuda()).cpu() - torch.bmm(a, b)).abs().max().item(), 1e-5)
ah, bh = a.half(), b.half()
report("bmm outer product f16",
       (torch.bmm(ah.cuda(), bh.cuda()).float().cpu() - torch.bmm(ah.float(), bh.float())).abs().max().item(), 2e-2)
report("bmm outer products not run by torch._native's Triton kernel", 2 - calls, 0)
sys.exit(1 if failures else 0)
