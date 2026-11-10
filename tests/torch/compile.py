# torch.compile (Inductor -> Triton) on the Vulkanizer GPU against eager on the CPU, forward and
# backward of a small MLP, and the forward pass with mode="reduce-overhead" (the Triton kernels
# replayed from CUDA graphs): ./run_torch.sh tests/torch/compile.py
import copy
import sys
import torch

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.LayerNorm(64), torch.nn.Linear(64, 8))
x = torch.randn(128, 32)


def run(device, compile):
    m = model.to(device)
    f = torch.compile(m) if compile else m
    xd = x.to(device, copy=True).requires_grad_()
    out = f(xd)
    (out.square().mean() + out.softmax(-1)[:, 0].sum()).backward()
    r = [out.detach(), xd.grad] + [p.grad for p in m.parameters()]
    r = [t.cpu().clone() for t in r]
    m.zero_grad()
    return r


ref = run("cpu", False)
got = run("cuda", True)
err = max((a - b).abs().max().item() / (1 + a.abs().max().item()) for a, b in zip(ref, got))
ok = err < 1e-4
print(f"{'OK' if ok else 'DIFF'} torch.compile MLP forward+backward: max rel err {err:.2e}")

reference = copy.deepcopy(model).cpu()
graphed = torch.compile(model.to("cuda"), mode="reduce-overhead")
with torch.no_grad():
    for step in range(4):  # the first calls record, later ones replay the CUDA graph
        xs = torch.randn(128, 32)
        e = (graphed(xs.cuda()).cpu() - reference(xs)).abs().max().item()
        print(f"{'OK' if e < 1e-4 else 'DIFF'} reduce-overhead call {step}: max err {e:.2e}")
        ok &= e < 1e-4
sys.exit(0 if ok else 1)
