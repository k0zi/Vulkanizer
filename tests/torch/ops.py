# PyTorch operations on the Vulkanizer GPU against the CPU: ./run_torch.sh tests/torch/ops.py
import sys
import torch, torch.nn.functional as F
failures = 0
torch.manual_seed(0)
def fail():
    global failures
    failures += 1
def check(name, fn, tol=1e-4):
    try:
        cpu = fn('cpu'); gpu = fn('cuda')
        cpu = cpu if isinstance(cpu, (list, tuple)) else [cpu]; gpu = gpu if isinstance(gpu, (list, tuple)) else [gpu]
        err = max((a.float() - b.float().cpu()).abs().max().item() for a, b in zip(cpu, gpu))
        print(f"{'OK  ' if err <= tol else 'DIFF'} {name}: max err {err:.2e}", flush=True)
        if err > tol: fail()
    except Exception as e:
        fail()
        print(f"FAIL {name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
x = torch.randn(4, 3, 16, 16); w = torch.randn(8, 3, 3, 3)
check("conv2d (cudnn)", lambda d: F.conv2d(x.to(d), w.to(d), padding=1), 1e-3)
def conv_nocudnn(d):
    torch.backends.cudnn.enabled = False
    try: return F.conv2d(x.to(d), w.to(d), padding=1)
    finally: torch.backends.cudnn.enabled = True
check("conv2d (no cudnn)", conv_nocudnn, 1e-3)
a = torch.randn(8, 100)
check("softmax", lambda d: F.softmax(a.to(d), -1), 1e-6)
check("layer_norm", lambda d: F.layer_norm(a.to(d), (100,)), 1e-5)
emb = torch.randn(50, 16); idx = torch.randint(0, 50, (20,))
check("embedding", lambda d: F.embedding(idx.to(d), emb.to(d)), 0)
q = torch.randn(2, 4, 32, 16)
check("sdpa", lambda d: F.scaled_dot_product_attention(q.to(d), q.to(d), q.to(d)), 1e-4)
m1 = torch.randn(64, 64); m2 = torch.randn(64, 64)
check("matmul f16", lambda d: (m1.to(d).half() @ m2.to(d).half()).float() if d == 'cuda' else (m1.half().float() @ m2.half().float()), 0.5)
check("cumsum/sort/topk", lambda d: [a.to(d).cumsum(1), a.to(d).sort(1).values, a.to(d).topk(5, 1).values], 1e-4)
check("argmax/max", lambda d: [a.to(d).argmax(1).float(), a.to(d).max(1).values], 0)
check("index/where", lambda d: [torch.where(a.to(d) > 0, a.to(d), 0), a.to(d)[:, ::3]], 0)
def tr(d):
    torch.manual_seed(2); layer = torch.nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True).to(d)
    return layer(torch.randn(2, 10, 32, generator=torch.Generator().manual_seed(3)).to(d))
check("transformer layer", tr, 1e-4)
src = torch.randn(30, 8); index = torch.randint(0, 5, (30,))
check("scatter_add f16", lambda d: torch.zeros(5, 8, device=d, dtype=torch.half).index_add_(0, index.to(d), src.to(d).half()).float(), 0.05)
check("index_add f32", lambda d: torch.zeros(5, 8, device=d).index_add_(0, index.to(d), src.to(d)), 1e-5)
# torch._native sends these to a Triton kernel unless TORCH_DISABLE_NATIVE_JIT=1 (run_torch.sh's default)
ca = torch.randn(6, 40, 1); cb = torch.randn(6, 1, 24)
check("bmm outer product f32", lambda d: torch.bmm(ca.to(d), cb.to(d)), 1e-5)
check("bmm outer product f16", lambda d: torch.bmm(ca.to(d).half(), cb.to(d).half()).float() if d == 'cuda' else torch.bmm(ca.half().float(), cb.half().float()), 2e-2)
sys.exit(1 if failures else 0)
