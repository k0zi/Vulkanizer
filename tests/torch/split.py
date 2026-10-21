# Kernels split over many batches (run with a tiny VULKANIZER_BATCH_MS, e.g. 0.01): every chunk
# must see its own parameters, also kernels whose parameters live in a buffer (TensorIterator
# offset calculators of non-contiguous tensors), after the batch slots have been reused.
import torch
import torch.nn.functional as F

torch.manual_seed(0)
ok = True


def check(name, gpu, cpu, tol):
    global ok
    err = (gpu.float().cpu() - cpu.float()).abs().max().item()
    good = err <= tol
    ok &= good
    print(f"{'OK  ' if good else 'FAIL'} {name}: max err {err:.2e}")


x = torch.randn(1, 4128, 3840)
xg = x.cuda().half()
# Launched back to back, without synchronizing in between: the later launches upload their
# parameters while the chunks of the earlier ones are still waiting in reused batch slots.
# Different sizes split into different numbers of batches, so the batch slots wrap around at
# varying points; one bad launch is enough to fail.
xh = x.half()
views = [lambda t, s=s: t[:, s::s + 1, ::s + 2] for s in range(20)] + \
        [lambda t: t.transpose(1, 2), lambda t: t.view(1, 4128, 30, 128).permute(0, 2, 3, 1)]
out = [v(xg).contiguous() for _ in range(3) for v in views]
bad = 0
for i, o in enumerate(out):
    bad += not torch.equal(o.cpu(), views[i % len(views)](xh).contiguous())
check(f"{len(out)} strided copies, mismatching", torch.tensor(float(bad)), torch.tensor(0.0), 0)
check("layer_norm", F.layer_norm(xg, (3840,)), F.layer_norm(x, (3840,)), 2e-2)
a = torch.randn(512, 768)
b = torch.randn(768, 1024)
check("matmul", a.cuda() @ b.cuda(), a @ b, 1e-3)
print("ALL OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
