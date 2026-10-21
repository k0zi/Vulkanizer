# Trains a small MLP on the CPU and on the Vulkanizer GPU from the same start and compares:
# ./run_torch.sh tests/torch/mlp.py
import sys, time
import torch
torch.manual_seed(0)
X = torch.randn(256, 32); Y = (X[:, :4].sum(1, keepdim=True) > 0).float()
def make():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.GELU(),
                               torch.nn.Linear(64, 1))
def train(device, steps=50):
    model = make().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, y = X.to(device), Y.to(device)
    losses = []
    t = time.time()
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    acc = ((model(x) > 0).float() == y).float().mean().item()
    return losses, acc, [p.detach().cpu() for p in model.parameters()], time.time() - t
lc, ac, pc, tc = train('cpu')
lg, ag, pg, tg = train('cuda')
print(f"cpu  loss {lc[0]:.4f} -> {lc[-1]:.4f}  acc {ac:.3f}  {tc:.2f}s")
print(f"cuda loss {lg[0]:.4f} -> {lg[-1]:.4f}  acc {ag:.3f}  {tg:.2f}s")
print("max loss diff", max(abs(a - b) for a, b in zip(lc, lg)))
print("max param diff", max((a - b).abs().max().item() for a, b in zip(pc, pg)))
loss_diff = max(abs(a - b) for a, b in zip(lc, lg))
param_diff = max((a - b).abs().max().item() for a, b in zip(pc, pg))
sys.exit(0 if loss_diff < 1e-4 and param_diff < 1e-4 and ag == ac else 1)
