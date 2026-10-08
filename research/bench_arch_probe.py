"""Is the data learnable at all, and which component is blocking it?

Three models, same data, same lr, same step budget:
  mlp    - no transformer, flattened 6144 -> hidden -> 4096
  t2     - 2-layer Pre-LN transformer (depth is not the problem if this wins)
  t10    - 10-layer Pre-LN transformer (current params.yaml depth)

If mlp/t2 clearly beat t10, depth is the blocker. If all three stall near
ln-distinct-labels entropy, the mean-pool head is the blocker.
"""
import sys
import time

import torch
from torch import nn

from Chess.config.configuration import ConfigurationManager
from Chess.components.torch_base_model import ChessPolicyNet

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
BATCH = 64
LR = 3e-4
SEED = 0


class PreLNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.params_D_MODEL, num_heads=config.params_NUM_HEADS, batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.params_D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(config.params_D_MODEL, config.params_FF_DIM),
            nn.ReLU(),
            nn.Linear(config.params_FF_DIM, config.params_D_MODEL),
        )
        self.norm2 = nn.LayerNorm(config.params_D_MODEL)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h)
        x = x + self.dropout(a)
        h = self.norm2(x)
        return x + self.dropout(self.ffn(h))


class PreLNNet(ChessPolicyNet):
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([PreLNBlock(config) for _ in range(config.params_NUM_LAYERS)])
        self.final_norm = nn.LayerNorm(config.params_D_MODEL)
        with torch.no_grad():
            self.pos_embd.mul_(0.02)

    def forward(self, x):
        b = x.shape[0]
        x = self.input_proj(x.view(b, 64, -1)) + self.pos_embd
        for block in self.blocks:
            x = block(x)
        return self.out(self.relu(self.fc(self.final_norm(x).mean(dim=1))))


class MLP(nn.Module):
    """No attention, no pooling: flatten 8*8*12 and go straight to 4096."""

    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 8 * 12, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4096),
        )

    def forward(self, x):
        return self.net(x)


def entropy_of_labels(y):
    counts = torch.bincount(y, minlength=4096).float()
    p = counts[counts > 0] / len(y)
    return float(-(p * p.log()).sum())


def run(name, build, x, y):
    torch.manual_seed(SEED)
    net = build()
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    losses = []
    for step in range(STEPS):
        i = torch.randint(0, len(x), (BATCH,))
        opt.zero_grad()
        loss = ce(net(x[i]), y[i])
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == STEPS - 1:
            w = losses[-100:]
            print(f"  {name:5s} step {step:4d} | loss {sum(w)/len(w):7.4f} | {time.time()-t0:5.1f}s", flush=True)
    net.eval()
    with torch.no_grad():
        logits = net(x[:4096])
        pred = logits.argmax(1)
        acc = (pred == y[:4096]).float().mean().item()
    print(f"  {name:5s} FINAL loss {sum(losses[-100:])/100:.4f} | top-1 {acc*100:5.2f}% "
          f"| distinct {len(torch.unique(pred)):5d} | params {nparam/1e6:.2f}M\n", flush=True)
    return sum(losses[-100:]) / 100


def main():
    base = ConfigurationManager().get_torch_base_model()
    mk = lambda n: type(base)(**{**base.__dict__, "params_NUM_LAYERS": n})

    x = torch.load("artifacts/torch_dataset/x_tensor.pt", map_location="cpu")
    y = torch.load("artifacts/torch_dataset/y_tensor.pt", map_location="cpu").long()
    print(f"data {len(x)} positions | lr {LR} | batch {BATCH} | {STEPS} steps")
    print(f"uniform loss ln(4096)      = 8.3178")
    print(f"label-distribution entropy = {entropy_of_labels(y):.4f}  <- 'predict the marginal' floor\n")

    r = {}
    r["mlp"] = run("mlp", lambda: MLP(), x, y)
    r["t2"] = run("t2", lambda: PreLNNet(mk(2)), x, y)
    r["t10"] = run("t10", lambda: PreLNNet(mk(10)), x, y)
    print("summary (lower loss is better):", {k: round(v, 4) for k, v in r.items()})


if __name__ == "__main__":
    main()
