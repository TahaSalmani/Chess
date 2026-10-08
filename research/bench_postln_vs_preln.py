"""Controlled test: does Post-LN vs Pre-LN explain the stage-09 training collapse?

Trains both block variants for the same number of steps on the same data with the
same lr, and compares the loss curve. Run from the repo root with PYTHONPATH=src.
"""
import sys
import time

import torch
from torch import nn

from Chess.config.configuration import ConfigurationManager
from Chess.components.torch_base_model import ChessPolicyNet, TransformerEncoderBlock

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BATCH = 64
LR = 3e-4
LAYERS = 10
SEED = 0


class PreLNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.params_D_MODEL,
            num_heads=config.params_NUM_HEADS,
            batch_first=True,
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
        attn_out, _ = self.attn(h, h, h)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        return x + self.dropout(self.ffn(h))


class PreLNNet(ChessPolicyNet):
    """ChessPolicyNet with Pre-LN blocks, a final LayerNorm, and a scaled pos embedding."""

    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([PreLNBlock(config) for _ in range(config.params_NUM_LAYERS)])
        self.final_norm = nn.LayerNorm(config.params_D_MODEL)
        with torch.no_grad():
            self.pos_embd.mul_(0.02)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, 64, -1)
        x = self.input_proj(x)
        x = x + self.pos_embd
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x).mean(dim=1)
        return self.out(self.relu(self.fc(x)))


def run(build, x, y, label):
    torch.manual_seed(SEED)
    net = build()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    n = len(x)
    t0 = time.time()
    losses = []
    for step in range(STEPS):
        i = torch.randint(0, n, (BATCH,))
        xb, yb = x[i], y[i]
        opt.zero_grad()
        out = net(xb)
        loss = ce(out, yb)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(net.parameters(), 1e9).item()
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == STEPS - 1:
            w = losses[-50:]
            print(
                f"  {label:8s} step {step:4d} | loss {sum(w)/len(w):7.4f} "
                f"| grad-norm {gn:9.3g} | {time.time()-t0:5.1f}s",
                flush=True,
            )
    net.eval()
    with torch.no_grad():
        idx = torch.arange(0, 4096)
        logits = net(x[idx])
        acc = (logits.argmax(1) == y[idx]).float().mean().item()
        distinct = len(torch.unique(logits.argmax(1)))
    print(f"  {label:8s} FINAL loss {sum(losses[-50:])/50:.4f} | top-1 {acc*100:.2f}% "
          f"| distinct labels {distinct}\n", flush=True)
    return sum(losses[-50:]) / 50


def main():
    cfg = ConfigurationManager().get_torch_base_model()
    cfg = type(cfg)(**{**cfg.__dict__, "params_NUM_LAYERS": LAYERS})

    x = torch.load("artifacts/torch_dataset/x_tensor.pt", map_location="cpu")
    y = torch.load("artifacts/torch_dataset/y_tensor.pt", map_location="cpu").long()
    print(f"data: {len(x)} positions | {LAYERS} layers | lr {LR} | batch {BATCH} | {STEPS} steps")
    print(f"uniform-guess loss = ln(4096) = {torch.log(torch.tensor(4096.0)).item():.4f}\n")

    post = run(lambda: ChessPolicyNet(config=cfg), x, y, "Post-LN")
    pre = run(lambda: PreLNNet(cfg), x, y, "Pre-LN")

    print(f"Post-LN {post:.4f}  vs  Pre-LN {pre:.4f}  ->  Pre-LN is {post-pre:+.4f} lower")


if __name__ == "__main__":
    main()
