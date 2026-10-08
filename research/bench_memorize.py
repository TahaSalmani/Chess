"""Can a plain MLP memorize a few hundred (board -> move) pairs?

Binary question. If it drives loss toward ~0, the encoded boards do distinguish
positions and stage 03 is sound. If it stalls near the marginal entropy (6.61),
the input carries no usable signal and the bug is in the data encoding.
"""
import sys
import time

import torch
from torch import nn

N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800


class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 8 * 12, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4096),
        )

    def forward(self, x):
        return self.net(x)


def main():
    x = torch.load("artifacts/torch_dataset/x_tensor.pt", map_location="cpu")
    y = torch.load("artifacts/torch_dataset/y_tensor.pt", map_location="cpu").long()
    xs, ys = x[:N], y[:N]

    distinct_boards = len({tuple(b.flatten().tolist()) for b in xs})
    distinct_labels = len(torch.unique(ys))
    print(f"memorizing {N} positions | distinct boards {distinct_boards} | distinct labels {distinct_labels}")
    print(f"reference: ln(4096)=8.3178 (uniform), 6.6090 (predict the marginal)\n")

    torch.manual_seed(0)
    net = MLP()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()

    t0 = time.time()
    for s in range(STEPS + 1):
        opt.zero_grad()
        out = net(xs)
        loss = ce(out, ys)
        loss.backward()
        opt.step()
        if s % 100 == 0:
            with torch.no_grad():
                acc = (out.argmax(1) == ys).float().mean().item()
                print(f"  step {s:5d} | loss {loss.item():7.4f} | train top-1 {acc*100:6.2f}% "
                      f"| {time.time()-t0:.0f}s", flush=True)

    with torch.no_grad():
        acc = (net(xs).argmax(1) == ys).float().mean().item()
    print(f"\nfinal: loss {loss.item():.4f} | train top-1 {acc*100:.2f}%")
    print("VERDICT:", "input distinguishes positions -> data is sound"
          if acc > 0.5 else "input may NOT distinguish positions -> suspect the encoding")


if __name__ == "__main__":
    main()
