"""Does the Pre-LN'd ChessPolicyNet keep improving past the marginal floor?

Uses the real repo classes (torch_base_model.ChessPolicyNet) with the same
OneCycleLR warmup that stage 09 now uses. Depth is dropped to 2 so the run fits
in a few CPU minutes; the question is the trend, not the final number.
"""
import sys
import time

import torch
from torch import nn

from Chess.config.configuration import ConfigurationManager
from Chess.components.torch_base_model import ChessPolicyNet

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
BATCH = 64
LR = 3e-4


def main():
    base = ConfigurationManager().get_torch_base_model()
    cfg = type(base)(**{**base.__dict__, "params_NUM_LAYERS": LAYERS})

    x = torch.load("artifacts/torch_dataset/x_tensor.pt", map_location="cpu")
    y = torch.load("artifacts/torch_dataset/y_tensor.pt", map_location="cpu").long()

    torch.manual_seed(0)
    net = ChessPolicyNet(config=cfg)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    ce = nn.CrossEntropyLoss()

    held_x, held_y = x[-4096:], y[-4096:]
    print(f"{len(x)-4096} train positions | {LAYERS} layers | warmup OneCycleLR | {STEPS} steps "
          f"= {STEPS*BATCH/(len(x)-4096):.2f} epochs")
    print("floor: marginal 6.6090 / top-1 1.26%   uniform: 8.3178 / 0.02%\n", flush=True)

    t0 = time.time()
    losses = []
    for s in range(STEPS):
        if s % 250 == 0:
            net.eval()
            with torch.no_grad():
                logits = net(held_x[:2048])
                pred = logits.argmax(1)
                acc = (pred == held_y[:2048]).float().mean().item()
                loss = sum(losses[-250:]) / max(len(losses[-250:]), 1)
            print(f"  step {s:5d} | train loss {loss:7.4f} | held-out top-1 {acc*100:6.2f}% "
                  f"| distinct {len(torch.unique(pred)):4d} | lr {sched.get_last_lr()[0]:.2e} "
                  f"| {time.time()-t0:.0f}s", flush=True)
            net.train()
        i = torch.randint(0, len(x) - 4096, (BATCH,))
        opt.zero_grad()
        loss = ce(net(x[i]), y[i])
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())


if __name__ == "__main__":
    main()
