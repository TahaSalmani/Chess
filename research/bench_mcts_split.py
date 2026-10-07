"""Split per-simulation cost into python tree overhead vs neural-net forward."""
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import torch

from Chess.components.Mcts import MCTS, MCTSNode
from Chess.components.policy_value_torch_model import ValuePolicyNet
from Chess.config.configuration import ConfigurationManager

import chess

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
cm = ConfigurationManager()
model = ValuePolicyNet(cm.get_torch_base_model()).to(device)
model.eval()
rl = cm.get_rl_config()


class StubNet(torch.nn.Module):
    """Returns fixed policy/value instantly so only the python tree work is timed."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        n = x.shape[0]
        return torch.zeros(n, self.size, device=x.device), torch.zeros(n, 1, device=x.device)


cfg = dataclasses.replace(rl, params_num_simulation=32)
boards = [chess.Board() for _ in range(64)]

for pool in (1, 4, 8, 16, 32, 64):
    row = boards[:pool]
    results = {}
    for name, net in (("stub", StubNet(4096)), ("real", model)):
        mcts = MCTS(model=net, device=device, config=cfg)
        roots = [MCTSNode(board=b.copy()) for b in row]
        t0 = time.time()
        mcts.search_batch(roots)
        dt = time.time() - t0
        results[name] = dt
    sims = 32 * pool
    py, full = results["stub"], results["real"]
    print(f"pool={pool:>3} | total {full:6.2f}s ({full / sims * 1000:6.2f} ms/root-sim) | "
          f"python {py:6.2f}s ({py / sims * 1000:6.2f} ms) | "
          f"nn {full - py:6.2f}s ({(full - py) / 32 * 1000:7.2f} ms/forward) | "
          f"python share {py / full * 100:5.1f}%")
