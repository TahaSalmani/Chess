"""Verify batched MCTS correctness and measure the speedup vs sequential search."""
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.abspath("src"))

import numpy as np
import torch

from Chess.components.Mcts import MCTS, MCTSNode
from Chess.components.policy_value_torch_model import ValuePolicyNet
from Chess.components.self_play import generate_self_play_data, save_self_play_data
from Chess.config.configuration import ConfigurationManager

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

cm = ConfigurationManager()
base_config = cm.get_torch_base_model()
rl_config = cm.get_rl_config()
model = ValuePolicyNet(base_config).to(device)
model.eval()

print(f"model: {base_config.params_NUM_LAYERS} layers, d={base_config.params_D_MODEL}, "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")


def small(**over):
    return dataclasses.replace(
        rl_config, params_num_simulation=16, params_max_moves=10,
        params_num_games=4, params_pool_size=4, **over
    )


# ---------------------------------------------------------------- correctness
import chess

mcts = MCTS(model=model, device=device, config=small())

boards = [
    chess.Board(),
    chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    chess.Board("8/8/8/4k3/8/8/8/R3K2R w KQ - 0 1"),
]

solo = []
for b in boards:
    root = MCTSNode(board=b.copy())
    mcts.search_batch([root])
    solo.append(MCTS.visit_counts(root))

batched = [MCTSNode(board=b.copy()) for b in boards]
mcts.search_batch(batched)
paired = [MCTS.visit_counts(r) for r in batched]

print("\n--- independence of batched search ---")
ok = True
for i, (s, p) in enumerate(zip(solo, paired)):
    same_best = np.argmax(s) == np.argmax(p)
    max_diff = float(np.abs(s - p).max())
    print(f"board {i}: visits solo={s.sum():.0f} batched={p.sum():.0f} | "
          f"max|diff|={max_diff:.4f} | same best move={same_best}")
    ok = ok and same_best and max_diff < 1e-3
print("PASS" if ok else "FAIL")

# duplicate roots must behave like one root searched alone
dup = [MCTSNode(board=boards[0].copy()) for _ in range(4)]
mcts.search_batch(dup)
dup_counts = [MCTS.visit_counts(r) for r in dup]
same = all(np.abs(d - dup_counts[0]).max() < 1e-3 for d in dup_counts)
print(f"duplicate roots identical: {same} -> {'PASS' if same else 'FAIL'}")

# priors must only cover legal moves and sum to 1
root = MCTSNode(board=boards[0].copy())
mcts.search_batch([root])
legal = {mv.from_square * 64 + mv.to_square for mv in boards[0].legal_moves}
visited = {i for i in np.nonzero(MCTS.visit_counts(root))[0]}
print(f"visited subset of legal moves: {visited <= legal} | children={len(root.children)} legal={len(legal)}")
assert visited <= legal

# ------------------------------------------------------------------- self-play
print("\n--- self-play smoke run (4 games, 16 sims, max 10 moves, pool 4) ---")
cfg = small()
data = generate_self_play_data(model=model, device=device, config=cfg)
x = np.stack([d[0] for d in data])
pol = np.stack([d[1] for d in data])
val = np.asarray([d[2] for d in data])
print(f"positions={len(data)} x={x.shape} policy={pol.shape} value={val.shape}")
print(f"policy row sums: min={pol.sum(1).min():.6f} max={pol.sum(1).max():.6f}")
print(f"value range: [{val.min():+.2f}, {val.max():+.2f}]")
assert x.shape[1:] == (8, 8, 12)
assert pol.shape[1] == 4096
assert np.allclose(pol.sum(1), 1.0, atol=1e-5)
assert np.isin(val, [-1.0, 0.0, 1.0]).all()
save_self_play_data(data=data, config=cfg)
print("self-play PASS")

# ----------------------------------------------------------------- throughput
print("\n--- throughput: equal total work, sequential vs batched ---")
SIMS = 32
MAXM = 8
bench_cfg = dataclasses.replace(rl_config, params_num_simulation=SIMS, params_max_moves=MAXM)
bench_mcts = MCTS(model=model, device=device, config=bench_cfg)


def timed(pool):
    games = [chess.Board() for _ in range(pool)]
    t0 = time.time()
    rounds = 0
    for _ in range(MAXM):
        roots = [MCTSNode(board=g.copy()) for g in games]
        bench_mcts.search_batch(roots)
        rounds += 1
        for i, r in enumerate(roots):
            mv = MCTS.best_move(r)
            if mv is not None and not games[i].is_game_over():
                games[i].push(mv)
    dt = time.time() - t0
    moves = pool * rounds
    return dt, moves, bench_mcts.forward_calls, bench_mcts.positions_evaluated


for pool in (1, 4, 8, 16):
    bench_mcts.forward_calls = 0
    bench_mcts.positions_evaluated = 0
    dt, moves, calls, pos = timed(pool)
    print(f"pool={pool:>3} | {moves:>4} game-moves in {dt:6.2f}s | {moves / dt:6.2f} moves/s | "
          f"{calls} forwards x{pos // max(calls, 1):>2} | {dt / moves * 1000:7.1f} ms/move")
    if pool == 1:
        base = dt / moves
    else:
        print(f"{'':>9}   speedup vs pool=1: {base / (dt / moves):.2f}x")
