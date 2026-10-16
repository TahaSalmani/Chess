"""Policy distillation: teach the network to play the SEARCH's move, not humans'.

The supervised policy imitates human moves and therefore blunders under argmax
(~0% vs a depth-2 engine). But we already have a strong teacher: the
policy+search player from arena.py. This script runs one round of policy
iteration -- label positions with the teacher search's chosen move, then
fine-tune the policy on those labels -- which is the AlphaZero idea of using
search output as the policy target, with a classical search standing in for
MCTS because the value head is not yet trustworthy.

Effect to look for afterwards: `supervised-policy vs minimax-d2` in arena.py
should jump well above ~0%, because the distilled policy no longer hangs pieces
even without search.

Usage (from the repo root, in a live session that already has stage-09 weights
and dataset splits, e.g. the Kaggle notebook):
    python research/distill.py --n 60000 --depth 1 --qdepth 3 --epochs 2
"""
import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from Chess.config.configuration import ConfigurationManager
from Chess.components.torch_base_model import ChessPolicyNet
from arena import PolicySearchAgent


def find_splits():
    root = Path("artifacts")
    xs = sorted(root.rglob("X_train.npy"))
    ys = sorted(root.rglob("y_train.npy"))
    if not xs or not ys:
        sys.exit("ERROR: X_train.npy / y_train.npy not found under artifacts/; run stages 02/03 first.")
    return xs[0], ys[0]


def array_to_board(arr) -> chess.Board:
    """Rebuild a Board from the 17-plane encoding (en-passant is not encoded)."""
    rows = []
    for row in range(7, -1, -1):  # FEN starts at rank 8 = row index 7
        empties = 0
        cells = []
        for col in range(8):
            chans = arr[row, col, :12]
            idx = int(np.flatnonzero(chans)[0]) if chans.any() else -1
            if idx < 0:
                empties += 1
                continue
            if empties:
                cells.append(str(empties))
                empties = 0
            piece_type = (idx % 6) + 1
            letter = "pnbrqk"[piece_type - 1]
            cells.append(letter.upper() if idx < 6 else letter)
        if empties:
            cells.append(str(empties))
        rows.append("".join(cells))
    turn = "w" if arr[:, :, 12].mean() > 0.5 else "b"
    castling = ""
    for ch, plane in (("K", 13), ("Q", 14), ("k", 15), ("q", 16)):
        if arr[0, 0, plane] > 0.5:
            castling += ch
    fen = f"{'/'.join(rows)} {turn} {castling or '-'} - 0 1"
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    return board if board.is_valid() and board.legal_moves.count() > 0 else None


def label_positions(x_mmap, indices, teacher, log):
    ys = []
    kept = []
    t0 = time.time()
    for n, i in enumerate(indices):
        board = array_to_board(np.asarray(x_mmap[i], dtype=np.uint8))
        if board is None:
            continue
        move = teacher.choose(board)
        kept.append(i)
        ys.append(move.from_square * 64 + move.to_square)
        if (n + 1) % 2000 == 0:
            rate = (n + 1) / (time.time() - t0)
            eta = (len(indices) - n - 1) / rate / 60
            log(f"  labeled {n + 1}/{len(indices)} ({rate:.1f}/s, ETA {eta:.1f} min)")
    return np.asarray(kept, dtype=np.int64), np.asarray(ys, dtype=np.int64)


def finetune(model, xs, ys, hx, hy, device, epochs, lr, batch, log):
    x_t = torch.from_numpy(xs).float()
    y_t = torch.from_numpy(ys).long()
    if hx is not None and len(hx):
        x_t = torch.cat([x_t, torch.from_numpy(hx).float()])
        y_t = torch.cat([y_t, torch.from_numpy(hy).long()])
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = lossfn(model(bx), by)
            loss.backward()
            opt.step()
            total += loss.item()
        log(f"  distill epoch {epoch + 1}/{epochs} | loss {total / len(loader):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60000, help="positions to label with the teacher search")
    parser.add_argument("--depth", type=int, default=1, help="teacher negamax plies (keep low: labeling is CPU-bound)")
    parser.add_argument("--qdepth", type=int, default=3, help="teacher quiescence plies")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--mix-human", type=float, default=0.3, help="fraction of human-labeled samples to mix in (retains opening diversity)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def log(msg):
        print(msg, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    cm = ConfigurationManager()
    base_cfg = cm.get_torch_base_model()
    training_cfg = cm.get_torch_training_config()
    best = Path(training_cfg.trained_model_path).with_name("best_model.pth")
    if not best.exists():
        best = Path(training_cfg.trained_model_path)
    if not best.exists():
        sys.exit("ERROR: no supervised weights; run stage 09 first.")

    x_path, y_path = find_splits()
    log(f"splits: {x_path}")
    x_mmap = np.load(x_path, mmap_mode="r")
    y_mmap = np.load(y_path, mmap_mode="r")
    total = len(x_mmap)

    net = ChessPolicyNet(base_cfg).to(device)
    net.load_state_dict(torch.load(str(best), map_location=device))
    teacher = PolicySearchAgent(net, device, "teacher", top_k=0, depth=args.depth, qdepth=args.qdepth)

    n = min(args.n, total)
    indices = np.random.choice(total, size=n, replace=False)
    log(f"labeling {n} positions with search(d{args.depth},q{args.qdepth}) ...")
    kept, ys = label_positions(x_mmap, indices, teacher, log)
    log(f"labeled {len(kept)} usable positions")

    xs = np.asarray(x_mmap[kept], dtype=np.uint8)
    xs = xs.astype(np.float32) / 1.0  # encoding is already 0/1

    hx = hy = None
    if args.mix_human > 0:
        m = int(len(kept) * args.mix_human)
        hi = np.random.choice(total, size=m, replace=False)
        hx = np.asarray(x_mmap[hi], dtype=np.float32)
        hy = np.asarray(y_mmap[hi], dtype=np.int64)
        log(f"mixing {m} human-labeled samples")

    backup = best.with_name("best_model_pre_distill.pth")
    if not backup.exists():
        shutil.copy(str(best), str(backup))
        log(f"backed up original weights to {backup}")

    log("fine-tuning policy on search labels ...")
    finetune(net, xs, ys, hx, hy, device, args.epochs, args.lr, args.batch, log)

    torch.save(net.state_dict(), str(best))
    log(f"distilled policy saved to {best}")
    log("now re-measure with: python research/arena.py --games 20 --search --hard-opp --no-mcts")
    log("(watch 'supervised-policy vs minimax-d2': it should jump well above ~0%)")


if __name__ == "__main__":
    main()
