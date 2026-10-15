"""Play a terminal chess game against the trained policy guided by search.

The agent is the supervised policy network used only to order/tie-break moves,
wrapped in the full-width material negamax + capture-quiescence search from
arena.py (PolicySearchAgent). This is the project's usable "player": the raw
policy argmax is far too weak to play directly (see research/README.md).

Usage (from the repo root, after stages 09 have produced weights):
    python research/play.py                 # you play White, agent plays Black
    python research/play.py --color black   # you play Black
    python research/play.py --depth 3       # stronger but slower search

Moves are entered in SAN (e4, Nf3, O-O) or UCI (e2e4). Type 'quit' to exit.
"""
import argparse
import os
import sys

import chess
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from Chess.config.configuration import ConfigurationManager
from Chess.components.torch_base_model import ChessPolicyNet
from arena import PolicySearchAgent


def load_agent(device, depth, qdepth):
    cm = ConfigurationManager()
    base_cfg = cm.get_torch_base_model()
    training_cfg = cm.get_torch_training_config()
    from pathlib import Path

    best = Path(training_cfg.trained_model_path).with_name("best_model.pth")
    path = best if best.exists() else Path(training_cfg.trained_model_path)
    if not path.exists():
        sys.exit(f"ERROR: no supervised weights at {path}; run stage 09 first.")
    net = ChessPolicyNet(base_cfg).to(device)
    net.load_state_dict(torch.load(str(path), map_location=device))
    return PolicySearchAgent(net, device, "policy-search", top_k=0, depth=depth, qdepth=qdepth)


def read_human_move(board):
    while True:
        raw = input("your move (SAN/UCI, 'quit' to exit): ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            sys.exit("game abandoned")
        move = None
        try:
            move = board.parse_san(raw)
        except ValueError:
            try:
                move = chess.Move.from_uci(raw)
            except ValueError:
                move = None
        if move is not None and move in board.legal_moves:
            return move
        print("illegal or unparseable move; try again (e.g. e4, Nf3, O-O, e2e4)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", choices=["white", "black"], default="white", help="the human's color")
    parser.add_argument("--depth", type=int, default=2, help="full-width negamax plies for the agent")
    parser.add_argument("--qdepth", type=int, default=4, help="capture-quiescence plies at the leaves")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(device, args.depth, args.qdepth)
    board = chess.Board()
    human_white = args.color == "white"
    print(f"agent plays {'Black' if human_white else 'White'} (search depth {args.depth}+q{args.qdepth}) on {device}")

    while not board.is_game_over():
        if (board.turn == chess.WHITE) == human_white:
            print(board)
            move = read_human_move(board)
        else:
            move = agent.choose(board)
            print(f"agent plays {board.san(move)}")
        board.push(move)

    print(board)
    print("result:", board.result(), "|", board.outcome().termination if board.outcome() else "")


if __name__ == "__main__":
    main()
