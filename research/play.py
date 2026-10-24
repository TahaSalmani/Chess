"""Play a terminal chess game against the trained policy guided by search.

The agent is the supervised policy network used only to order/tie-break moves,
wrapped in the full-width material negamax + capture-quiescence search from
arena.py (PolicySearchAgent). This is the project's usable "player": the raw
policy argmax is far too weak to play directly (see research/README.md).

Usage (from the repo root, after stages 09 have produced weights):
    python research/play.py                 # you play White, agent plays Black
    python research/play.py --color black   # you play Black
    python research/play.py --depth 3       # stronger but slower search
    python research/play.py --weights path/to/best_model.pth   # explicit weights

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


def load_agent(device, depth, qdepth, weights=None):
    from pathlib import Path

    if weights:
        path = Path(weights)
    else:
        cm = ConfigurationManager()
        training_cfg = cm.get_torch_training_config()
        best = Path(training_cfg.trained_model_path).with_name("best_model.pth")
        path = best if best.exists() else Path(training_cfg.trained_model_path)
    if not path.exists():
        sys.exit(f"ERROR: no supervised weights at {path}; pass --weights or run stage 09 first.")

    cm = ConfigurationManager()
    base_cfg = cm.get_torch_base_model()

    net = ChessPolicyNet(base_cfg).to(device)
    state = torch.load(str(path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # The checkpoint may have been trained with a different width/depth than the
    # current params.yaml; rebuild the net to match it so loading never fails.
    in_ch = state["input_proj.weight"].shape[1]
    n_layers = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
    if in_ch != base_cfg.params_IN_CHANNELS or n_layers != base_cfg.params_NUM_LAYERS:
        print(f"adapting net to checkpoint: IN_CHANNELS={in_ch}, NUM_LAYERS={n_layers} "
              f"(params.yaml had {base_cfg.params_IN_CHANNELS}/{base_cfg.params_NUM_LAYERS})")
        from dataclasses import replace
        base_cfg = replace(base_cfg, params_IN_CHANNELS=in_ch, params_NUM_LAYERS=n_layers)
        net = ChessPolicyNet(base_cfg).to(device)

    net.load_state_dict(state)
    net.eval()
    print(f"loaded weights from {path}")
    return PolicySearchAgent(net, device, "policy-search", top_k=0, depth=depth, qdepth=qdepth)


def show_board(board):
    print(board, "\n")


def san_history(board):
    b = chess.Board()
    sans = []
    for mv in board.move_stack:
        sans.append(b.san(mv))
        b.push(mv)
    return " ".join(sans)


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
    parser.add_argument("--weights", default=None, help="path to best_model.pth (defaults to the repo artifacts)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(device, args.depth, args.qdepth, weights=args.weights)
    board = chess.Board()
    human_white = args.color == "white"
    print(f"agent plays {'Black' if human_white else 'White'} (search depth {args.depth}+q{args.qdepth}) on {device}")
    print("enter moves in SAN (e4, Nf3, O-O) or UCI (e2e4); 'quit' to exit\n")

    while not board.is_game_over():
        if (board.turn == chess.WHITE) == human_white:
            show_board(board)
            move = read_human_move(board)
        else:
            move = agent.choose(board)
            print(f"agent plays {board.san(move)}")
        board.push(move)

    show_board(board)
    print("move history:", san_history(board))
    print("result:", board.result(), "|", board.outcome().termination if board.outcome() else "")


if __name__ == "__main__":
    main()
