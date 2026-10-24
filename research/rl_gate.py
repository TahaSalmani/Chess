"""Gate the RL candidate against the supervised champion before adopting it.

Plays the RL model (MCTS with its value+policy head) against the supervised
policy+search player -- the project's champion -- for `--games` games, colours
alternating. The candidate is ACCEPTED only if its score meets `--min-score`.
On acceptance the RL weights are copied to a *separate* promoted file; the
supervised `best_model.pth` is NEVER overwritten, so a rejected candidate leaves
the shipped player completely unchanged.

    python research/rl_gate.py --games 20 --min-score 0.55
    python research/rl_gate.py --games 8 --sims 50 --depth 2 --qdepth 4

The gate logic (evaluate/decide/gate) is pure and agent-injected so it can be
unit-tested with fake players -- see research/test_rl_gate.py.
"""
import argparse
import hashlib
import os
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import chess  # noqa: E402

from Chess.config.configuration import ConfigurationManager  # noqa: E402
from Chess.components.torch_base_model import ChessPolicyNet  # noqa: E402
from Chess.components.policy_value_torch_model import ValuePolicyNet  # noqa: E402
from Chess.pipeline.stage_10_Pytorch_RL import _arch_overrides, _load_state  # noqa: E402
from arena import PolicySearchAgent, MctsAgent, play_game  # noqa: E402


# ----------------------------- pure gate logic ----------------------------- #

def evaluate(challenger, champion, games, seed=0, max_plies=200, log=print):
    """Play `games` between challenger and champion, alternating colours.
    Returns (score, (wins, draws, losses)) from the challenger's perspective."""
    random.seed(seed)
    np.random.seed(seed)
    wins = draws = losses = 0
    for i in range(games):
        challenger_white = (i % 2 == 0)
        if challenger_white:
            white, black = challenger, champion
        else:
            white, black = champion, challenger
        result, plies, reason = play_game(white, black, max_plies=max_plies)
        cr = result if challenger_white else -result
        if cr > 0:
            wins += 1
        elif cr < 0:
            losses += 1
        else:
            draws += 1
        log(f"  game {i + 1}/{games} | challenger={'W' if challenger_white else 'B'} | "
            f"{'win' if cr > 0 else 'loss' if cr < 0 else 'draw'} ({reason}, {plies} plies)")
    score = (wins + 0.5 * draws) / games if games else 0.0
    return score, (wins, draws, losses)


def decide(score, min_score):
    """Acceptance criterion: the candidate must MEET OR BEAT min_score."""
    return bool(score >= float(min_score))


def _file_sha(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate(challenger, champion, games, min_score, seed=0, max_plies=200,
         rl_path=None, promote_to=None, supervised_path=None, log=print):
    """Run the match and apply the acceptance decision.

    On ACCEPT: copy rl_path -> promote_to (a separate file; the supervised
    checkpoint is never touched). On REJECT: change nothing.
    Returns a dict with the score, record, decision and whether it promoted.
    """
    supervised_before = _file_sha(supervised_path)
    score, record = evaluate(challenger, champion, games, seed=seed,
                             max_plies=max_plies, log=log)
    accepted = decide(score, min_score)
    promoted = False
    if accepted and rl_path and promote_to:
        os.makedirs(os.path.dirname(str(promote_to)) or ".", exist_ok=True)
        shutil.copyfile(str(rl_path), str(promote_to))
        promoted = True
        log(f"ACCEPTED: score {score * 100:.0f}% >= {min_score * 100:.0f}% "
            f"-> promoted {rl_path} to {promote_to}")
    else:
        log(f"REJECTED: score {score * 100:.0f}% < {min_score * 100:.0f}% "
            f"-> supervised player unchanged")

    supervised_after = _file_sha(supervised_path)
    assert supervised_before == supervised_after, "gate must never modify the supervised checkpoint"
    return {"score": score, "record": record, "accepted": accepted, "promoted": promoted}


# --------------------------- model wiring (CLI) ---------------------------- #

def _adapt(base_cfg, state):
    ov = _arch_overrides(state, base_cfg)
    return replace(base_cfg, **ov) if ov else base_cfg


def build_supervised_agent(path, base_cfg, device, depth, qdepth):
    state = _load_state(str(path), device)
    net = ChessPolicyNet(_adapt(base_cfg, state)).to(device)
    net.load_state_dict(state)
    net.eval()
    return PolicySearchAgent(net, device, "supervised-search", top_k=0, depth=depth, qdepth=qdepth)


def build_rl_agent(path, base_cfg, rl_cfg, device, sims):
    state = _load_state(str(path), device)
    net = ValuePolicyNet(_adapt(base_cfg, state)).to(device)
    net.load_state_dict(state)
    net.eval()
    return MctsAgent(net, device, rl_cfg, sims, "rl-mcts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.55,
                        help="candidate is adopted only if its score vs the champion >= this")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations/move for the RL agent")
    parser.add_argument("--depth", type=int, default=2, help="champion search plies")
    parser.add_argument("--qdepth", type=int, default=4, help="champion quiescence plies")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--supervised", default=None, help="path to best_model.pth (default: repo artifacts)")
    parser.add_argument("--rl", default=None, help="path to rl_trained_model.pth (default: repo artifacts)")
    parser.add_argument("--promote-to", default=None, help="destination for accepted RL weights")
    args = parser.parse_args()

    def log(m):
        print(m, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm = ConfigurationManager()
    base_cfg = cm.get_torch_base_model()
    training_cfg = cm.get_torch_training_config()
    rl_cfg = cm.get_rl_config()

    best = Path(training_cfg.trained_model_path).with_name("best_model.pth")
    sup_path = Path(args.supervised) if args.supervised else (best if best.exists() else Path(training_cfg.trained_model_path))
    rl_path = Path(args.rl) if args.rl else (Path(rl_cfg.root_dir) / "rl_trained_model.pth")
    promote_to = Path(args.promote_to) if args.promote_to else sup_path.with_name("promoted_model.pth")

    if not sup_path.exists():
        sys.exit(f"ERROR: supervised weights not found at {sup_path}")
    if not rl_path.exists():
        sys.exit(f"ERROR: RL candidate not found at {rl_path}; run stage 10 first.")

    log(f"supervised (champion): {sup_path}")
    log(f"rl candidate         : {rl_path}")
    log(f"promote-to (if accept): {promote_to}")
    log(f"device={device} games={args.games} min-score={args.min_score} sims={args.sims}")

    champion = build_supervised_agent(sup_path, base_cfg, device, args.depth, args.qdepth)
    challenger = build_rl_agent(rl_path, base_cfg, rl_cfg, device, args.sims)

    result = gate(challenger, champion, args.games, args.min_score, seed=args.seed,
                  max_plies=args.max_plies, rl_path=rl_path, promote_to=promote_to,
                  supervised_path=sup_path, log=log)

    log("\n===== GATE RESULT =====")
    w, d, l = result["record"]
    log(f"rl-mcts vs supervised-search: W {w} / D {d} / L {l} -> {result['score'] * 100:.0f}%")
    log(f"decision: {'ACCEPT (promoted)' if result['accepted'] else 'REJECT (supervised kept)'}")


if __name__ == "__main__":
    main()
