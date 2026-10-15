"""Play trained models against simple opponents to judge real playing strength.

Move-prediction accuracy (top-1) only says how human-like the policy is; it does
not say whether the model can actually win a game. This arena plays the models
against a random mover and a 1-ply material-greedy mover and reports results.

Usage (from the repo root, e.g. inside the Kaggle notebook):
    python research/arena.py --games 20 --mcts-sims 50

Artifacts read (created by stages 09 and 10):
    artifacts/prepare_torch_callbacks/best_model.pth   (supervised policy)
    artifacts/rl_model/rl_trained_model.pth            (RL value+policy)
"""
import argparse
import os
import random
import sys
from dataclasses import replace
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from Chess.config.configuration import ConfigurationManager
from Chess.components.torch_base_model import ChessPolicyNet
from Chess.components.policy_value_torch_model import ValuePolicyNet
from Chess.components.Mcts import MCTS, board_to_array

PIECE_VALUE = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}
MAX_PLIES = 200  # 100 moves each side; longer games are scored as draws


def _legal_scored(board, logits):
    """Return the legal move with the highest policy logit (handles promotion)."""
    legal = list(board.legal_moves)
    scores = [logits[m.from_square * 64 + m.to_square] for m in legal]
    return legal[int(np.argmax(scores))]


class PolicyAgent:
    """Plays the argmax legal move under the network policy. Works for both the
    supervised ChessPolicyNet (logits only) and the RL ValuePolicyNet (tuple)."""

    def __init__(self, model, device, name):
        self.model = model
        self.device = device
        self.name = name
        self.model.eval()

    def choose(self, board):
        x = torch.from_numpy(board_to_array(board)[None]).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        logits = (out[0] if isinstance(out, tuple) else out)[0].cpu().numpy()
        return _legal_scored(board, logits)


class PolicySearchAgent:
    """Policy-guided tactical search: the network supplies a short list of root
    candidate moves (top-K by logit) and a tie-break, while a capture-quiescence
    search with alpha-beta and incremental material scoring decides which one is
    safe. Unlike a fixed 2-ply lookahead this follows capture/recapture chains to
    a quiet position, so trades and hanging pieces are evaluated correctly.
    """

    MATE = 100000.0

    def __init__(self, model, device, name, top_k=6, qdepth=4):
        self.model = model
        self.device = device
        self.name = name
        self.top_k = int(top_k)
        self.qdepth = int(qdepth)
        self.model.eval()

    def _logits(self, board):
        x = torch.from_numpy(board_to_array(board)[None]).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        return (out[0] if isinstance(out, tuple) else out)[0].cpu().numpy()

    @staticmethod
    def _material(board):
        mat = 0
        for p in board.piece_map().values():
            v = PIECE_VALUE.get(p.piece_type, 0)
            mat += v if p.color == chess.WHITE else -v
        return mat

    def _qsearch(self, board, mat, alpha, beta, depth):
        """Negamax over captures only; returns score for the side to move."""
        stand = mat if board.turn == chess.WHITE else -mat
        if depth == 0:
            return stand
        best = stand
        for move in board.legal_moves:
            target = board.piece_at(move.to_square)
            if target is None:
                continue  # tactical search: captures only
            v = PIECE_VALUE.get(target.piece_type, 0)
            newmat = mat - v if target.color == chess.WHITE else mat + v
            board.push(move)
            if board.is_checkmate():
                child = -self.MATE  # side to move in child is mated
            else:
                child = self._qsearch(board, newmat, -beta, -alpha, depth - 1)
            board.pop()
            val = -child
            if val > best:
                best = val
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
        return best

    def choose(self, board):
        logits = self._logits(board)
        legal = list(board.legal_moves)
        scores = [logits[m.from_square * 64 + m.to_square] for m in legal]
        order = sorted(range(len(legal)), key=lambda i: scores[i], reverse=True)
        candidates = [legal[i] for i in order[: self.top_k]]
        mat0 = self._material(board)

        best = None
        best_key = None
        for rank, move in enumerate(candidates):
            target = board.piece_at(move.to_square)
            v = PIECE_VALUE.get(target.piece_type, 0) if target else 0
            newmat = mat0 - v if (target and target.color == chess.WHITE) else mat0 + v
            board.push(move)
            if board.is_checkmate():
                score = self.MATE
            elif board.is_stalemate():
                score = -0.5  # do not stalemate a winning position
            else:
                # child is from the opponent's perspective; flip to ours
                score = -self._qsearch(board, newmat, -self.MATE, self.MATE, self.qdepth)
            board.pop()
            # prefer higher search score; break ties toward the stronger policy move
            key = (score, -rank)
            if best_key is None or key > best_key:
                best_key = key
                best = move
        return best


class MctsAgent:
    """Plays via PUCT search using the value+policy head. Stronger than the raw
    policy argmax and is how the RL model is actually meant to be used."""

    def __init__(self, model, device, rl_config, sims, name):
        self.name = name
        cfg = replace(rl_config, params_num_simulation=int(sims))
        self.mcts = MCTS(model=model, device=device, config=cfg)

    def choose(self, board):
        move, _ = self.mcts.get_best_move(board)
        if move is None:  # search produced nothing (should not happen with legal moves)
            return random.choice(list(board.legal_moves))
        return move


class RandomOpponent:
    name = "random"

    def choose(self, board):
        return random.choice(list(board.legal_moves))


class GreedyOpponent:
    """1-ply material greedy: take the capture/move with the best immediate
    material delta, always take a mate if one exists, random among ties."""
    name = "greedy-material"

    def choose(self, board):
        best_score = -1e9
        best_moves = []
        for move in board.legal_moves:
            captured = board.piece_at(move.to_square)
            cap_val = PIECE_VALUE.get(captured.piece_type, 0) if captured else 0.0
            board.push(move)
            if board.is_checkmate():
                score = 1e6
            else:
                score = float(cap_val)
                landed = board.piece_at(move.to_square)
                # board.turn is now the opponent: if they attack our landed piece
                # it is hanging, so subtract what we would lose
                if landed is not None and board.is_attacked_by(board.turn, move.to_square):
                    score -= PIECE_VALUE.get(landed.piece_type, 0)
            board.pop()
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        return random.choice(best_moves)


class MinimaxOpponent:
    """Full-width material minimax with alpha-beta at a fixed depth. A stricter
    sparring partner than the 1-ply greedy mover: it sees one extra ply, so it
    both takes free material and avoids hanging its own pieces. Used to gauge
    the model's strength against something closer to a real beginner.
    """

    def __init__(self, depth=2):
        self.name = f"minimax-d{depth}"
        self.depth = int(depth)

    def _mat(self, board):
        mat = 0
        for p in board.piece_map().values():
            v = PIECE_VALUE.get(p.piece_type, 0)
            mat += v if p.color == chess.WHITE else -v
        return mat

    def _negamax(self, board, depth, alpha, beta):
        if board.is_checkmate():
            return -PolicySearchAgent.MATE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        if depth == 0:
            m = self._mat(board)
            return float(m) if board.turn == chess.WHITE else float(-m)
        best = -1e18
        for move in board.legal_moves:
            board.push(move)
            val = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if val > best:
                best = val
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
        return best

    def choose(self, board):
        best_score = -1e18
        best_moves = []
        for move in board.legal_moves:
            board.push(move)
            val = -self._negamax(board, self.depth - 1, -1e18, 1e18)
            board.pop()
            if val > best_score:
                best_score = val
                best_moves = [move]
            elif val == best_score:
                best_moves.append(move)
        return random.choice(best_moves)


def play_game(white, black, max_plies=MAX_PLIES):
    """Returns (result_from_white, plies, reason). result: +1 white, -1 black, 0 draw."""
    board = chess.Board()
    plies = 0
    while not board.is_game_over() and plies < max_plies:
        agent = white if board.turn == chess.WHITE else black
        move = agent.choose(board)
        if move not in board.legal_moves:
            move = random.choice(list(board.legal_moves))
        board.push(move)
        plies += 1

    if board.is_checkmate():
        return (1.0 if board.turn == chess.BLACK else -1.0), plies, "checkmate"
    if board.is_stalemate():
        return 0.0, plies, "stalemate"
    if board.is_insufficient_material():
        return 0.0, plies, "insufficient-material"
    if board.is_seventyfive_moves():
        return 0.0, plies, "75-move"
    if board.is_fivefold_repetition():
        return 0.0, plies, "repetition"
    return 0.0, plies, "ply-cap"


def run_match(agent, opponent, games, log):
    wins = draws = losses = 0
    total_plies = 0
    for i in range(games):
        # alternate colors so neither side is favoured
        if i % 2 == 0:
            white, black, agent_is_white = agent, opponent, True
        else:
            white, black, agent_is_white = opponent, agent, False
        result, plies, reason = play_game(white, black)
        total_plies += plies
        agent_result = result if agent_is_white else -result
        if agent_result > 0:
            wins += 1
        elif agent_result < 0:
            losses += 1
        else:
            draws += 1
        log(f"  game {i + 1}/{games} | agent={'W' if agent_is_white else 'B'} | "
            f"{'win' if agent_result > 0 else 'loss' if agent_result < 0 else 'draw'} "
            f"({reason}, {plies} plies)")
    score = (wins + 0.5 * draws) / games
    return wins, draws, losses, score, total_plies / games


def report(label, stats, log):
    wins, draws, losses, score, avg_plies = stats
    log(f"[{label}]  W {wins} / D {draws} / L {losses}  -> score {score * 100:.0f}%  "
        f"(avg {avg_plies:.0f} plies/game)")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20, help="games per matchup")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per move for the RL agent")
    parser.add_argument("--no-mcts", action="store_true", help="play the RL policy argmax instead of MCTS (much faster, weaker)")
    parser.add_argument("--search", action="store_true", help="also play the supervised policy through a 2-ply material search (policy-guided)")
    parser.add_argument("--search-k", type=int, default=6, help="top-K policy candidates for --search")
    parser.add_argument("--search-depth", type=int, default=4, help="capture-quiescence depth (plies of captures) for --search")
    parser.add_argument("--hard-opp", action="store_true", help="also play a depth-2 material minimax opponent for a stricter strength read")
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
    rl_cfg = cm.get_rl_config()

    best = Path(training_cfg.trained_model_path).with_name("best_model.pth")
    supervised_path = best if best.exists() else Path(training_cfg.trained_model_path)
    rl_path = Path(rl_cfg.root_dir) / "rl_trained_model.pth"

    log(f"supervised weights: {supervised_path} (exists={supervised_path.exists()})")
    log(f"rl weights:         {rl_path} (exists={rl_path.exists()})")
    if not supervised_path.exists() and not rl_path.exists():
        log("ERROR: no model weights found; run stages 09/10 first.")
        sys.exit(2)

    opponents = [RandomOpponent(), GreedyOpponent()]
    if args.hard_opp:
        opponents.append(MinimaxOpponent(depth=2))
    results = {}

    # --- supervised policy model (policy argmax; no value head, so no MCTS) ---
    if supervised_path.exists():
        net = ChessPolicyNet(base_cfg).to(device)
        net.load_state_dict(torch.load(str(supervised_path), map_location=device))
        agent = PolicyAgent(net, device, "supervised-policy")
        log(f"\n=== {agent.name} ===")
        for opp in opponents:
            log(f"vs {opp.name}:")
            stats = run_match(agent, opp, args.games, log)
            results[(agent.name, opp.name)] = report(f"{agent.name} vs {opp.name}", stats, log)

        if args.search:
            sagent = PolicySearchAgent(net, device, f"supervised-search(k{args.search_k},d{args.search_depth})", top_k=args.search_k, qdepth=args.search_depth)
            log(f"\n=== {sagent.name} ===")
            for opp in opponents:
                log(f"vs {opp.name}:")
                stats = run_match(sagent, opp, args.games, log)
                results[(sagent.name, opp.name)] = report(f"{sagent.name} vs {opp.name}", stats, log)

    # --- RL model ---
    if rl_path.exists():
        rl_net = ValuePolicyNet(base_cfg).to(device)
        rl_net.load_state_dict(torch.load(str(rl_path), map_location=device))
        if args.no_mcts:
            agent = PolicyAgent(rl_net, device, "rl-policy")
        else:
            agent = MctsAgent(rl_net, device, rl_cfg, args.mcts_sims, f"rl-mcts({args.mcts_sims})")
        log(f"\n=== {agent.name} ===")
        for opp in opponents:
            log(f"vs {opp.name}:")
            stats = run_match(agent, opp, args.games, log)
            results[(agent.name, opp.name)] = report(f"{agent.name} vs {opp.name}", stats, log)

    # --- verdict ---
    log("\n================ VERDICT ================")
    random_scores = [s for (a, o), s in results.items() if o == "random"]
    greedy_scores = [s for (a, o), s in results.items() if o == "greedy-material"]
    for (a, o), s in results.items():
        log(f"  {a:22s} vs {o:16s} : {s * 100:5.1f}%")

    if random_scores:
        best_random = max(random_scores)
        if best_random >= 0.9:
            log(f"\nPASS: beats random at {best_random * 100:.0f}% -- the model plays legal, purposeful chess.")
        elif best_random >= 0.6:
            log(f"\nWEAK: only {best_random * 100:.0f}% vs random -- plays but blunders often.")
        else:
            log(f"\nFAIL: {best_random * 100:.0f}% vs random -- not trustworthy yet.")
    if greedy_scores:
        best_greedy = max(greedy_scores)
        log(f"Best score vs material-greedy baseline: {best_greedy * 100:.0f}% "
            f"({'above' if best_greedy > 0.5 else 'at/below'} 50% = the bar for a usable club-level prior).")


if __name__ == "__main__":
    main()
