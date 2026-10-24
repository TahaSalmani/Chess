import os
import time

import chess
import numpy as np
import torch

from Chess import logger
from Chess.components.Mcts import MCTS, MCTSNode, board_to_array
from Chess.entity.config_entity import PrepareRlModelConfig


class SelfPlayGame:
    __slots__ = ("board", "history", "move_count", "resigned_by", "bad_streak")

    def __init__(self):
        self.board = chess.Board()
        self.history = []
        self.move_count = 0
        self.resigned_by = None
        self.bad_streak = 0

    def is_done(self, max_moves: int) -> bool:
        return (self.resigned_by is not None
                or self.board.is_game_over()
                or self.move_count >= max_moves)

    def note_root_value(self, value: float, threshold: float, window: int, min_moves: int) -> bool:
        """Track the search's own evaluation and resign once it stays hopeless.

        `value` is the root value from the side-to-move's perspective (the sign
        MCTS backpropagation leaves at the root). Without this, two weak copies
        of the same net shuffle pieces until the move cap and every game scores
        0.0, so the value head only ever sees draw targets. Resigning converts
        those dead games into +/-1 signal. The streak + min_moves guards stop a
        still-random value head from resigning won or equal positions early.
        Returns True the moment the game is marked as resigned.
        """
        if self.resigned_by is not None:
            return True
        if self.move_count < int(min_moves):
            return False
        if value < -float(threshold):
            self.bad_streak += 1
        else:
            self.bad_streak = 0
        if self.bad_streak >= int(window):
            self.resigned_by = self.board.turn
            return True
        return False

    def finish_reason(self) -> str:
        if self.resigned_by is not None:
            return "resign"
        if self.board.is_checkmate():
            return "checkmate"
        if self.board.is_game_over():
            return "draw"
        return "move-cap"

    def result(self) -> float:
        if self.resigned_by is not None:
            return -1.0 if self.resigned_by == chess.WHITE else 1.0
        if self.board.is_checkmate():
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        return 0.0

    def training_data(self):
        result = self.result()
        return [
            (position, policy, result if player == chess.WHITE else -result)
            for position, policy, player in self.history
        ]


def generate_self_play_data(model, device, config: PrepareRlModelConfig):
    """Play `num_games` games with several of them searched in lockstep.

    Games sharing a slot count means every MCTS simulation evaluates them all
    in one batched forward pass instead of one forward per game.
    """
    mcts = MCTS(model=model, device=device, config=config)
    num_games = int(config.params_num_games)
    max_moves = int(config.params_max_moves)
    pool_size = max(1, min(int(config.params_pool_size), num_games))
    resign_threshold = float(getattr(config, "params_resign_threshold", 0.8))
    resign_window = int(getattr(config, "params_resign_window", 3))
    resign_min_moves = int(getattr(config, "params_resign_min_moves", 30))

    logger.info(
        f"Self-play: {num_games} games | {config.params_num_simulation} simulations/move | "
        f"max {max_moves} moves | {pool_size} games batched per forward | "
        f"resign at value < -{resign_threshold} for {resign_window} moves after move {resign_min_moves}"
    )

    slots = [SelfPlayGame() for _ in range(pool_size)]
    started = pool_size
    finished = 0
    rounds = 0
    all_data = []
    started_at = time.time()

    while finished < num_games:
        active = [i for i, game in enumerate(slots) if game is not None]
        if not active:
            break

        roots = [MCTSNode(board=slots[i].board.copy()) for i in active]
        mcts.search_batch(
            roots,
            dirichlet_alpha=float(config.params_dirichlet_alpha),
            dirichlet_epsilon=float(config.params_dirichlet_epsilon),
        )
        rounds += 1

        for i, root in zip(active, roots):
            game = slots[i]
            # the root value is the search's own verdict on the side to move;
            # a persistently hopeless verdict ends the game with a real +/-1
            resigned = game.note_root_value(
                root.value(), resign_threshold, resign_window, resign_min_moves
            )
            counts = MCTS.visit_counts(root)
            total = counts.sum()
            # early moves sampled by visit-count temperature for diverse data,
            # later moves played as argmax to convert advantages cleanly
            temp = 1.0 if game.move_count < int(config.params_temp_moves) else 0.0
            move = MCTS.sample_move(root, temp)

            if resigned:
                pass  # no move played; the game finalises below with its resign result
            elif total > 0 and move is not None:
                game.history.append((board_to_array(game.board), counts / total, game.board.turn))
                game.board.push(move)
                game.move_count += 1
            else:
                logger.warning(f"Slot {i} produced no visits, retiring that game.")

            if not game.is_done(max_moves):
                continue

            all_data.extend(game.training_data())
            finished += 1
            logger.info(
                f"Game {finished}/{num_games} done | {game.move_count} moves | "
                f"result (White) {game.result():+.1f} ({game.finish_reason()}) | positions {len(all_data)}"
            )

            if started < num_games:
                slots[i] = SelfPlayGame()
                started += 1
            else:
                slots[i] = None

        if rounds % 10 == 0:
            elapsed = time.time() - started_at
            progress = min(1.0, rounds / max_moves)
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0.0
            live = sum(1 for g in slots if g is not None)
            logger.info(
                f"round {rounds}/{max_moves} | in flight {live} | finished {finished}/{num_games} "
                f"({finished / num_games * 100:.0f}%) | positions {len(all_data)} | "
                f"forwards {mcts.forward_calls} x{mcts.positions_evaluated // max(mcts.forward_calls, 1)} | "
                f"elapsed {elapsed / 60:.1f}m | ETA {eta / 60:.1f}m"
            )

    logger.info(
        f"Self-play finished: {finished} games, {len(all_data)} positions, "
        f"{mcts.forward_calls} forward calls for {mcts.positions_evaluated} evaluations "
        f"in {(time.time() - started_at) / 60:.1f} min"
    )
    return all_data


def save_self_play_data(data, config: PrepareRlModelConfig):
    save_dir = str(config.self_play_data)
    os.makedirs(save_dir, exist_ok=True)

    x_tensor = torch.from_numpy(np.stack([d[0] for d in data]))
    policy_tensor = torch.from_numpy(np.stack([d[1] for d in data]))
    value_tensor = torch.from_numpy(np.asarray([d[2] for d in data], dtype=np.float32))

    torch.save(x_tensor, os.path.join(save_dir, "selfplay_x.pt"))
    torch.save(policy_tensor, os.path.join(save_dir, "selfplay_policy.pt"))
    torch.save(value_tensor, os.path.join(save_dir, "selfplay_value.pt"))

    logger.info(f"Self-play data saved to {save_dir} | Total positions: {len(data)}")
