import chess
import numpy as np
import torch

from Chess import logger
from Chess.components.Mcts import MCTS


def play_one_game(model, device, num_simulations=100, max_moves=500):

    board = chess.Board()
    mcts = MCTS(model=model, device=device, num_simulations=num_simulations)

    game_history = []
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        root = mcts.run(board)

        visit_counts = np.zeros(4096, dtype=np.float32)
        for move, child in root.children.items():
            idx = move.from_square * 64 + move.to_square
            visit_counts[idx] = child.visit_count

        total_visits = visit_counts.sum()
        if total_visits == 0:
            logger.info("No children explored, stopping game early.")
            break

        policy_target = visit_counts / total_visits

        board_matrix = mcts.board_to_tensor(board).squeeze(0).cpu().numpy()
        game_history.append((board_matrix, policy_target, board.turn))

        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        board.push(best_move)

        move_count += 1
        logger.info(f"Move {move_count}: {best_move}")

    if board.is_checkmate():
        result = 1.0 if board.turn == chess.BLACK else -1.0
    else:
        result = 0.0

    logger.info(f"Game finished after {move_count} moves. Result (White perspective): {result}")

    training_data = []
    for board_matrix, policy_target, player in game_history:
        value_target = result if player == chess.WHITE else -result
        training_data.append((board_matrix, policy_target, value_target))

    return training_data
def generate_self_play_data(model, device, num_games=5, num_simulations=50, max_moves=500):

    all_data = []

    for game_idx in range(num_games):
        logger.info(f"--- Starting self-play game {game_idx + 1}/{num_games} ---")
        game_data = play_one_game(
            model=model,
            device=device,
            num_simulations=num_simulations,
            max_moves=max_moves
        )
        all_data.extend(game_data)
        logger.info(f"Game {game_idx + 1} finished. Positions so far: {len(all_data)}")

    return all_data


def save_self_play_data(data, save_dir):

    import os

    os.makedirs(save_dir, exist_ok=True)

    boards = np.array([d[0] for d in data], dtype=np.float32)
    policies = np.array([d[1] for d in data], dtype=np.float32)
    values = np.array([d[2] for d in data], dtype=np.float32)

    x_tensor = torch.tensor(boards)
    policy_tensor = torch.tensor(policies)
    value_tensor = torch.tensor(values)

    torch.save(x_tensor, os.path.join(save_dir, "selfplay_x.pt"))
    torch.save(policy_tensor, os.path.join(save_dir, "selfplay_policy.pt"))
    torch.save(value_tensor, os.path.join(save_dir, "selfplay_value.pt"))

    logger.info(f"Self-play data saved to {save_dir} | Total positions: {len(data)}")
