import sys
import os
import chess

sys.path.append(os.path.abspath("src"))

import torch
from Chess.components.Mcts import MCTS
from Chess.components.policy_value_torch_model import ValuePolicyNet
from Chess.config.configuration import ConfigurationManager

device = "cuda" if torch.cuda.is_available() else "cpu"

config_manager = ConfigurationManager()
base_config = config_manager.get_torch_base_model()
mcts_config = config_manager.get_mcts_config()

model = ValuePolicyNet(base_config).to(device)
model.load_state_dict(torch.load("artifacts/prepare_torch_callbacks/rl_trained_model.pth", map_location=device))
model.eval()

mcts = MCTS(model=model, device=device, config=mcts_config)

board = chess.Board()

print("=== بازی انسان (سفید) با مدل RL (سیاه) ===")
print("فرمت حرکت‌ها به صورت uci است (مثلاً e2e4 یا g1f3)\n")
print(board)

while not board.is_game_over():
    if board.turn == chess.WHITE:
        # نوبت شما
        user_move = input("\nحرکت شما (مثلاً e2e4): ").strip()
        try:
            move = chess.Move.from_uci(user_move)
            if move in board.legal_moves:
                board.push(move)
            else:
                print("حرکت غیرقانونی است! دوباره تلاش کنید.")
                continue
        except ValueError:
            print("فرمت حرکت اشتباه است!")
            continue
    else:
        # نوبت مدل RL
        print("\nمدل در حال فکر کردن...")
        best_move, _ = mcts.get_best_move(board)
        print(f"حرکت مدل RL: {best_move.uci()}")
        board.push(best_move)

    print("\n" + str(board))

print("\nپایان بازی! نتیجه:", board.result())