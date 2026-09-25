import json
import os
from pathlib import Path
from typing import Union

from Chess import logger
from Chess.entity.config_entity import (
    DataTransformationConfig,
    PrepareEvaluationConfig,
)
import chess
import chess.pgn as pgn
import numpy as np
from sklearn.model_selection import train_test_split


class DataTransformation:

    def __init__(
        self, config: Union[DataTransformationConfig, PrepareEvaluationConfig]
    ):
        self.config = config

    def board_to_matrix(self, board: chess.Board) -> np.ndarray:
        matrix = np.zeros((8, 8, 12), dtype=np.uint8)
        piece_map = board.piece_map()

        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            piece_type = piece.piece_type - 1
            color = 0 if piece.color else 6
            channel = piece_type + color
            matrix[row, col, channel] = 1

        return matrix

    def move_to_index(self, move: chess.Move) -> int:
        from_sq = move.from_square
        to_sq = move.to_square
        return (from_sq * 64) + to_sq

    def extract_dataset(self):
        x_list = []
        y_list = []

        with open(self.config.status_file, "r") as file:
            status = file.read().split()[-1].strip()

            if status.lower() == "false":
                logger.error("Status file validation failed")
                raise ValueError("Validation status is False")

        with open(self.config.data_path, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                headers = game.headers

                try:
                    white_elo = int(headers.get("WhiteElo", 0))
                    black_elo = int(headers.get("BlackElo", 0))
                except ValueError:
                    continue

                if white_elo < 1200 or black_elo < 1200:
                    continue

                termination = headers.get("Termination", "")

                if "Time forfeit" in termination:
                    continue

                board = game.board()

                for move in game.mainline_moves():
                    x_list.append(self.board_to_matrix(board))
                    y_list.append(self.move_to_index(move))
                    board.push(move)

        return np.array(x_list, dtype=np.uint8), np.array(y_list, dtype=np.int16)

    def split_and_save_data(self, X: np.ndarray, y: np.ndarray) -> None:
        output_dir = Path(self.config.root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)  # ساخت پوشه در صورت عدم وجود

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.6667, random_state=42
        )

        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "y_train.npy", y_train)

        np.save(output_dir / "X_val.npy", X_val)
        np.save(output_dir / "y_val.npy", y_val)

        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_test.npy", y_test)

        logger.info(
            f"All dataset splits saved to {output_dir}  samples: {len(X)}"
        )

    def initiate_data_transformation(self):
        X, y = self.extract_dataset()
        if len(X) == 0:
            raise ValueError("No valid samples ")
        self.split_and_save_data(X, y)