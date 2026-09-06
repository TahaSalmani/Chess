import os
import  chess
import chess.pgn as pgn
from Chess import logger
from Chess.entity.config_entity import DataTransformationConfig
import numpy as np

class DataTransformation:
    def __init__(self , config :DataTransformationConfig):
        self.config = config

    def board_to_matrix(self, board: chess.Board):
        matrix = np.zeros((8, 8, 12), dtype=np.int8)
        piece_map = board.piece_map()

        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            piece_type = piece.piece_type - 1
            color = 0 if piece.color else 6
            channel = piece_type + color
            matrix[row, col, channel] = 1

        return matrix

    def transform_pgn_to_numpy(self):
        X = []
        Y = []
        game_count  = 0
        with open(self.config.status_file) as file:
            status = file.read().split()[-1].strip()

        if status != "True" :
            logger.error("Invalid status file")
            raise Exception("Invalid status file")

        os.makedirs(self.config.transformed_data_dir , exist_ok=True)
        max_games = 100
        with open (self.config.data_path  , encoding="utf-8") as pgn_file :
            while game_count < max_games :
                game_file = pgn.read_game(pgn_file)
                if game_file is None:
                    break

                board = game_file.board()
                result_in_file = game_file.headers.get("Result" , "*")
                result = 1.0 if result_in_file == "1-0" else (-1.0 if result_in_file == "0-1" else 0.0)
                for move in game_file.mainline_moves():
                    board.push(move)
                    X.append(self.board_to_matrix(board))
                    Y.append(result)

                game_count += 1

            x_arr = np.array(X)
            y_arr = np.array(Y)

            X_path = os.path.join(self.config.transformed_data_dir , "X")
            Y_path = os.path.join(self.config.transformed_data_dir , "Y")

            np.save(X_path , x_arr)
            np.save(Y_path , y_arr)


            logger.info(f"Added Successfully {game_count} games ")

