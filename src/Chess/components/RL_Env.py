import gymnasium as gym
from gymnasium import spaces
import numpy as np
import chess
from Chess.entity.config_entity import PrepareRlEnvConfig

class ChessEnv(gym.Env) :
    metadata = {'render.modes': ['human' , 'unicode'  ]}

    def __init__(self , config : PrepareRlEnvConfig  ):
        super(ChessEnv , self).__init__()
        self.config = config
        self.board = chess.Board()
        self.observation_space = spaces.Box(
            low = 0 ,
            high = 1 ,
            shape = (self.config.board_cols , self.config.board_rows , self.config.pieces_type),
            dtype = np.float32
        )

        self.action_space = spaces.Discrete(self.config.action_space_size)

    def board_to_matrix(self):
        matrix = np.zeros((8, 8, 12), dtype=np.int8)
        piece_map = self.board.piece_map()

        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            piece_type = piece.piece_type - 1
            color = 0 if piece.color else 6
            channel = piece_type + color
            matrix[row, col, channel] = 1

        return matrix

    def reset(self , seed = None , options = None):
        super().reset() ### reset Env
        self.board.reset()
        return  self.board_to_matrix() , {}

    def step (self , actions) :
        from_square = actions // 64
        to_square = actions % 64
        move = chess.Move(from_square , to_square )
        rewards = self.config.reward

        terminated = True
        truncated = False

        if move not in self.board.legal_moves:
            return self.board_to_matrix() , rewards["illegal_move"], terminated , truncated , {"error " : "illegal_move"}


        self.board.push(move)

        terminated = False
        truncated = False
        if  self.board.is_checkmate():
            reward = rewards["checkmate"]
            terminated = True

        elif self.board.is_game_over():
            reward = rewards["draw"]
            terminated = True
        else :
            reward = rewards["step_penalty"]
        return self.board_to_matrix() , reward , terminated , truncated , {}


    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)

        for move in self.board.legal_moves:
            action_idx = move.from_square * 64 + move.to_square

            mask[action_idx] = True

        return mask




