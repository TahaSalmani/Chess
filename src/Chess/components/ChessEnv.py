import torch
import chess
import gymnasium as gym
import numpy as np

from Chess.components.encoding import NUM_PLANES, board_to_array


class ChessEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config : None):
        super(ChessEnv, self).__init__()
        self.config = config
        self.board = chess.Board()
        self.observation_space = gym.spaces.Box(
            high = 1 ,
            low = 0,
            shape=(8,8,NUM_PLANES) ,
            dtype = np.float32
        )
        self.action_space = gym.spaces.Discrete(4096)

    def board_to_matrix(self):
        return board_to_array(self.board)

    def reset(self , seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        return self.board_to_matrix(), {}

    def step(self, action):
        from_square = action // 64
        to_square = action % 64

        move = chess.Move(from_square, to_square)

        if move not in self.board.legal_moves:
            promo_move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            if promo_move in self.board.legal_moves:
                move = promo_move
        if move not in self.board.legal_moves:
            return self.board_to_matrix(), -1.0, True, False, {"error": "illegal_move"}

        self.board.push(move)
        terminated = False
        reward = 0.0

        if self.board.is_checkmate():
            reward = 1.0
            terminated = True
        elif self.board.is_game_over():
            reward = 0.0
            terminated = True

        return self.board_to_matrix(), reward, terminated, False, {}

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for move in self.board.legal_moves:
            idx = move.from_square * 64 + move.to_square
            mask[idx] = True
        return mask

    def render(self):
        print(self.board)

