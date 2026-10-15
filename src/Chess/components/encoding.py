"""Canonical board encoding shared by training, self-play and inference.

Every consumer (data_transformation, ChessEnv, Mcts) must go through
``board_to_array`` so the planes never diverge between training and play.

Plane layout (8, 8, 17):
  0-5   : white pieces (pawn, knight, bishop, rook, queen, king)
  6-11  : black pieces (same order)
  12    : side to move (all ones when White is to move)
  13-16 : castling rights (white king-side, white queen-side,
          black king-side, black queen-side), each a constant plane
"""

import chess
import numpy as np

NUM_PLANES = 17


def board_to_array(board: chess.Board, dtype=np.float32) -> np.ndarray:
    matrix = np.zeros((8, 8, NUM_PLANES), dtype=dtype)

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel = (piece.piece_type - 1) + (0 if piece.color else 6)
        matrix[row, col, channel] = 1

    if board.turn == chess.WHITE:
        matrix[:, :, 12] = 1

    matrix[:, :, 13] = int(board.has_kingside_castling_rights(chess.WHITE))
    matrix[:, :, 14] = int(board.has_queenside_castling_rights(chess.WHITE))
    matrix[:, :, 15] = int(board.has_kingside_castling_rights(chess.BLACK))
    matrix[:, :, 16] = int(board.has_queenside_castling_rights(chess.BLACK))

    return matrix
