import math
import numpy as np
import torch
import chess
from Chess.config.configuration import PrepareRlModelConfig

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, prior: float = 0.0 , config = PrepareRlModelConfig):
        self.config = config
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0


class MCTS:
    def __init__(self, model, device,config = PrepareRlModelConfig):
        self.config = config
        self.model = model
        self.device = device

    def board_to_tensor(self, board: chess.Board):
        matrix = np.zeros((8, 8, 12), dtype=np.float32)
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            channel = (piece.piece_type - 1) + (0 if piece.color else 6)
            matrix[row, col, channel] = 1
        return torch.tensor(matrix).unsqueeze(0).to(self.device)

    def run(self, root_board: chess.Board):
        root = MCTSNode(root_board.copy())
        self._expand(root)

        for _ in range(self.config.params_num_simulation):
            node = root
            search_path = [node]

            while node.is_expanded():
                move, node = self._select_child(node)
                search_path.append(node)

            value = self._expand(node)
            self._backpropagate(search_path, value)

        return root

    def _select_child(self, node: MCTSNode):
        best_score = -float("inf")
        best_move, best_child = None, None

        for move, child in node.children.items():
            score = child.value() + self.config.params_c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_move, best_child = move, child

        return best_move, best_child

    def _expand(self, node: MCTSNode):
        board = node.board

        if board.is_game_over():
            if board.is_checkmate():
                return -1.0
            return 0.0

        x = self.board_to_tensor(board)
        self.model.eval()

        with torch.no_grad():
            policy_logits, value = self.model(x)

        policy = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
        value = value.item()

        for move in board.legal_moves:
            idx = move.from_square * 64 + move.to_square
            child_board = board.copy()
            child_board.push(move)
            node.children[move] = MCTSNode(child_board, parent=node, prior=policy[idx])

        return value

    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def get_best_move(self, root_board: chess.Board):
        root = self.run(root_board)
        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_move, root