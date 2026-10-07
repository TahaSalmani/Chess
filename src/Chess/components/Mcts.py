import math

import chess
import numpy as np
import torch

from Chess.entity.config_entity import PrepareRlModelConfig

ACTION_SPACE = 64 * 64


def move_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def board_to_array(board: chess.Board) -> np.ndarray:
    matrix = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel = (piece.piece_type - 1) + (0 if piece.color else 6)
        matrix[row, col, channel] = 1.0
    return matrix


class MCTSNode:
    __slots__ = ("board", "parent", "move", "prior", "children", "visit_count", "value_sum", "terminal_value")

    def __init__(self, board=None, parent=None, move=None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        # None -> never expanded, {} -> expanded terminal node
        self.children = None
        self.visit_count = 0
        self.value_sum = 0.0
        self.terminal_value = None

    @property
    def is_expanded(self) -> bool:
        return self.children is not None

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """PUCT search whose network evaluations are batched across independent roots.

    Passing one root per self-play game makes each simulation a single
    batched forward pass instead of `num_games` separate ones.
    """

    def __init__(self, model, device, config: PrepareRlModelConfig):
        self.config = config
        self.model = model
        self.device = device
        self._c_puct = float(config.params_c_puct)
        self._num_simulation = int(config.params_num_simulation)
        self.model.eval()
        self.forward_calls = 0
        self.positions_evaluated = 0

    def _predict(self, boards, indices):
        x = torch.from_numpy(np.stack([board_to_array(b) for b in boards])).to(self.device)

        with torch.no_grad():
            logits, value = self.model(x)

        mask = np.zeros(logits.shape, dtype=bool)
        for i, idx in enumerate(indices):
            mask[i, idx] = True
        logits = logits.masked_fill(~torch.from_numpy(mask).to(self.device), float("-inf"))

        policy = torch.softmax(logits, dim=-1)
        return policy.cpu().numpy(), value.reshape(-1).cpu().numpy()

    def _select_leaf(self, root: MCTSNode):
        node = root
        path = [node]
        while node.children:
            move, child = self._select_child(node)
            if child.board is None:
                child.board = node.board.copy()
                child.board.push(move)
            node = child
            path.append(node)
        return node, path

    def _select_child(self, node: MCTSNode):
        parent_visits = math.sqrt(node.visit_count)
        best_score = -math.inf
        best_move, best_child = None, None
        for move, child in node.children.items():
            score = child.value() + self._c_puct * child.prior * parent_visits / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_move, best_child = move, child
        return best_move, best_child

    def _backpropagate(self, search_path, value: float):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def search_batch(self, roots):
        """Run `num_simulation` iterations on every root, sharing one forward per iteration."""
        if not roots:
            return

        for _ in range(self._num_simulation):
            leaves = []
            paths = []
            for root in roots:
                leaf, path = self._select_leaf(root)
                leaves.append(leaf)
                paths.append(path)

            values = [0.0] * len(leaves)
            pending = []
            for i, leaf in enumerate(leaves):
                if leaf.terminal_value is not None:
                    values[i] = leaf.terminal_value
                    continue
                board = leaf.board
                if board.is_game_over():
                    leaf.terminal_value = -1.0 if board.is_checkmate() else 0.0
                    leaf.children = {}
                    values[i] = leaf.terminal_value
                else:
                    pending.append(i)

            if pending:
                boards = [leaves[i].board for i in pending]
                legal = [list(b.legal_moves) for b in boards]
                indices = [
                    np.fromiter((m.from_square * 64 + m.to_square for m in moves),
                                dtype=np.int64, count=len(moves))
                    for moves in legal
                ]
                policy, value = self._predict(boards, indices)
                self.forward_calls += 1
                self.positions_evaluated += len(pending)

                for k, i in enumerate(pending):
                    leaf = leaves[i]
                    priors = policy[k][indices[k]]
                    leaf.children = {
                        move: MCTSNode(parent=leaf, move=move, prior=float(prior))
                        for move, prior in zip(legal[k], priors)
                    }
                    values[i] = float(value[k])

            for i, path in enumerate(paths):
                self._backpropagate(path, values[i])

    def run(self, board: chess.Board) -> MCTSNode:
        root = MCTSNode(board=board.copy())
        self.search_batch([root])
        return root

    @staticmethod
    def best_move(root: MCTSNode):
        if not root.children:
            return None
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def get_best_move(self, board: chess.Board):
        root = self.run(board)
        return self.best_move(root), root

    @staticmethod
    def visit_counts(root: MCTSNode) -> np.ndarray:
        counts = np.zeros(ACTION_SPACE, dtype=np.float32)
        for move, child in root.children.items():
            counts[move_index(move)] = child.visit_count
        return counts
