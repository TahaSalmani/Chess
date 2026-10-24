"""Unit + smoke tests for self-play resignation (RL round 2).

The resignation machinery is the round-2 lever: without it two weak copies of the
net shuffle pieces to the move cap and every value target is 0.0, so the value
head learns a constant. These tests pin the deterministic behaviour of
SelfPlayGame.note_root_value / result / finish_reason, then run one tiny CPU
self-play pass with a stub net to prove generate_self_play_data still integrates.

Run with:  python research/test_self_play.py
"""
import dataclasses
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from Chess.components.self_play import SelfPlayGame, generate_self_play_data  # noqa: E402
from Chess.entity.config_entity import PrepareRlModelConfig  # noqa: E402


def _game(move_count=0):
    g = SelfPlayGame()
    g.move_count = move_count
    return g


def test_no_resign_before_min_moves():
    g = _game(move_count=10)
    for _ in range(5):
        assert g.note_root_value(-0.99, threshold=0.8, window=3, min_moves=30) is False
    assert g.resigned_by is None, "must not resign before min_moves even on a hopeless value"
    print("PASS test_no_resign_before_min_moves")


def test_resign_after_window():
    g = _game(move_count=40)
    assert g.note_root_value(-0.9, 0.8, window=3, min_moves=30) is False
    assert g.note_root_value(-0.9, 0.8, window=3, min_moves=30) is False
    assert g.resigned_by is None, "two low values < window must not resign yet"
    assert g.note_root_value(-0.9, 0.8, window=3, min_moves=30) is True
    assert g.resigned_by == g.board.turn, "resign side must be the side to move"
    assert g.finish_reason() == "resign"
    print("PASS test_resign_after_window")


def test_streak_resets_on_recovered_value():
    g = _game(move_count=40)
    g.note_root_value(-0.9, 0.8, window=3, min_moves=30)
    g.note_root_value(-0.9, 0.8, window=3, min_moves=30)
    # a non-hopeless value clears the streak, so the count restarts
    assert g.note_root_value(-0.1, 0.8, window=3, min_moves=30) is False
    assert g.bad_streak == 0, "streak must reset when the value recovers"
    assert g.note_root_value(-0.9, 0.8, window=3, min_moves=30) is False
    assert g.note_root_value(-0.9, 0.8, window=3, min_moves=30) is False
    assert g.resigned_by is None, "reset streak needs a fresh full window to resign"
    print("PASS test_streak_resets_on_recovered_value")


def test_boundary_value_does_not_resign():
    g = _game(move_count=40)
    # value == -threshold is NOT strictly below, so it must not count as hopeless
    for _ in range(5):
        assert g.note_root_value(-0.8, 0.8, window=3, min_moves=30) is False
    assert g.resigned_by is None, "value == -threshold must not trigger resignation"
    print("PASS test_boundary_value_does_not_resign")


def test_resign_result_signs():
    g = _game(move_count=40)
    turn = g.board.turn  # White at the start position
    for _ in range(3):
        g.note_root_value(-0.9, 0.8, window=3, min_moves=30)
    assert g.resigned_by == turn
    expected = -1.0 if turn == chess.WHITE else 1.0
    assert g.result() == expected, f"resigning side {turn} must score {expected}, got {g.result()}"
    print("PASS test_resign_result_signs")


def test_training_data_flips_value_for_black():
    g = _game(move_count=40)
    # fabricate two recorded positions: one for White, one for Black
    pos = np.zeros((8, 8, 17), dtype=np.float32)
    pol = np.zeros(4096, dtype=np.float32)
    pol[0] = 1.0
    g.history.append((pos, pol, chess.WHITE))
    g.history.append((pos, pol, chess.BLACK))
    for _ in range(3):
        g.note_root_value(-0.9, 0.8, window=3, min_moves=30)
    resigned_white = g.resigned_by == chess.WHITE
    data = g.training_data()
    white_target = data[0][2]
    black_target = data[1][2]
    assert white_target == -black_target, "Black's value target must be the negation of White's"
    if resigned_white:
        assert white_target == -1.0 and black_target == 1.0
    print("PASS test_training_data_flips_value_for_black")


class _StubNet:
    """Deterministic stand-in for ValuePolicyNet: uniform-ish policy, fixed value."""

    def __init__(self, value=-0.9):
        self._value = value

    def eval(self):
        return self

    def __call__(self, x):
        b = x.shape[0]
        logits = torch.zeros(b, 4096)
        value = torch.full((b, 1), self._value)
        return logits, value


def _rl_config(tmp, **over):
    base = dict(
        root_dir=tmp,
        self_play_data=os.path.join(tmp, "self_play_data"),
        params_num_simulation=4,
        params_max_moves=6,
        params_num_games=2,
        params_batch_size=8,
        params_learning_rate=3e-4,
        params_epochs=1,
        params_c_puct=1.5,
        params_pool_size=2,
        params_dirichlet_alpha=0.3,
        params_dirichlet_epsilon=0.25,
        params_temp_moves=6,
        params_resign_threshold=0.8,
        params_resign_window=2,
        params_resign_min_moves=0,
    )
    base.update(over)
    fields = {f.name for f in dataclasses.fields(PrepareRlModelConfig)}
    return PrepareRlModelConfig(**{k: v for k, v in base.items() if k in fields})


def test_generate_self_play_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _rl_config(tmp)
        data = generate_self_play_data(_StubNet(value=-0.9), torch.device("cpu"), cfg)
        assert isinstance(data, list) and len(data) > 0, "self-play must produce positions"
        for pos, pol, val in data:
            assert pos.shape == (8, 8, 17), f"bad position shape {pos.shape}"
            assert pol.shape == (4096,), f"bad policy shape {pol.shape}"
            assert abs(pol.sum() - 1.0) < 1e-4, "visit distribution must be normalised"
            assert val in (-1.0, 0.0, 1.0), f"value target must be a game result, got {val}"
    print("PASS test_generate_self_play_smoke")


if __name__ == "__main__":
    test_no_resign_before_min_moves()
    test_resign_after_window()
    test_streak_resets_on_recovered_value()
    test_boundary_value_does_not_resign()
    test_resign_result_signs()
    test_training_data_flips_value_for_black()
    test_generate_self_play_smoke()
    print("\nALL SELF-PLAY TESTS PASSED")
