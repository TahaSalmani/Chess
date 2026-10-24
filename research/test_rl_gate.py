"""Unit tests for the RL gate: verify BOTH acceptance and rejection paths and
that a rejected candidate never touches the supervised checkpoint.

Deterministic: both players are FirstLegalBot, so every game is identical and
ends at the ply cap as a draw (score 0.5). Thresholds are set either side of
0.5 to force accept and reject. Run with:  python research/test_rl_gate.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

import chess  # noqa: E402

from rl_gate import evaluate, decide, gate, _file_sha  # noqa: E402


class FirstLegalBot:
    name = "first-legal"

    def choose(self, board):
        return next(iter(board.legal_moves))


def _quiet(_m):
    pass


def test_decide_boundaries():
    assert decide(0.55, 0.55) is True, "score == min_score must accept"
    assert decide(0.56, 0.55) is True
    assert decide(0.54, 0.55) is False, "score < min_score must reject"
    assert decide(1.0, 0.55) is True
    assert decide(0.0, 0.55) is False
    print("PASS test_decide_boundaries")


def test_evaluate_is_deterministic_draw():
    score, record = evaluate(FirstLegalBot(), FirstLegalBot(), games=2, seed=0,
                             max_plies=20, log=_quiet)
    assert score == 0.5, f"expected deterministic draw score 0.5, got {score}"
    assert record == (0, 2, 0), f"expected (0 wins, 2 draws, 0 losses), got {record}"
    print("PASS test_evaluate_is_deterministic_draw")


def test_accept_promotes_and_leaves_supervised_untouched():
    with tempfile.TemporaryDirectory() as d:
        sup = os.path.join(d, "best_model.pth")
        rl = os.path.join(d, "rl_trained_model.pth")
        promoted = os.path.join(d, "sub", "promoted_model.pth")
        with open(sup, "wb") as f:
            f.write(b"SUPERVISED-WEIGHTS-v1")
        with open(rl, "wb") as f:
            f.write(b"RL-CANDIDATE-WEIGHTS")
        sup_before = _file_sha(sup)

        res = gate(FirstLegalBot(), FirstLegalBot(), games=2, min_score=0.4,
                   seed=0, max_plies=20, rl_path=rl, promote_to=promoted,
                   supervised_path=sup, log=_quiet)

        assert res["accepted"] is True, "score 0.5 >= 0.4 must accept"
        assert res["promoted"] is True
        assert os.path.exists(promoted), "accepted candidate must be promoted"
        with open(promoted, "rb") as f:
            assert f.read() == b"RL-CANDIDATE-WEIGHTS", "promoted file must be the RL weights"
        # supervised checkpoint byte-identical
        assert _file_sha(sup) == sup_before, "supervised checkpoint must be unchanged"
        with open(sup, "rb") as f:
            assert f.read() == b"SUPERVISED-WEIGHTS-v1"
    print("PASS test_accept_promotes_and_leaves_supervised_untouched")


def test_reject_does_not_promote_and_keeps_supervised():
    with tempfile.TemporaryDirectory() as d:
        sup = os.path.join(d, "best_model.pth")
        rl = os.path.join(d, "rl_trained_model.pth")
        promoted = os.path.join(d, "promoted_model.pth")
        with open(sup, "wb") as f:
            f.write(b"SUPERVISED-WEIGHTS-v1")
        with open(rl, "wb") as f:
            f.write(b"RL-CANDIDATE-WEIGHTS")
        sup_before = _file_sha(sup)

        res = gate(FirstLegalBot(), FirstLegalBot(), games=2, min_score=0.6,
                   seed=0, max_plies=20, rl_path=rl, promote_to=promoted,
                   supervised_path=sup, log=_quiet)

        assert res["accepted"] is False, "score 0.5 < 0.6 must reject"
        assert res["promoted"] is False
        assert not os.path.exists(promoted), "rejected candidate must NOT create a promoted file"
        assert _file_sha(sup) == sup_before, "supervised checkpoint must be unchanged on reject"
        with open(sup, "rb") as f:
            assert f.read() == b"SUPERVISED-WEIGHTS-v1"
    print("PASS test_reject_does_not_promote_and_keeps_supervised")


if __name__ == "__main__":
    test_decide_boundaries()
    test_evaluate_is_deterministic_draw()
    test_accept_promotes_and_leaves_supervised_untouched()
    test_reject_does_not_promote_and_keeps_supervised()
    print("\nALL GATE TESTS PASSED")
