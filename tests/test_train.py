import chess
import math
import pytest
import random
import torch

import brain.train as train_mod
from brain.train import (
    ChessPositionDataset,
    _greedy_move,
    _value_label,
    generate_synthetic_training_data,
)


@pytest.fixture(autouse=True)
def no_random_moves(monkeypatch):
    monkeypatch.setattr(train_mod, "RANDOM_MOVE_RATIO", 0.0)


@pytest.fixture
def material_gain_board():
    """White pawn e4 can capture the black pawn d5 (+100) while every other
    legal move is material-neutral."""
    return chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")


def test_greedy_move_takes_material(material_gain_board):
    board = material_gain_board
    move = _greedy_move(board, random.Random(0))
    assert move.uci() == "e4d5"


def test_material_sign_counts_mover_perspective(material_gain_board):
    """The regression for CH-02: after the buggy version counted material from
    the opponent's perspective, the capture scored 'keeps material equal' and
    lost to quiet moves."""
    board = material_gain_board
    scores = {}
    for move in list(board.legal_moves):
        board.push(move)
        score = 0.0
        for piece in board.piece_map().values():
            value = train_mod.PIECE_VALUES[piece.piece_type]
            score += value if piece.color != board.turn else -value
        board.pop()
        scores[move.uci()] = score
    best = max(scores, key=scores.get)
    assert best == "e4d5"
    assert scores["e4d5"] > 0


def test_dataset_determinism_with_seed():
    first = generate_synthetic_training_data(num_games=2, seed=7)
    second = generate_synthetic_training_data(num_games=2, seed=7)
    assert len(first) == len(second) > 0
    for (t1, m1, v1), (t2, m2, v2) in zip(first, second):
        assert torch.equal(t1, t2)
        assert m1 == m2
        assert v1 == v2 or (math.isnan(v1) and math.isnan(v2))


def test_dataset_diversity_with_seed():
    samples = generate_synthetic_training_data(num_games=20, seed=11)
    unique_fens = {tuple(t.flatten().tolist()) for t, _, _ in samples}
    # greedy games repeat, but the random-move share must produce variety
    assert len(unique_fens) > 60


def test_require_finished_skips_truncated_games():
    finished = generate_synthetic_training_data(num_games=30, seed=3, require_finished=True)
    unfinished = generate_synthetic_training_data(num_games=30, seed=3, require_finished=False)
    # truncated games must disappear from the default dataset (CH-08)
    assert len(finished) < len(unfinished)


def test_value_label_perspective():
    assert _value_label(chess.WHITE, "1-0", "white") == 1.0
    assert _value_label(chess.WHITE, "0-1", "white") == -1.0
    # side_to_move: black positions flip the label (CH-03)
    assert _value_label(chess.BLACK, "1-0", "side_to_move") == -1.0
    assert _value_label(chess.BLACK, "1-0", "white") == 1.0
    assert _value_label(chess.WHITE, "1-0", "side_to_move") == 1.0


def test_dataset_labels_are_valid_or_masked():
    for _, _, value in generate_synthetic_training_data(num_games=5, seed=5):
        assert value in (-1.0, 0.0, 1.0) or math.isnan(value)


def test_dataset_object():
    samples = generate_synthetic_training_data(num_games=1, seed=1)
    ds = ChessPositionDataset(samples)
    tensor, move_idx, value = ds[0]
    assert tensor.shape == (18, 8, 8)
    assert move_idx.dtype == torch.long
    assert value.dtype == torch.float32


def test_checkpoint_never_overwrites(tmp_path, monkeypatch):
    import brain.train

    monkeypatch.setattr(brain.train, "MODELS_DIR", tmp_path)
    path1 = brain.train._checkpoint_path("model.pt")
    path1.write_bytes(b"x")
    path2 = brain.train._checkpoint_path("model.pt")
    assert path1 != path2
    assert path2.exists() is False
