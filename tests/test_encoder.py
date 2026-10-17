import chess
import pytest
import torch

from brain.encoder import (
    board_to_tensor,
    index_to_legal_move,
    index_to_move,
    move_to_index,
    policy_best_move,
)


def test_board_to_tensor_shape():
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (18, 8, 8)
    assert tensor.dtype == torch.float32


def test_board_to_tensor_piece_planes():
    board = chess.Board()
    tensor = board_to_tensor(board)
    # white pawns on rank 2 -> row 6, plane 0
    for file_index in range(8):
        assert tensor[0][6][file_index].item() == 1.0
    # black pawns on rank 7 -> row 1, plane 6
    for file_index in range(8):
        assert tensor[6][1][file_index].item() == 1.0
    # side-to-move plane for white to move
    assert tensor[12].sum().item() == 64.0


def test_index_roundtrip_and_promotion():
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    move = chess.Move(chess.A7, chess.A8, promotion=chess.QUEEN)
    index = move_to_index(move)
    assert index_to_move(index).uci() == "a7a8"
    # underpromotion shares the same (from, to) index
    under = chess.Move(chess.A7, chess.A8, promotion=chess.KNIGHT)
    assert move_to_index(under) == index
    # policy read resolves to the queen variant because that promotion is legal
    resolved = index_to_legal_move(index, board)
    assert resolved is not None and resolved.promotion == chess.QUEEN


def test_index_to_legal_move_rejects_illegal():
    board = chess.Board()
    assert index_to_legal_move(move_to_index(chess.Move(chess.A1, chess.A8)), board) is None


def test_policy_best_move_prefers_queen_promotion():
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    logits = torch.zeros(4096)
    logits[move_to_index(chess.Move(chess.A7, chess.A8))] = 5.0
    move = policy_best_move(logits, board)
    assert move is not None
    assert move.from_square == chess.A7
    assert move.to_square == chess.A8
    assert move.promotion == chess.QUEEN  # not knight/bishop/rook variants


def test_policy_best_move_masks_illegal():
    board = chess.Board()
    logits = torch.zeros(4096)
    logits[move_to_index(chess.Move(chess.E2, chess.E4))] = 0.5
    logits[move_to_index(chess.Move(chess.A1, chess.H8))] = 9.0  # rook teleport
    move = policy_best_move(logits, board)
    # the a1h8 index is masked out, so the only positively flagged legal slot wins
    assert move is not None and move.uci() == "e2e4"


def test_policy_best_move_none_when_stalemate():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert policy_best_move(torch.zeros(4096), board) is None
