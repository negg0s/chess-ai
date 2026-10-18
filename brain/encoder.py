"""
Chess Board to Tensor Representation
Encodes a python-chess Board into an 18x8x8 PyTorch tensor.
"""
import torch
import numpy as np
import chess

PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encodes chess.Board into 18x8x8 float tensor:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (p, n, b, r, q, k)
    12: Turn (1 if White, 0 if Black)
    13: White Kingside castling
    14: White Queenside castling
    15: Black Kingside castling
    16: Black Queenside castling
    17: En passant target square
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)  # rank 8 at row 0
        col = square % 8         # file a at col 0
        plane_offset = 0 if piece.color == chess.WHITE else 6
        plane = plane_offset + PIECE_TO_PLANE[piece.piece_type]
        tensor[plane, row, col] = 1.0

    # Turn
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # En passant
    if board.ep_square is not None:
        ep_row = 7 - (board.ep_square // 8)
        ep_col = board.ep_square % 8
        tensor[17, ep_row, ep_col] = 1.0

    return torch.from_numpy(tensor)

def move_to_index(move: chess.Move) -> int:
    """Encodes move (from_sq, to_sq) into index 0..4095.

    Underpromotions and queen promotion share the same index (from, to);
    the 4096 space has no promotion dimension.
    """
    return move.from_square * 64 + move.to_square

def index_to_move(index: int) -> chess.Move:
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)

def index_to_legal_move(index: int, board: chess.Board) -> chess.Move | None:
    """Map a policy index to a legal move on `board`, promoting pawns to queen.

    Returns the queen-promotion variant when the encoded (from, to) pair is a
    legal pawn push or capture to the last rank; returns None for illegal
    encodings. Underpromotion is not representable in the 4096 space, so the
    read is lossy by design: any underpromotion recorded in a dataset would
    share an index with the queen promotion.
    """
    move = chess.Move(index // 64, index % 64)
    if move in board.legal_moves:
        return move
    pawn = board.piece_at(move.from_square)
    if pawn is not None and pawn.piece_type == chess.PAWN:
        promo_rank = chess.square_rank(move.to_square)
        if promo_rank in (0, 7):
            for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                candidate = chess.Move(move.from_square, move.to_square, promotion=promo)
                if candidate in board.legal_moves:
                    return candidate
    return None

def policy_best_move(policy_logits, board: chess.Board) -> chess.Move | None:
    """Best legal move from policy logits for `board`.

    Masks illegal indexes and promotes pawns to queen. Returns None when the
    board has no legal moves.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None

    duplicates = {}  # 4096 index -> best representative move (prefer queen promo)
    for move in legal:
        index = move_to_index(move)
        current = duplicates.get(index)
        if current is None or (current.promotion != chess.QUEEN and move.promotion == chess.QUEEN):
            duplicates[index] = move

    mask = torch.full_like(policy_logits, float("-inf"))
    for index in duplicates:
        mask[index] = 0.0
    best_index = int(torch.argmax(policy_logits + mask).item())

    move = index_to_legal_move(best_index, board)
    if move is not None:
        return move
    return duplicates[best_index]


