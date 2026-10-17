from __future__ import annotations

import argparse
import sys

import chess

from engine.stockfish_bridge import StockfishEngine


def analyze_fen(fen_str: str, depth: int = 20, time_limit: float = 1.5):
    """Analyze a FEN with Stockfish; the engine subprocess is always closed."""
    try:
        board = chess.Board(fen_str)
    except ValueError as exc:
        print(f"Invalid FEN: {exc}")
        return None

    print("Position:")
    print(board)
    print("-" * 40)

    try:
        with StockfishEngine() as engine:
            result = engine.get_best_move(board, depth=depth, time_limit=time_limit)
    except RuntimeError as exc:
        print(f"Engine error: {exc}")
        return None

    print(f"Turn:        {result['turn']}")
    print(f"Eval:        {result['score']}")
    print(f"Depth:       {result['depth']}")
    print(f"Best move:   {result['best_move']}")
    if result.get("pv"):
        print(f"Line:        {' '.join(result['pv'][:6])}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Chess analysis and training CLI")
    parser.add_argument("--train", action="store_true", help="Train neural network")
    parser.add_argument("--fen", type=str, help="Analyze FEN string")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for training")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible training")
    parser.add_argument("--num-games", type=int, default=100, help="Synthetic games for training")
    args = parser.parse_args()

    if args.train:
        from brain.train import train_chess_brain

        train_chess_brain(epochs=args.epochs, seed=args.seed, num_games=args.num_games)
        return 0

    rc = analyze_fen(args.fen if args.fen else chess.STARTING_FEN, depth=18, time_limit=1.0)
    return 0 if rc is not None else 1


if __name__ == "__main__":
    sys.exit(main())
