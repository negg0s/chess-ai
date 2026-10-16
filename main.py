import argparse
import sys
import chess
from pathlib import Path

from config import BASE_DIR, DATA_DIR, MODELS_DIR
from engine.stockfish_bridge import StockfishEngine

def analyze_fen(fen_str: str, depth: int = 20, time_limit: float = 1.5):
    try:
        board = chess.Board(fen_str)
    except ValueError as e:
        print(f"Invalid FEN: {e}")
        return

    print("Position:")
    print(board)
    print("-" * 40)

    engine = StockfishEngine()
    engine.start()
    result = engine.get_best_move(board, depth=depth, time_limit=time_limit)
    engine.close()

    print(f"Turn:        {result['turn']}")
    print(f"Eval:        {result['score']}")
    print(f"Depth:       {result['depth']}")
    print(f"Best move:   {result['best_move']}")
    if result.get('pv'):
        print(f"Line:        {' '.join(result['pv'][:6])}")

def main():
    parser = argparse.ArgumentParser(description="Chess analysis and training CLI")
    parser.add_argument("--train", action="store_true", help="Train neural network")
    parser.add_argument("--fen", type=str, help="Analyze FEN string")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for training")

    args = parser.parse_args()

    if args.train:
        from brain.train import train_chess_brain
        train_chess_brain(epochs=args.epochs)
    elif args.fen:
        analyze_fen(args.fen)
    else:
        analyze_fen(chess.STARTING_FEN, depth=18, time_limit=1.0)

if __name__ == "__main__":
    main()
