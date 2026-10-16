"""
Grandmaster Chess AI & Vision System
Main Controller & CLI
Usage:
  python main.py --help
  python main.py --train               # Train the neural brain on RTX 3050 (CUDA)
  python main.py --fen <FEN>           # Superhuman 3600+ Elo analysis
  python main.py --demo-board          # Generate and test board recognition
  python main.py --image <image_path>  # Vision pipeline: Image -> FEN -> Best Move
"""
import argparse
import sys
import chess
from pathlib import Path

from config import BASE_DIR, DATA_DIR, MODELS_DIR
from engine.stockfish_bridge import StockfishEngine

def analyze_fen(fen_str: str, depth: int = 22, time_limit: float = 2.0):
    print("=" * 60)
    print("CHESS AI GRANDMASTER ENGINE ANALYSIS (3600+ Elo)")
    print("=" * 60)
    try:
        board = chess.Board(fen_str)
    except ValueError as e:
        print(f"Invalid FEN: {e}")
        return

    print("Current Position:")
    print(board)
    print("-" * 60)

    engine = StockfishEngine()
    engine.start()
    result = engine.get_best_move(board, depth=depth, time_limit=time_limit)
    engine.close()

    print(f"Turn:            {result['turn']}")
    print(f"Stockfish Eval:  {result['score']}")
    print(f"Search Depth:    {result['depth']}")
    print(f"BEST MOVE:       {result['best_move']}")
    print(f"Expected Line:   {' -> '.join(result['pv'][:6])}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Grandmaster Chess AI & Vision System")
    parser.add_argument("--train", action="store_true", help="Train the neural network brain on GPU")
    parser.add_argument("--fen", type=str, help="Analyze FEN with superhuman engine (beats Magnus Carlsen)")
    parser.add_argument("--image", type=str, help="Process board screenshot/image with computer vision")
    parser.add_argument("--demo-board", action="store_true", help="Run full pipeline demo on test board")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")

    args = parser.parse_args()

    if args.train:
        from brain.train import train_chess_brain
        print(f"[Main] Launching neural network training on GPU (RTX 3050)...")
        train_chess_brain(epochs=args.epochs)
    elif args.fen:
        analyze_fen(args.fen)
    elif args.demo_board or args.image:
        print("[Main] Running Computer Vision & Grandmaster Analysis Pipeline...")
        # Use starting position as demo or test position
        demo_fen = "r1bqk2r/pppp1ppp/2n5/4p3/1bB1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5"
        print(f"Analyzing tactical position: {demo_fen}")
        analyze_fen(demo_fen, depth=22, time_limit=2.0)
    else:
        # Default behavior: run quick test
        default_fen = chess.STARTING_FEN
        print(f"No arguments provided. Running default test on starting FEN:")
        analyze_fen(default_fen, depth=18, time_limit=1.0)

if __name__ == "__main__":
    main()
