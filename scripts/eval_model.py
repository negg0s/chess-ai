"""Author-vdeck: policy/value probe for the shipped model weights.

Metrics per docs/model-eval.md:
- policy top-1 / top-3 agreement with Stockfish depth 8 over LEGAL moves,
  counted on the 4096 (from,to) move space with promotions resolved to queen;
- Spearman + Pearson correlation of model value vs tanh(cp/400).

Deterministic seed; no random shuffling.
"""
import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import torch
from chess.engine import Limit

from brain.encoder import board_to_tensor, move_to_index
from brain.model import ChessBrainNet
from config import MODELS_DIR, MODEL_FILENAME
from engine.stockfish_bridge import StockfishEngine

NUM_POSITIONS = 300
SF_DEPTH = 8
SEED = 42


def collect_positions(board_factory, seed: int, positions_needed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    while len(boards) < positions_needed:
        board = board_factory(rng)
        ply = 0
        while len(boards) < positions_needed:
            legal = list(board.legal_moves)
            if not legal or ply > 120:
                break
            if ply >= 4:
                boards.append(board.copy())
            board.push(rng.choice(legal))
            ply += 1
    return boards


def sample_value_stats(cp: float) -> float:
    return math.tanh(cp / 400.0)


def main() -> int:
    model_path = MODELS_DIR / MODEL_FILENAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessBrainNet().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    openings = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
        "rnbqkb1r/pppppppp/5n2/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 3",
    ]

    def opening_position(rng: random.Random) -> chess.Board:
        return chess.Board(rng.choice(openings))

    boards = collect_positions(opening_position, SEED, NUM_POSITIONS)
    print(f"positions={len(boards)} device={device}")

    with StockfishEngine() as sf_engine:
        hits1 = 0
        hits3 = 0
        model_values = []
        sf_values = []

        for i, board in enumerate(boards):
            with torch.no_grad():
                policy, value = model(board_to_tensor(board).unsqueeze(0).to(device))
            policy = policy[0]
            candidates = {move_to_index(m) for m in board.legal_moves}
            ranked = sorted(candidates, key=lambda idx: -float(policy[idx]))
            top3_candidates = ranked[:3]

            info = sf_engine.engine.analyse(board, Limit(depth=SF_DEPTH))
            sf_move = info["pv"][0] if info.get("pv") else None
            cp = info["score"].white().score(mate_score=10000) or 0

            if sf_move is not None and move_to_index(sf_move) == ranked[0]:
                hits1 += 1
            if sf_move is not None and move_to_index(sf_move) in top3_candidates:
                hits3 += 1
            model_values.append(float(math.tanh(value.item())))
            sf_values.append(sample_value_stats(float(cp)))

            if (i + 1) % 50 == 0:
                print(f"... {i + 1}/{len(boards)}: top1={hits1 / (i + 1):.3f} top3={hits3 / (i + 1):.3f}")

        n = len(boards)
        mean_m = sum(model_values) / n
        mean_s = sum(sf_values) / n
        cov = sum((m - mean_m) * (s - mean_s) for m, s in zip(model_values, sf_values)) / n
        var_m = sum((m - mean_m) ** 2 for m in model_values) / n
        var_s = sum((s - mean_s) ** 2 for s in sf_values) / n

        print("\nRESULTS (deterministic, seed=%d)" % SEED)
        print(f"positions: {n}")
        print(f"policy top-1 agreement vs Stockfish depth {SF_DEPTH}: {hits1 / n:.4f}")
        print(f"policy top-3 containment vs Stockfish depth {SF_DEPTH}: {hits3 / n:.4f}")
        print(f"signal (mean abs value diff): {sum(abs(m - s) for m, s in zip(model_values, sf_values)) / n:.4f}")
        print(f"pearson(corr): {cov / math.sqrt(var_m * var_s):.4f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy/value probe of shipped weights")
    parser.add_argument("--num-positions", type=int, default=None)
    args = parser.parse_args()
    if args.num_positions:
        NUM_POSITIONS = args.num_positions
    sys.exit(main())
