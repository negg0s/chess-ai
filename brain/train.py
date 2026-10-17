r"""
Chess Brain Training Pipeline (PyTorch)

Trains the ChessBrainNet policy-value network on synthetic self-play data from
greedy-material games. No grandmaster PGN loading is implemented; the data is
synthetic and has known limitations (see README and docs/model-eval.md).

Checkpoints are written to a new file with a date suffix and never overwrite
existing weights in models/.
"""
import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from brain.model import ChessBrainNet
from brain.encoder import board_to_tensor, move_to_index
from config import MODELS_DIR

PIECE_VALUES = [0, 100, 320, 330, 500, 900, 20000]
RANDOM_MOVE_RATIO = 0.1  # share of random legal moves for game diversity


class ChessPositionDataset(Dataset):
    def __init__(self, positions):
        """positions: list of (tensor_18x8x8, move_index_target, value_target)."""
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx):
        tensor, move_idx, val = self.positions[idx]
        return tensor, torch.tensor(move_idx, dtype=torch.long), torch.tensor(val, dtype=torch.float32)


def _greedy_move(board: chess.Board, rng: random.Random) -> chess.Move:
    """Move choice for synthetic games: material-maximizing with randomness."""
    legal_moves = list(board.legal_moves)
    if RANDOM_MOVE_RATIO and rng.random() < RANDOM_MOVE_RATIO:
        return rng.choice(legal_moves)

    best_score = -99999.0
    best_moves: list[chess.Move] = []
    for move in legal_moves:
        board.push(move)
        score = 0.0
        for piece in board.piece_map().values():
            value = PIECE_VALUES[piece.piece_type]
            # after the push board.turn is the opponent, so the mover's own
            # material is the piece color that is NOT to move
            score += value if piece.color != board.turn else -value
        if board.is_checkmate():
            score += 100000.0  # prioritize the fastest mate
        elif board.is_check():
            score += 50.0
        board.pop()
        if score > best_score + 1e-9:
            best_score = score
            best_moves = [move]
        elif abs(score - best_score) <= 1e-9:
            best_moves.append(move)
    return rng.choice(best_moves)


def _value_label(turn: chess.Color, result: str, perspective: str) -> float:
    """Value target for a position.

    Labels are computed from the WHITE side perspective by default
    (1.0 white wins, -1.0 black wins, 0.0 draw). With perspective
    == 'side_to_move' the label is flipped for positions where black is to
    move, matching the model docstring's wording.

    An unknown result ("*", i.e. a game truncated at the half-move limit)
    yields NaN, which the training loop excludes from the value-head loss
    (CH-08): these positions keep their policy label but never pretend to be
    draws.
    """
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = -1.0
    elif result == "*":
        outcome = float("nan")
    else:
        outcome = 0.0
    if perspective == "side_to_move" and turn == chess.BLACK:
        outcome = -outcome
    return outcome


def generate_synthetic_training_data(
    num_games: int = 50,
    seed: int | None = None,
    perspective: str = "white",
    require_finished: bool = False,
):
    """Generates training data from simulated games.

    Games are greedy-material with a small share of random moves and seeded
    randomness, so two runs with the same seed produce identical samples.

    Value labels for unfinished games are NaN and are masked out of the
    value-head loss by the training loop (CH-08). With require_finished=True
    unfinished games are dropped from the dataset entirely; note that with
    deterministic greedy play that can filter out everything, because
    material-greedy games frequently run out of the 80-half-move limit before
    mate.
    """
    rng = random.Random(seed)
    print(f"[Dataset] Generating {num_games} training games (seed={seed})...")
    samples = []
    for _ in range(num_games):
        board = chess.Board()
        move_history = []
        board_states = []

        while not board.is_game_over() and len(move_history) < 80:
            best_move = _greedy_move(board, rng)
            board_states.append((board_to_tensor(board), move_to_index(best_move), board.turn))
            board.push(best_move)
            move_history.append(best_move)

        result = board.result()
        if require_finished and result == "*":
            continue

        for tensor, move_idx, turn in board_states:
            samples.append((tensor, move_idx, _value_label(turn, result, perspective)))

    print(f"[Dataset] Generated {len(samples)} position samples.")
    return samples


def _checkpoint_path(save_name: str | None) -> Path:
    """Non-overwriting checkpoint path in models/."""
    stamp = datetime.now().strftime("%Y%m%d")
    if save_name:
        candidate = MODELS_DIR / save_name
        if not candidate.exists():
            return candidate
        base = candidate.stem
    else:
        base = f"chess_brain_{stamp}"
    suffix = 0
    while True:
        candidate = MODELS_DIR / f"{base}{'_' + str(suffix) if suffix else ''}.pt"
        if not candidate.exists():
            return candidate
        suffix += 1


def train_chess_brain(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    num_games: int = 100,
    seed: int | None = None,
    perspective: str = "white",
    save_name: str | None = None,
) -> Path | None:
    """Train on synthetic data with a validation split.

    Best model selection uses validation loss, not training loss. A new
    checkpoint file is created; existing files are never overwritten. Returns
    the path of the saved checkpoint or None.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Device: {device}")

    samples = generate_synthetic_training_data(num_games=num_games, seed=seed, perspective=perspective)
    if not samples:
        print("[Training] Dataset is empty, aborting.")
        return None

    dataset = ChessPositionDataset(samples)
    val_count = max(1, int(len(dataset) * 0.1))
    train_count = len(dataset) - val_count
    train_set, val_set = random_split(dataset, [train_count, val_count])
    print(f"[Training] Split: {train_count} train / {val_count} validation")

    generator = torch.Generator().manual_seed(seed or 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = ChessBrainNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    def _value_loss(pred_value, target_value):
        # NaN value labels (unfinished games) are excluded from the loss (CH-08)
        mask = ~torch.isnan(target_value).squeeze(-1)
        if not mask.any():
            return torch.zeros((), device=device)
        return value_criterion(pred_value[mask], target_value[mask])

    best_val = float("inf")
    save_path = _checkpoint_path(save_name)
    print(f"[Training] Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, target_policy, target_value in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            batch_x = batch_x.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.unsqueeze(1).to(device)

            optimizer.zero_grad()
            pred_policy, pred_value = model(batch_x)
            loss = policy_criterion(pred_policy, target_policy) + _value_loss(pred_value, target_value)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, target_policy, target_value in val_loader:
                batch_x = batch_x.to(device)
                target_policy = target_policy.to(device)
                target_value = target_value.unsqueeze(1).to(device)
                pred_policy, pred_value = model(batch_x)
                val_loss += (
                    policy_criterion(pred_policy, target_policy).item()
                    + _value_loss(pred_value, target_value).item()
                )
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch}: train={total_loss / max(1, len(train_loader)):.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[Checkpoint] Saved to {save_path} (val_loss={val_loss:.4f})")

    print(f"[Training] Complete. Best validation loss: {best_val:.4f}, checkpoint: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChessBrainNet on synthetic data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--value-perspective", choices=["white", "side_to_move"], default="white")
    args = parser.parse_args()
    train_chess_brain(epochs=args.epochs, num_games=args.num_games, seed=args.seed, perspective=args.value_perspective)
