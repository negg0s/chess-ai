r"""
Chess Brain Training Pipeline (PyTorch + CUDA on RTX 3050)
Trains ChessBrainNet policy-value network using:
- Supervised learning from grandmaster PGN games / FEN positions
- Synthetic self-play position generator
Stores model checkpoints strictly in D:\chess_ai\models\
"""
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from brain.model import ChessBrainNet
from brain.encoder import board_to_tensor, move_to_index
from config import MODELS_DIR, DATA_DIR

class ChessPositionDataset(Dataset):
    def __init__(self, positions):
        """
        positions is a list of tuples: (tensor_18x8x8, move_index_target, value_target)
        """
        self.positions = positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        tensor, move_idx, val = self.positions[idx]
        return tensor, torch.tensor(move_idx, dtype=torch.long), torch.tensor(val, dtype=torch.float32)

def generate_synthetic_training_data(num_games: int = 50):
    """
    Generates training data from simulated games with heuristic/search supervision.
    Useful for kickstarting the model training immediately.
    """
    print(f"[Dataset] Generating {num_games} training games...")
    samples = []
    for g in range(num_games):
        board = chess.Board()
        move_history = []
        board_states = []

        while not board.is_game_over() and len(move_history) < 80:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Simple material-aware move selection for sample generation
            best_move = legal_moves[0]
            best_score = -99999
            for m in legal_moves:
                board.push(m)
                # Quick material count
                score = 0
                for sq, piece in board.piece_map().items():
                    val = [0, 100, 320, 330, 500, 900, 20000][piece.piece_type]
                    score += val if piece.color == board.turn else -val
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = m

            board_states.append((board_to_tensor(board), move_to_index(best_move)))
            board.push(best_move)
            move_history.append(best_move)

        # Game outcome
        result = board.result()
        if result == "1-0":
            outcome = 1.0
        elif result == "0-1":
            outcome = -1.0
        else:
            outcome = 0.0

        for state_tensor, move_idx in board_states:
            samples.append((state_tensor, move_idx, outcome))

    print(f"[Dataset] Generated {len(samples)} position samples.")
    return samples

def train_chess_brain(epochs: int = 5, batch_size: int = 64, lr: float = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using computing device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Generate or load dataset
    samples = generate_synthetic_training_data(num_games=100)
    dataset = ChessPositionDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = ChessBrainNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    best_loss = float("inf")
    save_path = MODELS_DIR / "chess_brain_best.pt"

    print(f"[Training] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        p_loss_sum = 0.0
        v_loss_sum = 0.0

        for batch_x, target_policy, target_value in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            batch_x = batch_x.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.unsqueeze(1).to(device)

            optimizer.zero_grad()
            pred_policy, pred_value = model(batch_x)

            loss_p = policy_criterion(pred_policy, target_policy)
            loss_v = value_criterion(pred_value, target_value)
            loss = loss_p + loss_v

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            p_loss_sum += loss_p.item()
            v_loss_sum += loss_v.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} complete | Total Loss: {avg_loss:.4f} (Policy: {p_loss_sum/len(dataloader):.4f}, Value: {v_loss_sum/len(dataloader):.4f})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"[Checkpoint] Model saved to {save_path}")

    print(f"[Training] Complete! Best model saved to {save_path}")

if __name__ == "__main__":
    train_chess_brain(epochs=3)
