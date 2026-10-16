"""
Chess Piece Classifier (Computer Vision to FEN)
Classifies 64 sliced square images into 13 classes:
0: Empty (.)
1-6: White pieces (P, N, B, R, Q, K)
7-12: Black pieces (p, n, b, r, q, k)
Generates FEN notation from visual predictions.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

CLASSES = [
    ".",  # Empty
    "P", "N", "B", "R", "Q", "K",  # White
    "p", "n", "b", "r", "q", "k"   # Black
]

class PieceClassifierCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        return self.classifier(feats)

class VisualBoardParser:
    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = PieceClassifierCNN(num_classes=len(CLASSES)).to(self.device)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"[VisualBoardParser] Note: Model weights not loaded ({e}), using untrained/feature mode.")

    def preprocess_square(self, sq_img: np.ndarray, target_size: int = 64) -> torch.Tensor:
        """Resizes and normalizes an image square to [1, 3, target_size, target_size]."""
        resized = cv2.resize(sq_img, (target_size, target_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        # Normalize with standard ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0)

    def predict_square(self, sq_img: np.ndarray) -> str:
        """Classifies a single square into a piece symbol or empty '.'."""
        tensor = self.preprocess_square(sq_img).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
        return CLASSES[pred_idx]

    def grid_to_fen(self, grid_symbols: List[List[str]], turn: str = "w", castling: str = "KQkq", ep: str = "-") -> str:
        """
        Converts 8x8 matrix of piece characters into FEN string.
        grid_symbols[0][0] is a8, grid_symbols[7][7] is h1.
        """
        fen_rows = []
        for row in grid_symbols:
            empty_count = 0
            row_str = ""
            for char in row:
                if char == ".":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += char
            if empty_count > 0:
                row_str += str(empty_count)
            fen_rows.append(row_str)

        board_fen = "/".join(fen_rows)
        full_fen = f"{board_fen} {turn} {castling} {ep} 0 1"
        return full_fen
