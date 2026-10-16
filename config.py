r"""
Chess AI & Vision Project Configuration
All models, weights, data, and cache are strictly stored on D:\chess_ai
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VISION_DIR = BASE_DIR / "vision"
BRAIN_DIR = BASE_DIR / "brain"
ENGINE_DIR = BASE_DIR / "engine"

for d in [DATA_DIR, MODELS_DIR, VISION_DIR, BRAIN_DIR, ENGINE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
