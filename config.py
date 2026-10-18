r"""Project layout and runtime configuration for the chess bot."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VISION_DIR = BASE_DIR / "vision"
BRAIN_DIR = BASE_DIR / "brain"
ENGINE_DIR = BASE_DIR / "engine"

# Weights file the bot loads. train.py writes chess_brain_best.pt via the
# checkpoint helper; point MODEL_FILENAME at whichever file you want the bot
# to use. Override through .env (MODEL_FILENAME=...).
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "chess_brain_ultra.pt")

# Optional env overrides
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "")
LICHESS_BOT_ID = os.getenv("LICHESS_BOT_ID", "")

_DIRS_CREATED = False


def ensure_dirs() -> None:
    """Create project directories on first call (no work at import time)."""
    global _DIRS_CREATED
    if not _DIRS_CREATED:
        for d in (DATA_DIR, MODELS_DIR, VISION_DIR, BRAIN_DIR, ENGINE_DIR):
            d.mkdir(parents=True, exist_ok=True)
        _DIRS_CREATED = True
