import os
import sys
import zipfile
import urllib.request
from pathlib import Path
import chess
import chess.engine

ENGINE_DIR = Path(__file__).resolve().parent
STOCKFISH_EXE = ENGINE_DIR / "stockfish.exe"
STOCKFISH_URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-windows-x86-64-universal.zip"

class StockfishEngine:
    def __init__(self, executable_path: Path = STOCKFISH_EXE):
        self.exe_path = executable_path
        self.engine: chess.engine.SimpleEngine = None

    def ensure_installed(self):
        if self.exe_path.exists():
            return True

        print(f"[Stockfish] Binary not found at {self.exe_path}. Downloading...")
        ENGINE_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = ENGINE_DIR / "stockfish.zip"

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(STOCKFISH_URL, headers=headers)
            with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_file:
                out_file.write(resp.read())

            with zipfile.ZipFile(zip_path, "r") as z:
                for member in z.namelist():
                    if member.endswith(".exe"):
                        with z.open(member) as source, open(self.exe_path, "wb") as target:
                            target.write(source.read())
                        break

            if zip_path.exists():
                zip_path.unlink()

            print(f"[Stockfish] Downloaded to {self.exe_path}")
            return True
        except Exception as e:
            print(f"[Stockfish] Download error: {e}")
            return False

    def start(self):
        """Starts the UCI engine process."""
        if not self.exe_path.exists():
            self.ensure_installed()

        if self.exe_path.exists():
            self.engine = chess.engine.SimpleEngine.popen_uci(str(self.exe_path))
            # Configure threads & hash
            self.engine.configure({"Threads": 4, "Hash": 128})
            print(f"[StockfishEngine] Engine active: {self.engine.id.get('name')}")
        else:
            print("[StockfishEngine] Could not start engine: binary missing.")

    def get_best_move(self, board_or_fen, depth: int = 20, time_limit: float = 1.0):
        """
        Analyzes position and returns:
          - best_move: chess.Move or uci string
          - evaluation: score in centipawns or mate
          - pv: list of expected principal variation moves
        """
        if self.engine is None:
            self.start()

        if self.engine is None:
            raise RuntimeError("Stockfish engine is not available.")

        if isinstance(board_or_fen, str):
            board = chess.Board(board_or_fen)
        else:
            board = board_or_fen

        limit = chess.engine.Limit(depth=depth, time=time_limit)
        result = self.engine.analyse(board, limit)

        score = result["score"].relative
        best_move = result.get("pv", [None])[0]
        pv_uci = [m.uci() for m in result.get("pv", [])]

        return {
            "best_move": best_move.uci() if best_move else None,
            "score": str(score),
            "depth": result.get("depth", depth),
            "pv": pv_uci,
            "turn": "White" if board.turn == chess.WHITE else "Black"
        }

    def close(self):
        if self.engine:
            self.engine.quit()
            self.engine = None
