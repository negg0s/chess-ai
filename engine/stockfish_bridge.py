"""Stockfish bridge: process management, download, move search."""
import atexit
import hashlib
import logging
import zipfile
import urllib.request
from pathlib import Path

import chess
import chess.engine

logger = logging.getLogger("chessbot.stockfish")

ENGINE_DIR = Path(__file__).resolve().parent
STOCKFISH_EXE = ENGINE_DIR / "stockfish.exe"
STOCKFISH_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/"
    "stockfish-windows-x86-64-universal.zip"
)
# sha256 of the release zip; filled in only if verified by the author
# (see requirements notes), otherwise left empty and verification is skipped
STOCKFISH_ZIP_SHA256 = ""

_ACTIVE_ENGINES: list["StockfishEngine"] = []


def _atexit_shutdown() -> None:
    for engine in list(_ACTIVE_ENGINES):
        try:
            engine.close()
        except Exception:
            pass


atexit.register(_atexit_shutdown)


class StockfishEngine:
    """Owns a UCI engine subprocess.

    Path resolution order (CH-23): explicit `path_override` (from env
    STOCKFISH_PATH), then engine/stockfish.exe in the repo, then download.
    """

    def __init__(self, executable_path: Path | None = None, path_override: str = ""):
        if path_override:
            self.exe_path = Path(path_override)
        elif executable_path is not None:
            self.exe_path = executable_path
        else:
            self.exe_path = STOCKFISH_EXE
        self.engine: chess.engine.SimpleEngine | None = None
        _ACTIVE_ENGINES.append(self)

    def __enter__(self) -> "StockfishEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ensure_installed(self) -> bool:
        if self.exe_path.exists():
            return True

        logger.info("Binary not found at %s. Downloading...", self.exe_path)
        self.exe_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path = self.exe_path.parent / "stockfish.zip"

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(STOCKFISH_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()

            if STOCKFISH_ZIP_SHA256:
                digest = hashlib.sha256(data).hexdigest()
                if digest != STOCKFISH_ZIP_SHA256:
                    raise RuntimeError(f"Checksum mismatch: {digest} != {STOCKFISH_ZIP_SHA256}")
            else:
                logger.warning("No checksum configured for the Stockfish release zip; "
                               "integrity is not verified.")

            with zipfile.ZipFile(zip_path, "r") as archive:
                member = next(
                    (name for name in archive.namelist()
                     if name.lower().endswith(".exe") and "stockfish" in Path(name).name.lower()),
                    None,
                )
                if member is None:
                    raise RuntimeError("No stockfish *.exe found inside the archive")
                with archive.open(member) as source, open(self.exe_path, "wb") as target:
                    target.write(source.read())

            zip_path.unlink(missing_ok=True)
            logger.info("Downloaded to %s", self.exe_path)
            return True
        except Exception:
            logger.exception("Stockfish download failed")
            return False

    def start(self) -> bool:
        """Starts the UCI engine process."""
        if self.engine is not None:
            return True

        if not self.exe_path.exists() and not self.ensure_installed():
            logger.error("Could not start engine: binary missing.")
            return False

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(str(self.exe_path))
            logger.info("Engine active: %s", self.engine.id.get("name"))
            return True
        except Exception:
            logger.exception("Failed to start Stockfish at %s", self.exe_path)
            self.engine = None
            return False

    def get_best_move(self, board_or_fen, depth: int = 20, time_limit: float = 1.0) -> dict:
        """Analyses a position and returns best move, score and PV.

        The context manager form (`with StockfishEngine() as engine:`) is the
        preferred way to guarantee the subprocess is terminated (CH-23).
        """
        if self.engine is None:
            self.start()
        if self.engine is None:
            raise RuntimeError("Stockfish engine is not available.")

        board = chess.Board(board_or_fen) if isinstance(board_or_fen, str) else board_or_fen

        result = self.engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))

        score = result["score"].relative
        best_move = result.get("pv", [None])[0]
        pv_uci = [m.uci() for m in result.get("pv", [])]

        return {
            "best_move": best_move.uci() if best_move else None,
            "score": str(score),
            "depth": result.get("depth", depth),
            "pv": pv_uci,
            "turn": "White" if board.turn == chess.WHITE else "Black",
        }

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                logger.exception("Error while quitting engine")
            finally:
                self.engine = None
        if self in _ACTIVE_ENGINES:
            _ACTIVE_ENGINES.remove(self)
