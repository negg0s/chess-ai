import os
import sys
import time
import json
import random
import logging
import threading
from pathlib import Path

import requests
import chess
import torch

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:  # running without python-dotenv: rely on real env vars
    pass

from config import MODEL_FILENAME, ensure_dirs, STOCKFISH_PATH
from engine.stockfish_bridge import StockfishEngine
from brain.model import ChessBrainNet
from brain.encoder import board_to_tensor, policy_best_move

ensure_dirs()

logger = logging.getLogger("chessbot")
if not logger.handlers:
    _logs_dir = ROOT_DIR / "logs"
    _logs_dir.mkdir(exist_ok=True)
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _file_handler = logging.FileHandler(_logs_dir / "bot.log", encoding="utf-8")
    _file_handler.setFormatter(_formatter)
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

TOKEN = os.getenv("LICHESS_TOKEN", "")
BASE_URL = "https://lichess.org"
BOOK_PATH = ROOT_DIR / "data" / "grandmaster_book.json"
MODEL_PATH = ROOT_DIR / "models" / MODEL_FILENAME

AUTO_MATCHMAKING = os.getenv("AUTO_MATCHMAKING", "1") not in ("0", "false", "no")
USE_NEURAL_FALLBACK = os.getenv("USE_NEURAL_FALLBACK", "0") not in ("0", "false", "no")


def compute_time_allocation(time_left_sec: float, increment_sec: float = 0.0):
    """Clock-aware thinking budget.

    Returns (time_alloc_seconds, search_depth). The increment is folded into
    the "safe to spend" estimate for passable time; below 5 seconds on the
    clock the bot plays fast and shallow.
    """
    time_left_sec = max(0.0, float(time_left_sec or 0.0))
    increment_sec = max(0.0, float(increment_sec or 0.0))
    budget = time_left_sec + min(increment_sec, 10.0)

    if budget < 5.0:
        return 0.05, 12
    if budget < 15.0:
        return 0.15, 16
    if budget < 30.0:
        return 0.5, 18
    return min(budget / 25.0, 2.0), 20


class LichessBot:
    def __init__(self, token: str = TOKEN):
        self.token = token
        if not self.token:
            logger.warning("LICHESS_TOKEN is empty. Set it in .env or pass as argument.")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "ChessBot/1.0"
        })
        self.bot_id = ""
        self.engine = StockfishEngine(path_override=STOCKFISH_PATH)
        self.engine_lock = threading.Lock()
        self.games_lock = threading.Lock()
        self.active_games = set()
        self.last_handled_ply = {}
        self.opening_book = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_brain = ChessBrainNet().to(self.device)
        self.neural_ready = False
        self._load_neural_model()
        self._load_opening_book()

    def _load_neural_model(self):
        if not MODEL_PATH.exists():
            logger.warning("Model file %s not found; neural fallback disabled.", MODEL_PATH.name)
            return
        try:
            try:
                state = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
            except TypeError:  # torch < 2.0 has no weights_only argument
                state = torch.load(MODEL_PATH, map_location=self.device)
            self.neural_brain.load_state_dict(state)
            self.neural_brain.eval()
            self.neural_ready = True
            logger.info("Loaded model (%s) on %s", MODEL_PATH.name, self.device)
        except Exception:
            logger.exception("Model load error for %s", MODEL_PATH.name)

    def _load_opening_book(self):
        if not BOOK_PATH.exists():
            logger.info("No opening book at %s (see data/book.example.json).", BOOK_PATH.name)
            return
        try:
            with open(BOOK_PATH, "r", encoding="utf-8") as f:
                self.opening_book = json.load(f)
            logger.info("Loaded opening book (%d positions)", len(self.opening_book))
        except Exception:
            logger.exception("Book load error for %s", BOOK_PATH.name)

    def neural_move(self, board: chess.Board) -> chess.Move | None:
        """Pick a move from the policy head; requires a loaded model."""
        if not self.neural_ready:
            return None
        tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, _ = self.neural_brain(tensor)
        return policy_best_move(policy[0], board)

    def start(self, increment_sec: float = 0.0):
        """Authenticate, resolve own id (CH-11) and start main loop."""
        logger.info("Starting Stockfish...")
        self.engine.start()
        if self.engine.engine:
            self.engine.engine.configure({
                "Threads": 4,
                "Hash": 512,
                "Move Overhead": 200
            })

        resp = self.session.get(f"{BASE_URL}/api/account")
        if resp.status_code != 200:
            logger.error("Authentication failed: %s", resp.text)
            return

        account = resp.json()
        self.bot_id = str(account.get("username", "")).lower()
        if os.getenv("LICHESS_BOT_ID"):
            self.bot_id = os.getenv("LICHESS_BOT_ID").lower()
        logger.info("Connected as: %s", account.get("username"))

        if AUTO_MATCHMAKING:
            threading.Thread(target=self.auto_matchmaker, daemon=True).start()

        logger.info("Listening for games...")
        while True:
            try:
                self.listen_events()
            except Exception:
                logger.exception("Connection error. Reconnecting in 10s...")
                time.sleep(10)

    def auto_matchmaker(self):
        target_bots = ["maia1", "maia5", "maia9"]
        time.sleep(5)
        while True:
            time.sleep(random.uniform(8, 18))
            with self.games_lock:
                has_games = len(self.active_games) > 0
            if has_games:
                continue
            target = random.choice(target_bots)
            try:
                r = self.session.post(f"{BASE_URL}/api/challenge/{target}", data={
                    "clock.limit": 180,
                    "clock.increment": 2,
                    "rated": "false"
                })
                if r.status_code == 429:
                    logger.warning("Rate limited by lichess, waiting 60s.")
                    time.sleep(60)
                elif r.status_code == 200:
                    logger.info("Challenge sent to %s", target)
                    time.sleep(20)
                else:
                    time.sleep(6)
            except Exception:
                time.sleep(6)

    def listen_events(self):
        url = f"{BASE_URL}/api/stream/event"
        resp = self.session.get(url, stream=True, timeout=(10, None))
        if resp.status_code == 429:
            logger.warning("Rate limited. Waiting 60s...")
            time.sleep(60)
            return
        if resp.status_code != 200:
            logger.error("Stream event error %s", resp.status_code)
            time.sleep(15)
            return

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                event = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            event_type = event.get("type")
            if event_type == "challenge":
                ch = event.get("challenge", {})
                ch_id = ch.get("id")
                challenger_id = str(ch.get("challenger", {}).get("id", "")).lower()
                challenger_name = ch.get("challenger", {}).get("name", "Unknown")
                variant_key = str(ch.get("variant", {}).get("key", "standard")).lower()

                # only supported rules variants (CH-10)
                if variant_key not in ("standard", "fromPosition"):
                    logger.info("Declining %s challenge (variant %s not supported)", challenger_name, variant_key)
                    self.session.post(f"{BASE_URL}/api/challenge/{ch_id}/decline",
                                      json={"reason": "variantUnsupported"})
                    continue

                if challenger_id == self.bot_id:
                    continue

                with self.games_lock:
                    has_games = len(self.active_games) >= 1
                if has_games:
                    self.session.post(f"{BASE_URL}/api/challenge/{ch_id}/decline", json={"reason": "later"})
                else:
                    logger.info("Accepting challenge from %s", challenger_name)
                    self.accept_challenge(ch_id)

            elif event_type == "gameStart":
                game_info = event.get("game", {})
                game_id = game_info.get("id")
                with self.games_lock:
                    new_game = game_id and game_id not in self.active_games
                    if new_game:
                        self.active_games.add(game_id)
                if new_game:
                    threading.Thread(target=self.play_game, args=(game_id,), daemon=True).start()

    def accept_challenge(self, challenge_id: str):
        url = f"{BASE_URL}/api/challenge/{challenge_id}/accept"
        resp = self.session.post(url)
        if resp.status_code == 200:
            logger.info("Accepted challenge %s", challenge_id)

    def play_game(self, game_id: str):
        logger.info("[Game %s] Connected", game_id)
        url = f"{BASE_URL}/api/bot/game/stream/{game_id}"
        resp = self.session.get(url, stream=True, timeout=(10, None))
        if resp.status_code != 200:
            with self.games_lock:
                self.active_games.discard(game_id)
            return

        bot_color = None

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            event_type = data.get("type")

            if event_type == "gameFull":
                white_id = str(data.get("white", {}).get("id", "")).lower()
                bot_color = chess.WHITE if white_id == self.bot_id else chess.BLACK
                logger.info("[Game %s] Playing as %s", game_id,
                            "White" if bot_color == chess.WHITE else "Black")

                state = data.get("state", {})
                moves = state.get("moves", "")
                wtime = state.get("wtime", 180000)
                btime = state.get("btime", 180000)
                winc = state.get("winc", 0)
                binc = state.get("binc", 0)
                clock = wtime if bot_color == chess.WHITE else btime
                inc = winc if bot_color == chess.WHITE else binc
                self.handle_turn(game_id, moves, bot_color, clock, inc)

            elif event_type == "gameState":
                status = data.get("status")
                if status != "started":
                    logger.info("[Game %s] Finished: %s", game_id, status)
                    break

                moves = data.get("moves", "")
                wtime = data.get("wtime", 180000)
                btime = data.get("btime", 180000)
                winc = data.get("winc", 0)
                binc = data.get("binc", 0)
                clock = wtime if bot_color == chess.WHITE else btime
                inc = winc if bot_color == chess.WHITE else binc
                self.handle_turn(game_id, moves, bot_color, clock, inc)

        with self.games_lock:
            self.active_games.discard(game_id)
        self.last_handled_ply.pop(game_id, None)

    def handle_turn(self, game_id: str, moves_str: str, bot_color: chess.Color,
                    clock_ms: int = 0, increment_ms: int = 0):
        board = chess.Board()
        move_list = moves_str.strip().split() if moves_str else []
        for m in move_list:
            try:
                board.push_uci(m)
            except Exception:
                break

        if board.turn != bot_color or board.is_game_over():
            return

        current_ply_count = len(move_list)
        with self.games_lock:
            if self.last_handled_ply.get(game_id) == current_ply_count:
                return
            self.last_handled_ply[game_id] = current_ply_count

        turn_num = board.fullmove_number
        fen_key = board.board_fen()

        if fen_key in self.opening_book:
            book_move = self.opening_book[fen_key]
            if chess.Move.from_uci(book_move) in board.legal_moves:
                logger.info("[Game %s] Move %d (Book): %s", game_id, turn_num, book_move)
                time.sleep(0.1)
                self.make_move(game_id, book_move)
                return

        time_alloc, search_depth = compute_time_allocation(
            clock_ms / 1000.0, increment_ms / 1000.0
        )

        with self.engine_lock:
            try:
                res = self.engine.get_best_move(board, depth=search_depth, time_limit=time_alloc)
                best_move = res.get("best_move")
                score = res.get("score")
                logger.info("[Game %s] Move %d: %s (Eval: %s)", game_id, turn_num, best_move, score)
                if best_move:
                    self.make_move(game_id, best_move)
                    return
            except Exception:
                logger.exception("[Game %s] Move calculation error", game_id)

        # fallback move (CH-09): neural policy if enabled, else first legal
        try:
            fallback = None
            if USE_NEURAL_FALLBACK:
                fallback = self.neural_move(board)
            if fallback is None:
                fallback = next(iter(board.legal_moves))
            logger.info("[Game %s] Move %d (Fallback): %s", game_id, turn_num, fallback.uci())
            self.make_move(game_id, fallback.uci())
        except StopIteration:
            logger.error("[Game %s] No legal move available", game_id)

    def make_move(self, game_id: str, move_uci: str):
        url = f"{BASE_URL}/api/bot/game/{game_id}/move/{move_uci}"
        try:
            resp = self.session.post(url)
            if resp.status_code != 200:
                logger.error("[Game %s] Move error: %s", game_id, resp.text)
        except requests.RequestException:
            logger.exception("[Game %s] Move request failed", game_id)


if __name__ == "__main__":
    LichessBot().start()
