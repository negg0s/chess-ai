import os
import sys
import time
import json
import random
import threading
from pathlib import Path
import requests
import chess
import torch

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

env_file = ROOT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from engine.stockfish_bridge import StockfishEngine
from brain.model import ChessBrainNet
from brain.encoder import board_to_tensor

TOKEN = os.getenv("LICHESS_TOKEN", "")
BOT_ID = os.getenv("LICHESS_BOT_ID", "")
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "User-Agent": "ChessBot/1.0"
}
BASE_URL = "https://lichess.org"
BOOK_PATH = ROOT_DIR / "data" / "grandmaster_book.json"
MODEL_PATH = ROOT_DIR / "models" / "chess_brain_ultra.pt"

class LichessBot:
    def __init__(self, token: str = TOKEN):
        self.token = token
        if not self.token:
            print("[Bot] LICHESS_TOKEN is empty. Set it in .env or pass as argument.", flush=True)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "ChessBot/1.0"
        })
        self.engine = StockfishEngine()
        self.engine_lock = threading.Lock()
        self.active_games = set()
        self.last_handled_ply = {}
        self.opening_book = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neural_brain = ChessBrainNet().to(self.device)
        if MODEL_PATH.exists():
            try:
                self.neural_brain.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                self.neural_brain.eval()
                print(f"[Bot] Loaded model ({MODEL_PATH.name}) on {self.device}", flush=True)
            except Exception as e:
                print(f"[Bot] Model load error: {e}", flush=True)

        if BOOK_PATH.exists():
            try:
                with open(BOOK_PATH, "r", encoding="utf-8") as f:
                    self.opening_book = json.load(f)
                print(f"[Bot] Loaded opening book ({len(self.opening_book)} positions)", flush=True)
            except Exception as e:
                print(f"[Bot] Book load error: {e}", flush=True)

    def start(self):
        print("[Bot] Starting Stockfish...", flush=True)
        self.engine.start()
        if self.engine.engine:
            self.engine.engine.configure({
                "Threads": 4,
                "Hash": 512,
                "Move Overhead": 200
            })

        resp = self.session.get(f"{BASE_URL}/api/account")
        if resp.status_code != 200:
            print(f"[Bot] Authentication failed: {resp.text}", flush=True)
            return

        acc = resp.json()
        print(f"[Bot] Connected as: {acc.get('username')}", flush=True)
        print("[Bot] Listening for games...", flush=True)

        threading.Thread(target=self.auto_matchmaker, daemon=True).start()

        while True:
            try:
                self.listen_events()
            except Exception as e:
                print(f"[Bot] Connection error: {e}. Reconnecting in 10s...", flush=True)
                time.sleep(10)

    def auto_matchmaker(self):
        target_bots = ["maia1", "maia5", "maia9"]
        time.sleep(5)
        while True:
            time.sleep(12)
            if len(self.active_games) == 0:
                target = random.choice(target_bots)
                try:
                    r = self.session.post(f"{BASE_URL}/api/challenge/{target}", data={
                        "clock.limit": 180,
                        "clock.increment": 2,
                        "rated": "false"
                    })
                    if r.status_code == 200:
                        print(f"[Bot] Challenge sent to {target}", flush=True)
                        time.sleep(20)
                    else:
                        time.sleep(6)
                except Exception as e:
                    time.sleep(6)

    def listen_events(self):
        url = f"{BASE_URL}/api/stream/event"
        resp = self.session.get(url, stream=True, timeout=(10, None))
        if resp.status_code == 429:
            print("[Bot] Rate limited. Waiting 60s...", flush=True)
            time.sleep(60)
            return
        if resp.status_code != 200:
            print(f"[Bot] Stream event error {resp.status_code}", flush=True)
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

                if challenger_id == BOT_ID.lower():
                    continue

                if len(self.active_games) >= 1:
                    self.session.post(f"{BASE_URL}/api/challenge/{ch_id}/decline", json={"reason": "later"})
                else:
                    print(f"[Bot] Accepting challenge from {challenger_name}", flush=True)
                    self.accept_challenge(ch_id)

            elif event_type == "gameStart":
                game_info = event.get("game", {})
                game_id = game_info.get("id")
                if game_id and game_id not in self.active_games:
                    self.active_games.add(game_id)
                    t = threading.Thread(target=self.play_game, args=(game_id,), daemon=True)
                    t.start()

    def accept_challenge(self, challenge_id: str):
        url = f"{BASE_URL}/api/challenge/{challenge_id}/accept"
        resp = self.session.post(url)
        if resp.status_code == 200:
            print(f"[Bot] Accepted challenge {challenge_id}", flush=True)

    def play_game(self, game_id: str):
        print(f"[Game {game_id}] Connected", flush=True)
        url = f"{BASE_URL}/api/bot/game/stream/{game_id}"
        resp = self.session.get(url, stream=True, timeout=(10, None))
        if resp.status_code != 200:
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
                bot_color = chess.WHITE if white_id == BOT_ID.lower() else chess.BLACK
                print(f"[Game {game_id}] Playing as {'White' if bot_color == chess.WHITE else 'Black'}", flush=True)

                moves = data.get("state", {}).get("moves", "")
                wtime = data.get("state", {}).get("wtime", 180000)
                btime = data.get("state", {}).get("btime", 180000)
                self.handle_turn(game_id, moves, bot_color, wtime if bot_color == chess.WHITE else btime)

            elif event_type == "gameState":
                status = data.get("status")
                if status != "started":
                    print(f"[Game {game_id}] Finished: {status}", flush=True)
                    break

                moves = data.get("moves", "")
                wtime = data.get("wtime", 180000)
                btime = data.get("btime", 180000)
                clock = wtime if bot_color == chess.WHITE else btime
                self.handle_turn(game_id, moves, bot_color, clock)

        self.active_games.discard(game_id)
        self.last_handled_ply.pop(game_id, None)

    def handle_turn(self, game_id: str, moves_str: str, bot_color: chess.Color, clock_ms: int):
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
        if self.last_handled_ply.get(game_id) == current_ply_count:
            return

        self.last_handled_ply[game_id] = current_ply_count
        turn_num = board.fullmove_number
        fen_key = board.board_fen()

        if fen_key in self.opening_book:
            book_move = self.opening_book[fen_key]
            if chess.Move.from_uci(book_move) in board.legal_moves:
                print(f"[Game {game_id}] Move {turn_num} (Book): {book_move}", flush=True)
                time.sleep(0.1)
                self.make_move(game_id, book_move)
                return

        time_left_sec = clock_ms / 1000.0 if clock_ms else 60.0
        if time_left_sec < 5.0:
            time_alloc = 0.05
            search_depth = 12
        elif time_left_sec < 15.0:
            time_alloc = 0.15
            search_depth = 16
        elif time_left_sec < 30.0:
            time_alloc = 0.5
            search_depth = 18
        else:
            time_alloc = min(time_left_sec / 25.0, 2.0)
            search_depth = 20

        with self.engine_lock:
            try:
                res = self.engine.get_best_move(board, depth=search_depth, time_limit=time_alloc)
                best_move = res.get("best_move")
                score = res.get("score")
                print(f"[Game {game_id}] Move {turn_num}: {best_move} (Eval: {score})", flush=True)
                if best_move:
                    self.make_move(game_id, best_move)
            except Exception as e:
                print(f"[Game {game_id}] Move calculation error: {e}", flush=True)

    def make_move(self, game_id: str, move_uci: str):
        url = f"{BASE_URL}/api/bot/game/{game_id}/move/{move_uci}"
        resp = self.session.post(url)
        if resp.status_code != 200:
            print(f"[Game {game_id}] Move error: {resp.text}", flush=True)

if __name__ == "__main__":
    bot = LichessBot()
    bot.start()
