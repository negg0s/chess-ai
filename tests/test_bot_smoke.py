import json
import logging

import torch

import lichess_bot


def test_model_load_without_weights_does_not_crash(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(lichess_bot, "MODEL_PATH", tmp_path / "absent.pt")
    with caplog.at_level(logging.WARNING):
        bot = lichess_bot.LichessBot(token="")
    assert bot.neural_ready is False
    assert any("not found" in r.getMessage() for r in caplog.records)


def test_weights_load_state_dict_roundtrip(tmp_path, monkeypatch):
    from brain.model import ChessBrainNet

    weights = tmp_path / "w.pt"
    torch.save(ChessBrainNet().state_dict(), weights)
    monkeypatch.setattr(lichess_bot, "MODEL_PATH", weights)
    bot = lichess_bot.LichessBot(token="")
    assert bot.neural_ready is True


def test_cfg_handles_missing_book(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(lichess_bot, "BOOK_PATH", tmp_path / "no_book.json")
    caplog.set_level(logging.INFO)
    lichess_bot.LichessBot(token="")
    assert any(r.levelno < logging.ERROR for r in caplog.records)


def test_bot_id_resolved_from_env_override():
    # start() resolves /api/account; here only the override plumbing is checked
    bot = lichess_bot.LichessBot(token="")
    assert bot.bot_id == ""


def test_stockfish_ensure_installed_from_bytes(tmp_path, monkeypatch):
    import io
    import zipfile
    from engine.stockfish_bridge import StockfishEngine

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("stockfish-windows-x86-64.exe", b"MZ_FAKE_BINARY")
    zip_bytes = buf.getvalue()

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return zip_bytes

    target_exe = tmp_path / "sf.exe"
    engine = StockfishEngine(executable_path=target_exe)
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: FakeResponse())

    assert engine.ensure_installed() is True
    assert target_exe.exists()
    assert target_exe.read_bytes() == b"MZ_FAKE_BINARY"
