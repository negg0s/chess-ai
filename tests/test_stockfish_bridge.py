from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

chess = pytest.importorskip("chess")
pytest.importorskip("chess.engine")

from engine.stockfish_bridge import StockfishEngine, _ACTIVE_ENGINES


def _create_mocked_engine() -> tuple[StockfishEngine, MagicMock]:
    engine = StockfishEngine(executable_path=Path("dummy_stockfish.exe"))
    mock_backend = MagicMock()
    engine.engine = mock_backend
    return engine, mock_backend


def test_get_best_move_empty_pv():
    engine, mock_backend = _create_mocked_engine()
    score_mock = MagicMock()
    score_mock.relative = "+35"
    mock_backend.analyse.return_value = {
        "pv": [],
        "score": score_mock,
        "depth": 10,
    }

    res = engine.get_best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert res["best_move"] is None
    assert res["pv"] == []
    assert res["score"] == "+35"
    assert res["depth"] == 10
    assert res["turn"] == "White"


def test_get_best_move_missing_pv():
    engine, mock_backend = _create_mocked_engine()
    score_mock = MagicMock()
    score_mock.relative = "-15"
    mock_backend.analyse.return_value = {
        "score": score_mock,
        "depth": 8,
    }

    res = engine.get_best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    assert res["best_move"] is None
    assert res["pv"] == []
    assert res["score"] == "-15"
    assert res["depth"] == 8
    assert res["turn"] == "Black"


def test_get_best_move_none_score():
    engine, mock_backend = _create_mocked_engine()
    mock_backend.analyse.return_value = {
        "pv": [],
        "score": None,
        "depth": 5,
    }

    res = engine.get_best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert res["best_move"] is None
    assert res["pv"] == []
    assert res["score"] == "0"
    assert res["depth"] == 5


def test_get_best_move_missing_score():
    engine, mock_backend = _create_mocked_engine()
    mock_backend.analyse.return_value = {
        "pv": [],
        "depth": 12,
    }

    res = engine.get_best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert res["best_move"] is None
    assert res["pv"] == []
    assert res["score"] == "0"
    assert res["depth"] == 12


def test_get_best_move_empty_dict():
    engine, mock_backend = _create_mocked_engine()
    mock_backend.analyse.return_value = {}

    res = engine.get_best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert res["best_move"] is None
    assert res["pv"] == []
    assert res["score"] == "0"
    assert res["depth"] == 20


def test_get_best_move_with_valid_pv():
    engine, mock_backend = _create_mocked_engine()
    move = chess.Move.from_uci("e2e4")
    score_mock = MagicMock()
    score_mock.relative = "+40"
    mock_backend.analyse.return_value = {
        "pv": [move],
        "score": score_mock,
        "depth": 18,
    }

    res = engine.get_best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert res["best_move"] == "e2e4"
    assert res["pv"] == ["e2e4"]
    assert res["score"] == "+40"
    assert res["depth"] == 18


def test_stockfish_engine_close():
    engine, mock_backend = _create_mocked_engine()
    assert engine in _ACTIVE_ENGINES

    engine.close()
    mock_backend.quit.assert_called_once()
    assert engine.engine is None
    assert engine not in _ACTIVE_ENGINES


def test_stockfish_engine_context_manager(monkeypatch):
    engine = StockfishEngine(executable_path=Path("dummy_stockfish.exe"))
    mock_backend = MagicMock()

    def fake_start():
        engine.engine = mock_backend
        return True

    monkeypatch.setattr(engine, "start", fake_start)

    with engine as active_engine:
        assert active_engine is engine
        assert active_engine.engine is mock_backend

    mock_backend.quit.assert_called_once()
    assert engine.engine is None
    assert engine not in _ACTIVE_ENGINES


def test_stockfish_engine_without_chess(monkeypatch):
    import engine.stockfish_bridge as sf

    monkeypatch.setattr(sf, "chess", None)
    inst = sf.StockfishEngine(executable_path=Path("dummy_stockfish.exe"))
    assert inst.start() is False
    with pytest.raises(RuntimeError, match="python-chess is not installed"):
        inst.get_best_move("8/8/8/8/8/8/8/8 w - - 0 1")
    inst.close()

