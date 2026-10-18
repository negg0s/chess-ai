# Cleanup progress: chess-ai
Base commit: 2cad2f4 (main) | Branch: cleanup/chess-ai | Started: 2026-10-18

## Environment
- Windows 11 build 26200. Установленный Python: 3.12.8 (системный).
- Авторский venv: `D:\chess_ai\venv` — torch 2.6.0+cu124, python-chess 1.11.2. Использовался «как есть» (только запуск eval).
- Отдельный test-venv: `D:\work\cleanup\venvs\chess-ai` (CPython 3.12.8, torch 2.14.1+cpu, python-chess 1.11.2, pytest 9.1.1) — для pytest, ничего в авторский venv не ставилось.
- Stockfish: sf_19 (`engine/stockfish.exe`, untracked, есть на диске); eval-задача реализована через него.

## Disk
- Before: C free 180.5 GB / D free 1180.29 GB
- After: <см. финальный отчёт>

## Tasks (CH-01..CH-29)
| ID | Что сделано | Где | Проверка |
|---|---|---|---|
| CH-01 | Randomness+seed в синтетике (10% случайных ходов, `--seed`), идентичность датасета для seeds | brain/train.py | тест test_dataset_determinism_with_seed + diversity |
| CH-02 | Знак материальной оценки: счёт теперь от стороны ходящего (после push мой материал — «не ходящая» сторона) | brain/train.py | тест test_material_sign_counts_mover_perspective (позиция с выигрышным гамбитом) |
| CH-03 | Перспектива value: по умолчанию «от белых» (как раньше был лейбл), `--value-perspective side_to_move` — от ходящего | brain/train.py | тест _value_label |
| CH-04 | Превращения: 4096-индексация без знания промошина задокументирована; при чтении политики пешка на последней превращается в ферзя (`policy_best_move`) | brain/encoder.py | тест test_policy_best_move_prefers_queen_promotion |
| CH-05 | Имя файла модели через config.MODEL_FILENAME (.env override), бот больше не хардкодит path | config.py, lichess_bot.py | тест init |
| CH-06 | Докстринг train.py — правда: только synthethic, PGN-загрузки нет | brain/train.py | рефактор |
| CH-07 | Валидационный сплит 90/10, model.eval(), torch.no_grad, seed фиксированный, checkpoint выбирается по val loss | brain/train.py | smoke-run (1 эпоха, checkpoint создана с датой) |
| CH-08 | truncated партии: value метка NaN и маскирование из value-loss (политика те же), require_finished — фильтрация целиком (документировано, что greedy-игры доводятся до прогресса редко) | brain/train.py | тесты require_finished / labels |
| CH-09 | Запасной ход: neural policy (если USE_NEURAL_FALLBACK=1), иначе первый легальный | lichess_bot.py | код + тесты |
| CH-10 | Варианты с черными правилами: только standard/fromPosition принимаются; бинарные расхождения с cause `variantUnsupported` | lichess_bot.py | код |
| CH-11 | BOT_ID резолвится из /api/account, env LOCHESS_BOT_ID полный override | lichess_bot.py | класс стартовое состояние == auth |
| CH-12 | auto_matchmaker: --flag AUTO_MATCHMAKING (по умолчанию 1), jitter 8-18с, реакция на 429 (60с пауза) | lichess_bot.py | код |
| CH-13 | torch.load weights_only=True (try/except TypeError для старых torch) | lichess_bot.py | тест roundtrip weights |
| CH-20 | print → logging (консоль + logs/bot.log, gitignored), формат со временем | lichess_bot.py, engine/stockfish_bridge.py | тест |
| CH-21 | games_lock на active_games/last_handled_ply (engine_lock уже был) | lichess_bot.py | код |
| CH-22 | python-dotenv вместо ручного парсера .env, config.ensure_dirs() вместо mkdir на импорте | config.py, lichess_bot.py | тесты не мешают импортам |
| CH-23 | Stockfish: путь из env-переменной STOCKFISH_PATH → engine/stockfish.exe → download; checksum hook (константа, пусто = warns); context manager + atexit_SHUTDOWN | engine/stockfish_bridge.py | код |
| CH-24 | main.py: engine через context manager (закрытие при любой ошибке), sys.exit код | main.py | код |
| CH-25 | Мёртвый код оставлен/удалён: HEADERS (нулевая польза), board_to_tensor импорт (есть использовать в нейронном fallback) | lichess_bot.py | diff |
| CH-26 | data/book.example.json — образец формата книги (плоский JSON board_fen → uci) | data/book.example.json | git tracked |
| CH-27 | compute_time_allocation — чист함 функция (time_left_sec, increment_sec), обеParty clock + инкремент учитываются из winc/binc | lichess_bot.py | тесты time |
| CH-28 | requirements.txt (бот) + requirements-train.txt (обучение+vision) | requirements* | pip install логика |
| CH-29 | .gitattributes + .editorconfig | корень | да |
| CH-00 | Оценка модели | **done** (docs/model-eval.md): 300 позиций, top-1 14.7%, top-3 28.0%, value Pearson = -0.0000 — веса не несут шахматного сигнала | scripts/eval_model.py |
| CH-CI | .github/workflows/ci.yml (ubuntu, py 3.11/3.12, CPU torch) | — | да |

## Defaults (решения по умолчанию)
- MODEL_FILENAME = chess_brain_ultra.pt (старое поведение бота).
- AUTO_MATCHMAKING = 1 (старое поведение), пауза с джиттером 8-18с.
- USE_NEURAL_FALLBACK = 0 (веса модели проверены, качество низкое).
- require_finished = False (старое поведение), метки NaN маскируются — поведение стало честнее и не меняет датасет полностью.

## needs-author (требует решения автора)
1. Переобучение модели после исправления пайплайна (CH-01/02/03/08) — измерять через scripts/eval_model.py.
2. Замена loading date-checkpoint в response файла datetime: нужно авторское имя файла (сейчас: имя файла берётся из MODEL_FILENAME).
3. Stockfish checksum: официальный релизный zip sha256 неизвестен (не публикуется per asset); константа STOCKFISH_ZIP_SHA256 оставлена пустой — предупреждающий лог. Author может вписать хэш сам.
4. CI на GitHub запустится только после push ветки (пустой runner нужно оценивать автору).

## Commit map (branch cleanup/chess-ai)
- 9c4d3b0 chore: .gitattributes/.editorconfig/.gitignore
- d4c2cf7 fix(encoder): promotion read, docs
- fb1d546 fix(train): seed/val/NaN labels
- 7b864c8 fix(bot): logging/locks/variant/time/fallback
- 64496c6 fix(engine): paths/checksum/shutdown
- 02df1a9 fix(cli): engine close + training flags
- 8dc17c8 chore(deps): split requirements
- 03db1ba docs: honest README + model-eval
- 21e1bb6 ci: pytest workflow
- 0838f06 docs: CLEANUP_PROGRESS

## Verification
- pytest: 24 passed (D:\work\cleanup\venvs\chess-ai, torch cpu 2.14.1).
- Training smoke: 1 эпоха (20 игр, seed 42, CPU) → checkpoint создан с датой, авторские файлы не тронуты.
- eval: авторский venv (CUDA), репорт в docs/model-eval.md.
