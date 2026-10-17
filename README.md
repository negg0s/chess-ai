# chess-ai

Шахматный бот для Lichess: двигателем выступает **Stockfish**, нейронная сеть
(ResNet policy-value) — экспериментальная часть, по умолчанию в выборе хода
не участвует. Есть CLI для анализа позиций и пайплайн обучения на
синтетических данных.

> **См. также**: `CLEANUP_PROGRESS.md` (список правок этой ветки),
> `docs/model-eval.md` (проверка весов модели, проведена в этой ветке).

## Что делает бот

- слушает `gameStart` и входящие челленджи (`/api/stream/event`);
- играет **только обычные правила**: chess960 и другие варианты отклоняются
  с причиной `variantUnsupported`;
- выбор хода: дебютная книга (если загружена) → Stockfish (глубина и время
  зависят от оставшихся часов) → запасной ход (первый легальный);
- потокобезопасно: общий за один раз матч, `engine_lock` на доступ к
  Stockfish, `games_lock` на общие структуры;
- опционально разыгрывает вызовы ботам maia1/maia5/maia9
  (`AUTO_MATCHMAKING=1`, пауза с джиттером, обработка 429).

Токен нужен **бот-токен** (учётка должна быть зарегистрирована как bot
аккаунт): https://lichess.org/account/oauth/token/create.

## Установка

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`requirements-train.txt` дополнительно ставит tqdm/torchvision/opencv/Pillow —
они нужны только для обучения и модуля `vision/`.

## Запуск бота

1. Скопируйте `.env.example` в `.env` и впишите `LICHESS_TOKEN=...`;
2. Запустите `python lichess_bot.py` (или `START_BOT.bat`).

Переменные `.env` (все опциональны, кроме токена) описаны в
`.env.example`: `MODEL_FILENAME`, `STOCKFISH_PATH`, `AUTO_MATCHMAKING`,
`USE_NEURAL_FALLBACK`, `LICHESS_BOT_ID`.

Логи пишутся в `logs/bot.log` и в stderr.

## Крупные файлы

- `models/chess_brain_ultra.pt` (75 МБ) — веса нейросети, **коммитится в git**.
- `data/grandmaster_book.json` (65 МБ) — дебютная книга, **не в git**;
  см. `data/book.example.json` для формата: плоский JSON
  `{board_fen: uci_move}`, где `board_fen` — `board.board_fen()` (без полей).
- `engine/stockfish.exe` (103 МБ) — скачается автоматически при первом
  запуске (sf_19, Windows), но можно положить вручную; путь переопределяется
  через `STOCKFISH_PATH`.

## Замечание о нейросети

Веса `chess_brain_ultra.pt` по измерению из `docs/model-eval.md`
(top-1 14.7 % / Pearson ≈ 0 против Stockfish d8) не несут полезного
шахматного сигнала: сеть не может использовать их для анализа. Играет
только Stockfish; нейронную сеть можно включить как запасной ход
(`USE_NEURAL_FALLBACK=1`), однако лучше сначала переобучить модель.

## Обучение

```bash
python main.py --train --epochs 5 --seed 42 --num-games 100
```

Пайплайн (`brain/train.py`) ограничен синтетикой: greedy-material партии.
Известные ограничения:

- датасет объединяет синтетику (PGN-загрузки в репозитории нет);
- truncated партии получают NaN-метку value и исключаются из value-лосса
  (`require_finished=True` отфильтрует их целиком, но с determinism
  greedy-игры могут не доходить до мата за 80 полуходов);
- метки value по умолчанию «от белых»; `--value-perspective side_to_move`
  переключает на "от ходущего";
- чек-поинты сохраняются в файл с датой и **не перезаписывают** старые веса;
  бот грузит файл из `MODEL_FILENAME` (по умолчанию
  `chess_brain_ultra.pt`).

## Анализ позиций

```bash
python main.py --fen "r1bqk2r/pppp1ppp/2n5/4p3/1bB1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5"
```

Engine закрывается корректно (context manager), даже если ход анализа
прервался.

## Структура

```
brain/     encoder.py (FEN → 18x8x8, перемещения в 4096-индексах), model.py (ResNet), train.py
engine/    stockfish_bridge.py (запуск/скачивание Stockfish)
vision/    вспомогательный модуль (не используется ботом)
scripts/   eval_model.py — оценка весов (см. docs/model-eval.md)
tests/     юнит-тесты (pytest)
```

## Тесты

```bash
pip install pytest
pytest tests -q
```

24 теста покрывают энкодер (планы, раскодирование хода из 4096, ферзевые
превращения), данные (детерминизм по seed, метки), правила распределения
времени и загрузку бота без токена/весов/книги.

## License

Repository does not have a LICENSE file yet (author decision needed).
