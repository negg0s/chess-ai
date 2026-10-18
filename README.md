# chess-ai

Шахматный бот для Lichess: полноценный движок **Stockfish**, опциональная
нейросеть (ResNet policy-value) — экспериментальная часть, по умолчанию не
влияет на ходы. Есть CLI для анализа позиций и запуска обучения на
синтетических партиях.

> **См. также**: `CLEANUP_PROGRESS.md` (список правок этой ветки) и
> `docs/model-eval.md` (проверка модели, воспроизводимая в этой же ветке).

## Как работает бот

- Слушает `gameStart` из входного потока (`/api/stream/event`);
- Открывает **только стандартные шахматы**: chess960 и прочие варианты
  отклоняются с причиной `variantUnsupported`;
- Выбор хода: дебютная книга (если есть) → Stockfish (леммит по времени
  от оставшихся часов) → аварийный ход (если движок подвёл);
- Паукообразность: игрока не чаще одного раза, `engine_lock` для стоящих
  в очереди ходов, `games_lock` для параллельных партий;
- Автоматизированные вызовы соперникам maia1/maia5/maia9
  (`AUTO_MATCHMAKING=1`), обратный отсчёт и обработка 429.

Токен бота нужен **бот-токен** (страница регистрации аккаунта ботом):
https://lichess.org/account/oauth/token/create.

## Установка

```bash
python -m venv venv
venv\Scripts\activate          # Windows (source venv/bin/activate в Linux)
pip install -r requirements.txt
```

Движок Stockfish: положите `stockfish.exe` в `engine/` (первый запуск
ищет в `engine/`, затем в PATH; может скачать сам, `.env` укажите
`STOCKFISH_PATH=...` чтобы фиксированный путь).

## Запуск бота

```bash
copy .env.example .env         # вписать LICHESS_TOKEN, при желании стальные
python lichess_bot.py
```

## Обучение нейросети (экспериментально)

```bash
pip install -r requirements-train.txt
python main.py --train --num-games 2500 --epochs 20 --seed 42
# либо напрямую: python brain/train.py --num-games 2500 --epochs 20 --seed 42
```

Синтетические партии — детерминированный жадный набор материала с 10 %
случайных ходов (seed сквозной через seed). Датасет от датасета
отличается: value-label — NaN, если партия не доиграна, для value-loss
недоигранные позиции маскируются (не «ноль», не «в нулях» как прежде).
`require_finished=True` в этой позиции отфильтрует почти всё — жадные
партии редко успевают закончиться за 80 полуходов.

## Оценка текущей модели (воспроизводимая)

`scripts/eval_model.py --num-positions 300` на GPU (seed 42):

| Метрика | chess_brain_ultra.pt (старая) | chess_brain_20261018.pt (после фиксов) |
|---|---|---|
| top-1 против Stockfish d8 | 14.7 % | **20.0 %** |
| top-3 против Stockfish d8 | 28.0 % | **38.0 %** |
| | Pearson value ≈ 0 | **0.44** |

Вывод: фиксы пайплайна дали реальный сигнал, но модель всё ещё далеко
от шахматной силы — данные (жадный материал) её ограничивают, val-loss
растёт после ~2 эпох на этом синтетическом наборе. Подробности:
`docs/model-eval.md`.

## Тесты / CI

```bash
pytest tests -q
```

GitHub Actions: CPU-torch, Python 3.11/3.12, `pytest` + smoke-обучение.

## Структура

```
brain/           модель, энкодер (4096→4096 policy, value), пайплайн обучения
lichess_bot.py   Lichess-стрим, матчмейкинг, выбор хода, надёжность
engine/          Stockfish bridge (chain путей, SHA-256 hook, atexit)
scripts/         eval_model.py — воспроизводимая оценка против SF d8
tests/           юнит-тесты (encoder, train, частируч, бот-unsupported)
main.py          CLI: --fen анализ, --train обучение
data/            book.example.json + папка для PGN/книг (gitignored)
```

## Лицензия

License ещё не выбрана автором (см. `CLEANUP_PROGRESS.md`, needs-author).
