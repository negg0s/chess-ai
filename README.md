# Chess AI

Шахматный бот и нейросеть для анализа позиций и игры на Lichess.

В репозитории содержатся архитектура нейросети, веса обученной модели (Policy-Value ResNet), модуль интеграции со Stockfish и скрипт бота для игры на Lichess.

## Структура проекта

- `brain/model.py` - архитектура нейросети ChessBrainNet (ResNet с policy и value головами).
- `brain/encoder.py` - кодирование доски в тензор 18x8x8.
- `brain/train.py` - пайплайн обучения на PyTorch (CUDA / CPU).
- `models/chess_brain_ultra.pt` - обученные веса модели.
- `engine/stockfish_bridge.py` - подключение к движку Stockfish через протокол UCI (автоматически скачивает бинарник при первом запуске, если его нет).
- `vision/board_detector.py` - распознавание и выравнивание шахматной доски по фото или скриншоту.
- `vision/piece_classifier.py` - классификация фигур и преобразование позиции в FEN.
- `lichess_bot.py` - клиент для игры на Lichess API (прием вызовов, расчет ходов).
- `START_BOT.bat` - скрипт запуска бота в один клик под Windows.

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/NEGG0/chess-ai.git
cd chess-ai
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### 1. Запуск Lichess бота

Создайте файл `.env` на основе примера `.env.example`:
```env
LICHESS_TOKEN=ваш_токен_lichess
LICHESS_BOT_ID=ваш_логин_бота
```

Запуск:
```bash
python lichess_bot.py
```
Либо через `START_BOT.bat` на Windows.

### 2. Анализ позиции через CLI

```bash
python main.py --fen "r1bqk2r/pppp1ppp/2n5/4p3/1bB1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5"
```

### 3. Обучение модели

```bash
python main.py --train --epochs 5
```

## Стек

- Python 3.10+
- PyTorch (CUDA)
- python-chess
- OpenCV
- Stockfish UCI
