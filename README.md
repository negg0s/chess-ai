# chess-ai

Шахматная нейросеть (ResNet Policy-Value), мост к Stockfish и бот для Lichess.

## Установка

```bash
pip install -r requirements.txt
```

## Использование

1. Указать токен в `.env`:
```env
LICHESS_TOKEN=ваш_токен
LICHESS_BOT_ID=логин_бота
```

2. Запуск бота:
```bash
python lichess_bot.py
```
Или запустить `START_BOT.bat` на Windows.

3. Анализ позиции:
```bash
python main.py --fen "r1bqk2r/pppp1ppp/2n5/4p3/1bB1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5"
```

4. Обучение модели:
```bash
python main.py --train --epochs 5
```
