@echo off
title Chess Bot
cd /d "%~dp0"

echo =========================================
echo             CHESS BOT LAUNCHER
echo =========================================
echo.

if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe lichess_bot.py
) else (
    python lichess_bot.py
)

pause
