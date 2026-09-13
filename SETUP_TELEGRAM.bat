@echo off
REM ===================================================================
REM  Sets up Telegram notifications, so the bot messages your phone
REM  whenever it trades, rejects a scam token, or posts its daily
REM  profit/loss summary.
REM
REM  Walks you through creating the bot, finds your chat ID for you,
REM  saves both to .env, and sends a test message.
REM ===================================================================

setlocal
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe scripts\setup_telegram.py
    goto :end
)

py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    py -3 scripts\setup_telegram.py
    goto :end
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    python scripts\setup_telegram.py
    goto :end
)

echo.
echo   The bot isn't set up in this folder yet.
echo   Double-click START_HERE.bat first, then run this again.
echo.
pause

:end
endlocal
