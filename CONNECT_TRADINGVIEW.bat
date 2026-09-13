@echo off
REM ===================================================================
REM  Gives your bot a temporary public web address so TradingView can
REM  send it alerts automatically, then tests that it works and prints
REM  the exact values to paste into TradingView.
REM
REM  Start the bot FIRST (START_HERE.bat), and leave both windows open.
REM
REM  Still paper trading. No real money is involved.
REM ===================================================================

setlocal
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe scripts\connect_tradingview.py
    goto :end
)

py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    py -3 scripts\connect_tradingview.py
    goto :end
)

echo.
echo   The bot isn't set up in this folder yet.
echo   Double-click START_HERE.bat first, then run this again.
echo.
pause

:end
endlocal
