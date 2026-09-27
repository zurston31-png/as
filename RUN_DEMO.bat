@echo off
REM ===================================================================
REM  Walks a made-up token through a complete trade so you can watch
REM  the whole cycle: position opens with a stop-loss and take-profit,
REM  the price moves, the bot closes it by itself, P&L is recorded.
REM
REM  Uses the real trading engine and risk limits. Only the security
REM  scanner is stubbed, and only for the one fake demo token - your
REM  real rug-check settings are untouched.
REM
REM  No wallet. No crypto. No real money. Nothing is bought or sold.
REM ===================================================================

setlocal
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe scripts\run_demo.py
    goto :end
)

echo.
echo   The bot isn't set up yet in this folder.
echo   Double-click START_HERE.bat first, then run this again.
echo.
pause

:end
endlocal
