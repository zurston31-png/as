@echo off
REM ===================================================================
REM  Sends a fake TradingView alert to your running bot, so you can
REM  watch how it reacts without setting up TradingView.
REM
REM  The bot must already be running (double-click START_HERE.bat first).
REM ===================================================================

setlocal
cd /d "%~dp0"

py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    py -3 scripts\send_test_signal.py
    goto :end
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    python scripts\send_test_signal.py
    goto :end
)

echo.
echo   Python isn't installed yet. Double-click START_HERE.bat first -
echo   it will walk you through installing it.
echo.
pause

:end
endlocal
