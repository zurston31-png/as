@echo off
REM ===================================================================
REM  Memecoin Trading Bot - Windows launcher
REM
REM  Just double-click this file. It sets everything up the first time,
REM  then starts the bot and opens the dashboard in your browser.
REM
REM  This runs in PAPER TRADING mode: no real money, no wallet, no
REM  crypto. Every trade is simulated.
REM ===================================================================

setlocal
cd /d "%~dp0"

echo.
echo   Starting the memecoin trading bot setup...
echo   (This window will show progress. Leave it open.)
echo.

REM --- Try the "py" launcher first (standard python.org installs) ---
py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    py -3 scripts\setup_and_run.py
    goto :end
)

REM --- Fall back to "python" on PATH ---
python --version >nul 2>&1
if %errorlevel% equ 0 (
    python scripts\setup_and_run.py
    goto :end
)

REM --- No usable Python found ---
echo.
echo  ====================================================================
echo   PYTHON IS NOT INSTALLED
echo  ====================================================================
echo.
echo   This bot needs Python to run. It's free and takes 2 minutes:
echo.
echo     1. Go to:  https://www.python.org/downloads/
echo     2. Click the big yellow "Download Python" button
echo     3. Run the installer
echo.
echo     4. IMPORTANT: On the very first screen of the installer,
echo        tick the box that says "Add python.exe to PATH"
echo        (it's at the bottom - easy to miss, and nothing works
echo         without it)
echo.
echo     5. Click "Install Now", wait for it to finish
echo     6. Close this window and double-click START_HERE.bat again
echo.
echo  ====================================================================
echo.
pause

:end
endlocal
