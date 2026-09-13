@echo off
REM ===================================================================
REM  Looks up one token and shows exactly what the security scanners
REM  report about it. Useful when the bot rejects a token and you want
REM  to know whether the data is genuinely missing or the filter is
REM  looking in the wrong place.
REM
REM  Nothing is traded. This only reads public data.
REM ===================================================================

setlocal
cd /d "%~dp0"

py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    py -3 scripts\diagnose_token.py
    goto :end
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    python scripts\diagnose_token.py
    goto :end
)

echo.
echo   Python isn't installed yet. Double-click START_HERE.bat first.
echo.
pause

:end
endlocal
