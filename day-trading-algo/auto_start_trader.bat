@echo off
echo Starting Day Trading Algorithm at %TIME% on %DATE%
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "..\venv\Scripts\activate.bat" (
    call "..\venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found, using system Python
)

REM Start the day trading algorithm
python auto_trader.py --start-now

REM If the algorithm exits, log it
echo Day Trading Algorithm exited at %TIME% on %DATE% >> logs\startup_log.txt

REM Keep the window open if there was an error
if %ERRORLEVEL% NEQ 0 (
    echo Error occurred. Press any key to close this window.
    pause > nul
)
