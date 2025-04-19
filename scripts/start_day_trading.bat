@echo off
echo Starting Day Trading Algorithm with Enhanced Console UI at %TIME% on %DATE%

REM Set the path to the day trading algorithm directory
set TRADING_ALGO_PATH=C:\Users\deana\Documents\Coding projects\trading-algo\swing-trading-algo\day-trading-algo

REM Change to the trading algorithm directory
cd /d "%TRADING_ALGO_PATH%"

REM Activate virtual environment if it exists
if exist "..\venv\Scripts\activate.bat" (
    call "..\venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found, using system Python
)

REM Check if rich library is installed
python -c "try: import rich; print('installed'); except ImportError: print('not installed')" > temp.txt
set /p RICH_INSTALLED=<temp.txt
del temp.txt

if "%RICH_INSTALLED%" == "not installed" (
    echo Installing rich library for enhanced console output...
    pip install rich
)

REM Start the day trading algorithm with enhanced console UI
echo Starting day trading algorithm with enhanced console UI...
python auto_trader.py --no-ui --start-now

REM If the algorithm exits, log it
echo Day Trading Algorithm exited at %TIME% on %DATE% >> logs\startup_log.txt

REM Keep the window open if there was an error
if %ERRORLEVEL% NEQ 0 (
    echo Error occurred. Press any key to close this window.
    pause > nul
)
