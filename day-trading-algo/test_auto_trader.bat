@echo off
echo Starting Automated Day Trading System in Test Mode...
cd %~dp0
call venv\Scripts\activate.bat
python auto_trader.py --config config/best_config_95plus_20250418_223546.yaml --stock-list config/stock_list.txt --timeframe 5m --test-mode
pause
