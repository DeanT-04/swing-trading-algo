#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the day trading algorithm.

This script initializes and runs the day trading algorithm, either in
backtest mode or paper trading mode.
"""

import os
import sys
import argparse
import yaml
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/day_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def run_backtest(config, start_date, end_date, symbols=None):
    """Run backtesting with the specified configuration."""
    logger.info(f"Starting backtest from {start_date} to {end_date}")

    # Import here to avoid circular imports
    from src.data.provider import YahooFinanceProvider
    from src.data.models import TimeFrame
    from src.strategy.intraday_strategy import IntradayStrategy

    # Initialize data provider
    data_provider = YahooFinanceProvider(config)

    # Initialize strategy
    strategy = IntradayStrategy(config)

    # Get symbols from config if not provided
    if not symbols:
        symbols = config.get("data", {}).get("symbols", [])
        if not symbols:
            logger.error("No symbols configured for trading")
            return False

    # Get timeframes from config
    timeframes_str = config.get("data", {}).get("timeframes", ["5m"])
    timeframes = [TimeFrame.from_string(tf) for tf in timeframes_str]

    # Fetch data for all symbols
    logger.info(f"Fetching data for {len(symbols)} symbols")
    stocks = data_provider.get_multiple_stocks_data(symbols, timeframes, start_date, end_date)

    if not stocks:
        logger.error("No data fetched for any symbol")
        return False

    logger.info(f"Data fetched for {len(stocks)} symbols")

    # Run backtest for each symbol
    results = []

    for symbol, stock in stocks.items():
        logger.info(f"Analyzing {symbol}...")

        # Analyze the stock
        analysis = strategy.analyze(stock, timeframes[0])

        if not analysis:
            logger.warning(f"No analysis results for {symbol}")
            continue

        # Get signals
        signals = analysis.get("signals", [])

        if not signals:
            logger.info(f"No signals found for {symbol}")
            continue

        logger.info(f"Found {len(signals)} signals for {symbol}")

        # Process signals
        for signal in signals:
            signal["symbol"] = symbol
            results.append(signal)

    # Save results
    if results:
        os.makedirs("reports", exist_ok=True)
        report_file = f"reports/backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w') as file:
            json.dump(results, file, indent=4)

        logger.info(f"Backtest results saved to {report_file}")
        logger.info(f"Found {len(results)} signals across {len(stocks)} symbols")
    else:
        logger.warning("No signals found in backtest")

    return True

def run_paper_trading(config):
    """Run paper trading with the specified configuration."""
    logger.info("Starting paper trading")

    # Import here to avoid circular imports
    from src.simulation.paper_trader import PaperTrader

    # Initialize paper trader
    paper_trader = PaperTrader(config)

    # Start paper trading
    success = paper_trader.start()

    return success

def main():
    """Main function to parse arguments and run the algorithm."""
    parser = argparse.ArgumentParser(description="Run day trading algorithm")
    parser.add_argument("--mode", choices=["backtest", "paper"], default="paper",
                        help="Mode to run (backtest or paper trading)")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--start-date", help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--symbols", help="Comma-separated list of symbols to trade")

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = args.symbols.split(',')

    # Run in specified mode
    if args.mode == "backtest":
        if not args.start_date or not args.end_date:
            logger.error("Start date and end date are required for backtest mode")
            sys.exit(1)

        success = run_backtest(config, args.start_date, args.end_date, symbols)
        if success:
            logger.info("Backtesting completed successfully")
        else:
            logger.error("Backtesting failed")

    elif args.mode == "paper":
        success = run_paper_trading(config)
        if success:
            logger.info("Paper trading completed successfully")
        else:
            logger.error("Paper trading failed")

    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
