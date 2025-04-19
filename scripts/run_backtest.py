#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run backtests on historical data.

This script runs backtests on historical data to evaluate
the performance of trading strategies.
"""

import os
import sys
import yaml
import logging
import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.append('.')

from src.data.models import TimeFrame
from src.data.provider import create_data_provider
from src.strategy.basic_swing import BasicSwingStrategy
from src.simulation.backtester import Backtester
from src.utils.logger import setup_logging


def load_config(config_path="config/config.yaml"):
    """Load configuration from config file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def create_output_dirs(config):
    """Create output directories for logs, data, and reports."""
    # Create logs directory
    logs_dir = Path("logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create data cache directory
    if "cache_dir" in config["data"]:
        cache_dir = Path(config["data"]["cache_dir"])
        os.makedirs(cache_dir, exist_ok=True)

    # Create reports directory
    reports_dir = Path("reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Create backtest directories
    backtest_dir = Path("backtest")
    backtest_results_dir = Path("backtest/results")
    os.makedirs(backtest_dir, exist_ok=True)
    os.makedirs(backtest_results_dir, exist_ok=True)

    return {
        "logs_dir": logs_dir,
        "reports_dir": reports_dir,
        "backtest_dir": backtest_dir,
        "backtest_results_dir": backtest_results_dir
    }


def plot_equity_curve(equity_curve, initial_balance, output_path):
    """Plot the equity curve and save to file."""
    plt.figure(figsize=(12, 6))

    # Convert to DataFrame if it's a list
    if isinstance(equity_curve, list):
        equity_curve = pd.DataFrame(equity_curve)

    # Plot equity curve
    plt.plot(equity_curve["date"], equity_curve["equity"], label="Equity")

    # Plot initial balance
    plt.axhline(y=initial_balance, color='r', linestyle='--', label="Initial Balance")

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Backtest Equity Curve")
    plt.legend()
    plt.grid(True)

    # Format x-axis dates
    plt.gcf().autofmt_xdate()

    # Save to file
    plt.savefig(output_path)
    plt.close()


def main():
    """Main function to run the backtest."""
    parser = argparse.ArgumentParser(description="Run backtests on historical data")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--start-date", help="Start date for the backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for the backtest (YYYY-MM-DD)")
    parser.add_argument("--symbols", help="Comma-separated list of symbols to backtest")
    parser.add_argument("--timeframe", choices=["daily", "4h", "1h"],
                        help="Timeframe for the backtest")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create output directories
    output_dirs = create_output_dirs(config)

    # Set up logging
    logger = setup_logging(config["general"], output_dirs["logs_dir"])
    logger.info("Starting Backtest")

    # Initialize backtest configuration
    if "backtest" not in config:
        config["backtest"] = {
            "start_date": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "timeframe": config["data"]["timeframes"][0]
        }

    # Override configuration with command-line arguments
    if args.start_date:
        config["backtest"]["start_date"] = args.start_date

    if args.end_date:
        config["backtest"]["end_date"] = args.end_date

    if args.timeframe:
        config["backtest"]["timeframe"] = args.timeframe

    if args.symbols:
        config["data"]["symbols"] = args.symbols.split(",")

    # Parse dates
    start_date = pd.to_datetime(config["backtest"]["start_date"]).to_pydatetime().replace(tzinfo=None)
    end_date = pd.to_datetime(config["backtest"]["end_date"]).to_pydatetime().replace(tzinfo=None)

    # Initialize data provider
    try:
        data_provider = create_data_provider(config["data"])
        logger.info(f"Data provider initialized: {data_provider.__class__.__name__}")
    except Exception as e:
        logger.error(f"Error initializing data provider: {e}")
        return

    # Retrieve historical data for symbols
    symbols = config["data"]["symbols"]
    timeframe = TimeFrame.from_string(config["backtest"]["timeframe"])

    stocks = {}
    for symbol in symbols:
        try:
            logger.info(f"Retrieving data for {symbol}")
            stock = data_provider.get_historical_data(symbol, timeframe, start_date, end_date)
            stocks[symbol] = stock
            logger.info(f"Retrieved {len(stock.data.get(timeframe, []))} data points for {symbol}")
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")

    if not stocks:
        logger.error("No data retrieved for any symbols. Exiting.")
        return

    # Initialize strategy
    strategy = BasicSwingStrategy(config["strategy"])
    logger.info(f"Strategy initialized: {strategy.name}")

    # Initialize backtester
    backtester = Backtester(config["risk"])

    # Run backtest
    logger.info(f"Running backtest from {start_date} to {end_date}")
    result = backtester.run_backtest(stocks, strategy, timeframe, start_date, end_date)

    if not result["success"]:
        logger.error(f"Backtest failed: {result.get('error', 'Unknown error')}")
        return

    # Print backtest results
    logger.info(f"Backtest completed")
    logger.info(f"Initial balance: {result['initial_balance']} {config['risk']['currency']}")
    logger.info(f"Final balance: {result['final_balance']} {config['risk']['currency']}")
    logger.info(f"Total return: {result['total_return']} {config['risk']['currency']} ({result['total_return_percent']:.2f}%)")
    logger.info(f"Total trades: {result['total_trades']}")
    logger.info(f"Winning trades: {result['winning_trades']} ({result['win_rate']:.2f}%)")
    logger.info(f"Losing trades: {result['losing_trades']}")
    logger.info(f"Profit factor: {result['profit_factor']:.2f}")
    logger.info(f"Max drawdown: {result['max_drawdown']} {config['risk']['currency']} ({result['max_drawdown_percent']:.2f}%)")
    logger.info(f"Sharpe ratio: {result['sharpe_ratio']:.2f}")
    logger.info(f"Sortino ratio: {result['sortino_ratio']:.2f}")

    # Save backtest results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dirs["backtest_results_dir"] / f"backtest_result_{timestamp}.json"

    # Convert equity curve to list of dictionaries for JSON serialization
    equity_curve_list = result["equity_curve"].to_dict(orient="records")
    result["equity_curve"] = equity_curve_list

    with open(result_file, 'w') as file:
        json.dump(result, file, indent=4, default=str)

    logger.info(f"Backtest results saved to {result_file}")

    # Plot equity curve
    equity_curve_file = output_dirs["backtest_results_dir"] / f"equity_curve_{timestamp}.png"
    plot_equity_curve(result["equity_curve"], result["initial_balance"], equity_curve_file)

    logger.info(f"Equity curve saved to {equity_curve_file}")


if __name__ == "__main__":
    main()
