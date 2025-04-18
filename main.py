#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Swing Trading Algorithm.

This module initializes and runs the trading system, coordinating between
the various components for data retrieval, analysis, strategy execution,
and performance tracking.
"""

import os
import sys
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from src.data.models import TimeFrame
from src.data.provider import create_data_provider
from src.strategy.basic_swing import BasicSwingStrategy
from src.simulation.simulator import TradeSimulator
from src.performance.analyzer import PerformanceAnalyzer
from src.utils.logger import setup_logging


def load_config():
    """Load configuration from config file."""
    try:
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            print("Config file not found. Using default configuration.")
            config_path = Path("config/config_template.yaml")

        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def create_output_dirs(config: Dict):
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

    return {
        "logs_dir": logs_dir,
        "reports_dir": reports_dir
    }


def main():
    """Main function to run the trading algorithm."""
    # Load configuration
    config = load_config()

    # Create output directories
    output_dirs = create_output_dirs(config)

    # Set up logging
    logger = setup_logging(config["general"], output_dirs["logs_dir"])
    logger.info("Starting Swing Trading Algorithm")
    logger.info(f"Configuration loaded: {config['general']['name']}")

    # Initialize data provider
    try:
        data_provider = create_data_provider(config["data"])
        logger.info(f"Data provider initialized: {data_provider.__class__.__name__}")
    except Exception as e:
        logger.error(f"Error initializing data provider: {e}")
        return

    # Retrieve historical data for symbols
    symbols = config["data"]["symbols"]
    # Convert timeframes from strings to TimeFrame enum values
    _ = [TimeFrame.from_string(tf) for tf in config["data"]["timeframes"]]  # Not used directly but validates config
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config["data"]["history_days"])

    stocks = {}
    for symbol in symbols:
        try:
            logger.info(f"Retrieving data for {symbol}")
            stock = data_provider.get_historical_data(symbol, TimeFrame.DAILY, start_date, end_date)
            stocks[symbol] = stock
            logger.info(f"Retrieved {len(stock.data.get(TimeFrame.DAILY, []))} data points for {symbol}")
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")

    if not stocks:
        logger.error("No data retrieved for any symbols. Exiting.")
        return

    # Initialize strategy
    strategy = BasicSwingStrategy(config["strategy"])
    logger.info(f"Strategy initialized: {strategy.name}")

    # Initialize simulator
    simulator = TradeSimulator(config["risk"])
    logger.info(f"Simulator initialized with balance: {simulator.account.current_balance} {simulator.account.currency}")

    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer(config["performance"]["metrics"])
    logger.info("Performance analyzer initialized")

    # Run simulation
    logger.info("Starting simulation")

    # Analyze each stock and generate signals
    for symbol, stock in stocks.items():
        logger.info(f"Analyzing {symbol}")

        # Analyze the stock
        results = strategy.analyze(stock, TimeFrame.DAILY)

        if not results or "signals" not in results or not results["signals"]:
            logger.info(f"No signals generated for {symbol}")
            continue

        logger.info(f"Generated {len(results['signals'])} signals for {symbol}")

        # Process signals
        for signal in results["signals"]:
            # Add symbol to signal
            signal["symbol"] = symbol

            # Get the timestamp and corresponding price data
            timestamp = signal["timestamp"]
            df = stock.get_dataframe(TimeFrame.DAILY)
            if timestamp not in df.index:
                logger.warning(f"Timestamp {timestamp} not found in data for {symbol}")
                continue

            current_price = df.loc[timestamp, "close"]

            # Process the signal
            trade = simulator.process_signal(signal, stock, current_price, timestamp)

            if trade:
                logger.info(f"Opened {trade.direction.value} trade for {symbol} at {trade.entry_price}")

    # Update positions for all stocks
    closed_trades = simulator.update_positions(stocks, end_date)

    if closed_trades:
        logger.info(f"Closed {len(closed_trades)} trades")

    # Generate performance report
    trade_history = simulator.get_trade_history()
    # Generate report and save to reports directory
    analyzer.generate_report(trade_history, simulator.initial_balance, output_dirs["reports_dir"])

    # Print summary
    account_summary = simulator.get_account_summary()
    logger.info("Simulation completed")
    logger.info(f"Final balance: {account_summary['current_balance']} {account_summary['currency']}")
    logger.info(f"Total profit/loss: {account_summary['profit_loss']} {account_summary['currency']} ({account_summary['profit_loss_percent']}%)")
    logger.info(f"Total trades: {account_summary['total_trades']}")
    logger.info(f"Open positions: {account_summary['open_positions']}")

    logger.info("Swing Trading Algorithm completed")


if __name__ == "__main__":
    main()
