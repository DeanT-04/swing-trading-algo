#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the paper trading system.

This script initializes and runs the paper trading system,
which simulates trades in real-time without actually executing them.
"""

import os
import sys
import yaml
import logging
import argparse
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.data.models import TimeFrame
from src.data.realtime import RealTimeDataManager
from src.strategy.basic_swing import BasicSwingStrategy
from src.simulation.paper_trader import PaperTrader
from src.performance.analyzer import PerformanceAnalyzer
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
    
    return {
        "logs_dir": logs_dir,
        "reports_dir": reports_dir
    }


def signal_handler(signum, frame):
    """Handle termination signals."""
    print("\nReceived termination signal. Shutting down...")
    global running
    running = False


def process_data_update(paper_trader, strategy, timeframe, stocks):
    """
    Process data update and execute trades if necessary.
    
    Args:
        paper_trader: Paper trader object
        strategy: Trading strategy object
        timeframe: Timeframe for analysis
        stocks: Dictionary of Stock objects by symbol
    """
    current_time = datetime.now()
    
    # Analyze each stock and generate signals
    for symbol, stock in stocks.items():
        logger.info(f"Analyzing {symbol}")
        
        # Analyze the stock
        results = strategy.analyze(stock, timeframe)
        
        if not results or "signals" not in results or not results["signals"]:
            logger.info(f"No signals generated for {symbol}")
            continue
        
        logger.info(f"Generated {len(results['signals'])} signals for {symbol}")
        
        # Process the most recent signal
        signal = results["signals"][-1]
        
        # Add symbol to signal
        signal["symbol"] = symbol
        
        # Get the timestamp and corresponding price data
        timestamp = signal["timestamp"]
        df = stock.get_dataframe(timeframe)
        if timestamp not in df.index:
            logger.warning(f"Timestamp {timestamp} not found in data for {symbol}")
            continue
        
        # Check if the signal is recent
        signal_time = timestamp
        if hasattr(signal_time, 'to_pydatetime'):
            signal_time = signal_time.to_pydatetime()
        
        # Only process signals from the last update interval
        if (current_time - signal_time).total_seconds() > 3600:  # 1 hour
            logger.info(f"Signal for {symbol} is too old: {signal_time}")
            continue
        
        current_price = df.loc[timestamp, "close"]
        
        # Process the signal
        trade = paper_trader.process_signal(signal, stock, current_price, current_time)
        
        if trade:
            logger.info(f"Opened {trade.direction.value} trade for {symbol} at {trade.entry_price}")
    
    # Update positions for all stocks
    closed_trades = paper_trader.update_positions(stocks, current_time)
    
    if closed_trades:
        logger.info(f"Closed {len(closed_trades)} trades")
        
        # Print trade details
        for trade in closed_trades:
            logger.info(f"Closed {trade.direction.value} trade for {trade.symbol} at {trade.exit_price} ({trade.exit_reason})")
            logger.info(f"Profit/Loss: {trade.profit_loss:.2f} ({trade.profit_loss_percent:.2f}%)")
    
    # Print account summary
    account_summary = paper_trader.get_account_summary()
    logger.info(f"Account balance: {account_summary['current_balance']:.2f} {account_summary['currency']}")
    logger.info(f"Profit/Loss: {account_summary['profit_loss']:.2f} {account_summary['currency']} ({account_summary['profit_loss_percent']:.2f}%)")
    logger.info(f"Open positions: {account_summary['open_positions']}")
    logger.info(f"Total trades: {account_summary['total_trades']}")


def main():
    """Main function to run the paper trading system."""
    parser = argparse.ArgumentParser(description="Run the paper trading system")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--update-interval", type=int, default=60,
                        help="Update interval in seconds")
    parser.add_argument("--state-file", default="paper_trader_state.json",
                        help="State file for paper trader")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    output_dirs = create_output_dirs(config)
    
    # Set up logging
    global logger
    logger = setup_logging(config["general"], output_dirs["logs_dir"])
    logger.info("Starting Paper Trading System")
    
    # Initialize paper trading configuration
    if "paper_trading" not in config:
        config["paper_trading"] = {
            "update_interval": args.update_interval,
            "state_file": args.state_file,
            "timeframe": config["data"]["timeframes"][0]
        }
    else:
        # Override configuration with command-line arguments
        config["paper_trading"]["update_interval"] = args.update_interval
        config["paper_trading"]["state_file"] = args.state_file
    
    # Initialize strategy
    strategy = BasicSwingStrategy(config["strategy"])
    logger.info(f"Strategy initialized: {strategy.name}")
    
    # Initialize paper trader
    paper_trader_config = config["risk"].copy()
    paper_trader_config["state_file"] = config["paper_trading"]["state_file"]
    paper_trader = PaperTrader(paper_trader_config)
    logger.info(f"Paper trader initialized with balance: {paper_trader.account.current_balance} {paper_trader.account.currency}")
    
    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer(config["performance"]["metrics"])
    logger.info("Performance analyzer initialized")
    
    # Initialize real-time data manager
    data_manager = RealTimeDataManager(config["data"])
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start real-time data manager
    data_manager.start()
    
    # Main loop
    global running
    running = True
    
    timeframe = TimeFrame.from_string(config["paper_trading"]["timeframe"])
    update_interval = config["paper_trading"]["update_interval"]
    last_update = datetime.now() - timedelta(seconds=update_interval)
    
    try:
        while running:
            # Check if it's time to update
            current_time = datetime.now()
            if (current_time - last_update).total_seconds() >= update_interval:
                # Get current stocks data
                stocks = data_manager.get_stocks()
                
                # Process data update
                process_data_update(paper_trader, strategy, timeframe, stocks)
                
                # Update last update time
                last_update = current_time
                
                # Generate performance report every hour
                if current_time.minute == 0 and current_time.second < 10:
                    # Get trade history
                    trade_history = paper_trader.get_trade_history()
                    
                    # Generate report
                    analyzer.generate_report(trade_history, paper_trader.initial_balance, output_dirs["reports_dir"])
                    
                    # Sleep to avoid generating multiple reports
                    time.sleep(50)
            
            # Sleep to avoid high CPU usage
            time.sleep(1)
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    
    finally:
        # Stop real-time data manager
        data_manager.stop()
        
        # Generate final performance report
        trade_history = paper_trader.get_trade_history()
        analyzer.generate_report(trade_history, paper_trader.initial_balance, output_dirs["reports_dir"])
        
        # Print final summary
        account_summary = paper_trader.get_account_summary()
        logger.info("Paper Trading System stopped")
        logger.info(f"Final balance: {account_summary['current_balance']} {account_summary['currency']}")
        logger.info(f"Total profit/loss: {account_summary['profit_loss']} {account_summary['currency']} ({account_summary['profit_loss_percent']}%)")
        logger.info(f"Total trades: {account_summary['total_trades']}")
        logger.info(f"Open positions: {account_summary['open_positions']}")


if __name__ == "__main__":
    main()
