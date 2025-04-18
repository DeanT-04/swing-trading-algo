#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated Day Trading Script

This script automatically starts the day trading algorithm at market open,
monitors multiple stocks, executes trades, and logs all trade details.
It runs autonomously and can be scheduled to start automatically.
"""

import os
import sys
import time
import yaml
import logging
import argparse
import schedule
import pandas as pd
from datetime import datetime, timedelta
import pytz
import signal
import subprocess

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.logging_utils import setup_logging
from src.data.models import TimeFrame
from src.simulation.paper_trader import PaperTrader
from src.strategy.intraday_strategy import IntradayStrategy
from src.data.provider import YahooFinanceProvider


# Global variables
running = True
paper_trader = None
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


def load_stock_list(file_path: str = "config/stock_list.txt") -> list:
    """
    Load list of stocks to trade from a file.
    
    Args:
        file_path: Path to the stock list file
        
    Returns:
        list: List of stock symbols
    """
    try:
        if not os.path.exists(file_path):
            # Create default stock list if file doesn't exist
            default_stocks = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
                "TSLA", "NVDA", "AMD", "INTC", "NFLX",
                "JPM", "BAC", "WMT", "DIS", "CSCO",
                "PFE", "KO", "PEP", "T", "VZ"
            ]
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write('\n'.join(default_stocks))
            logger.info(f"Created default stock list at {file_path}")
            return default_stocks
        
        with open(file_path, 'r') as file:
            stocks = [line.strip() for line in file if line.strip()]
        logger.info(f"Loaded {len(stocks)} stocks from {file_path}")
        return stocks
    except Exception as e:
        logger.error(f"Error loading stock list: {e}")
        # Return default list if there's an error
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "NFLX"]


def is_market_open() -> bool:
    """
    Check if the US stock market is currently open.
    
    Returns:
        bool: True if market is open, False otherwise
    """
    # Get current time in Eastern Time (US market time)
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check if it's between 9:30 AM and 4:00 PM Eastern
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close


def get_next_market_open() -> datetime:
    """
    Get the next market open time.
    
    Returns:
        datetime: Next market open time
    """
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Set to 9:30 AM today
    next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # If it's already past 9:30 AM, set to tomorrow
    if now > next_open:
        next_open = next_open + timedelta(days=1)
    
    # If it's weekend, move to Monday
    while next_open.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_open = next_open + timedelta(days=1)
    
    return next_open


def start_trading(config: dict, symbols: list, timeframe: TimeFrame):
    """
    Start the day trading algorithm.
    
    Args:
        config: Configuration dictionary
        symbols: List of symbols to trade
        timeframe: Timeframe to use
    """
    global paper_trader
    
    logger.info(f"Starting day trading algorithm with {len(symbols)} symbols")
    
    # Initialize data provider
    data_provider = YahooFinanceProvider()
    
    # Initialize strategy
    strategy = IntradayStrategy(config)
    
    # Initialize paper trader
    paper_trader = PaperTrader(config)
    
    # Start paper trading
    try:
        paper_trader.run(
            symbols=symbols,
            timeframe=timeframe,
            strategy=strategy,
            data_provider=data_provider,
            duration_hours=6.5  # Standard market hours duration
        )
    except Exception as e:
        logger.error(f"Error in paper trading: {e}")
    finally:
        if paper_trader:
            paper_trader.stop()
            logger.info("Paper trading stopped")


def stop_trading():
    """Stop the day trading algorithm."""
    global paper_trader
    
    if paper_trader:
        logger.info("Stopping day trading algorithm")
        paper_trader.stop()


def signal_handler(sig, frame):
    """Handle interrupt signals."""
    global running
    
    logger.info("Received interrupt signal, shutting down...")
    running = False
    stop_trading()
    sys.exit(0)


def schedule_trading(config: dict, symbols: list, timeframe: TimeFrame):
    """
    Schedule trading to start at market open and stop at market close.
    
    Args:
        config: Configuration dictionary
        symbols: List of symbols to trade
        timeframe: Timeframe to use
    """
    # Schedule trading to start at market open (9:30 AM Eastern)
    schedule.every().day.at("09:30").do(
        lambda: start_trading(config, symbols, timeframe)
    ).tag("trading")
    
    # Schedule trading to stop at market close (4:00 PM Eastern)
    schedule.every().day.at("16:00").do(stop_trading).tag("trading")
    
    logger.info("Trading scheduled to start at 9:30 AM and stop at 4:00 PM Eastern time")


def generate_daily_report():
    """Generate a daily trading report."""
    try:
        # Check if trade log exists
        if not os.path.exists("reports/trade_log.csv"):
            logger.info("No trade log found, skipping daily report")
            return
        
        # Read trade log
        trades_df = pd.read_csv("reports/trade_log.csv")
        
        # Filter trades for today
        today = datetime.now().strftime("%Y-%m-%d")
        today_trades = trades_df[trades_df["entry_time"].str.contains(today)]
        
        if len(today_trades) == 0:
            logger.info(f"No trades found for {today}, skipping daily report")
            return
        
        # Calculate statistics
        total_trades = len(today_trades)
        winning_trades = len(today_trades[today_trades["profit_loss"] > 0])
        losing_trades = len(today_trades[today_trades["profit_loss"] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        total_profit = today_trades["profit_loss"].sum()
        
        # Generate report
        report = f"""
=== Daily Trading Report for {today} ===
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Losing Trades: {losing_trades}
Win Rate: {win_rate:.2f}%
Total Profit/Loss: {total_profit:.2f}
=======================================
"""
        
        # Save report to file
        report_file = f"reports/daily_report_{today}.txt"
        with open(report_file, "w") as f:
            f.write(report)
        
        logger.info(f"Daily report saved to {report_file}")
        
        # Also print to console
        print(report)
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")


def main():
    """Main function."""
    global running
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Automated Day Trading Script")
    parser.add_argument("--config", default="config/best_config_95plus_20250418_223546.yaml",
                        help="Path to configuration file")
    parser.add_argument("--stock-list", default="config/stock_list.txt",
                        help="Path to stock list file")
    parser.add_argument("--timeframe", choices=["1m", "5m", "15m", "1h"], default="5m",
                        help="Timeframe to use")
    parser.add_argument("--start-now", action="store_true",
                        help="Start trading immediately instead of waiting for market open")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in test mode with simulated market hours")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load stock list
    symbols = load_stock_list(args.stock_list)
    
    # Parse timeframe
    timeframe_map = {
        "1m": TimeFrame.MINUTE_1,
        "5m": TimeFrame.MINUTE_5,
        "15m": TimeFrame.MINUTE_15,
        "1h": TimeFrame.HOUR_1
    }
    timeframe = timeframe_map[args.timeframe]
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Schedule daily report generation at 4:15 PM Eastern
    schedule.every().day.at("16:15").do(generate_daily_report).tag("reporting")
    
    if args.test_mode:
        logger.info("Running in test mode with simulated market hours")
        # Start trading immediately in test mode
        start_trading(config, symbols, timeframe)
    else:
        # Schedule trading
        schedule_trading(config, symbols, timeframe)
        
        # Start trading immediately if requested
        if args.start_now:
            logger.info("Starting trading immediately")
            start_trading(config, symbols, timeframe)
        else:
            # Check if market is open
            if is_market_open():
                logger.info("Market is open, starting trading")
                start_trading(config, symbols, timeframe)
            else:
                next_open = get_next_market_open()
                logger.info(f"Market is closed, next open at {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                logger.info("Waiting for market open...")
    
    # Main loop
    while running:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
