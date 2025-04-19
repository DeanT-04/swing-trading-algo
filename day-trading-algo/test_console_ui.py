#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the enhanced console UI.
"""

import os
import sys
import time
import argparse
import yaml
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.logging_utils import setup_logging
from src.simulation.paper_trader import PaperTrader
from src.data.models import TimeFrame
from src.strategy.intraday_strategy import IntradayStrategy
from src.data.provider import YahooFinanceProvider


def load_config(config_path="config/config.yaml"):
    """Load configuration from config file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the enhanced console UI")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--symbols", default="AAPL,MSFT,GOOGL,AMZN,TSLA", help="Comma-separated list of symbols to trade")
    parser.add_argument("--timeframe", default="5m", help="Timeframe to use (e.g., 1m, 5m, 15m)")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (ignore market hours)")
    args = parser.parse_args()

    # Set up logging
    setup_logging(log_level="INFO")

    # Load configuration
    config = load_config(args.config)

    # Parse symbols
    symbols = args.symbols.split(",")

    # Parse timeframe
    timeframe = TimeFrame.from_string(args.timeframe)

    # Initialize strategy
    strategy = IntradayStrategy(config)

    # Initialize data provider
    data_provider = YahooFinanceProvider(config)

    # Initialize paper trader with console UI enabled
    paper_config = config.get("paper_trading", {})
    paper_config["use_console_ui"] = True
    config["paper_trading"] = paper_config

    paper_trader = PaperTrader(config)

    # Run paper trader
    try:
        paper_trader.run(
            symbols=symbols,
            timeframe=timeframe,
            strategy=strategy,
            data_provider=data_provider,
            duration_hours=6.5,  # Standard market hours duration
            test_mode=args.test_mode  # Pass test mode flag
        )
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        paper_trader.stop()


if __name__ == "__main__":
    main()
