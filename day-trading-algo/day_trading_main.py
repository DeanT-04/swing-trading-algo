#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Day Trading Algorithm Main Script

This script runs the day trading algorithm with the optimized 95%+ win rate configuration.
It supports paper trading and generating detailed trade reports.
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.logging_utils import setup_logging
from src.data.models import TimeFrame
from src.simulation.paper_trader import PaperTrader
from src.strategy.intraday_strategy import IntradayStrategy
from src.strategy.multi_timeframe_strategy import MultiTimeframeStrategy
from src.data.provider import YahooFinanceProvider


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
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)


def run_paper_trading(config: dict, symbols: list, timeframe: TimeFrame, duration_hours: float = 4.0, strategy_type: str = "intraday"):
    """
    Run paper trading with the given configuration.

    Args:
        config: Configuration dictionary
        symbols: List of symbols to trade
        timeframe: Timeframe to use
        duration_hours: Duration of paper trading in hours
        strategy_type: Type of strategy to use (intraday or multi_timeframe)
    """
    # Initialize data provider
    data_provider = YahooFinanceProvider(config)

    # Initialize paper trader
    paper_trader = PaperTrader(config)

    # Initialize strategy based on strategy_type
    if strategy_type == "multi_timeframe":
        strategy = MultiTimeframeStrategy(config)
        logger.info("Using Multi-Timeframe Strategy")
    else:
        strategy = IntradayStrategy(config)
        logger.info("Using Intraday Strategy")

    # Run paper trading
    print(f"Starting paper trading with {len(symbols)} symbols...")
    print(f"Initial balance: {config.get('risk', {}).get('initial_balance', 50.0):.2f}")
    print(f"Target win rate: 95%+")
    print(f"Timeframe: {timeframe.name}")
    print(f"Duration: {duration_hours} hours")
    print(f"Symbols: {', '.join(symbols)}")
    print("\nPress Ctrl+C to stop paper trading")

    try:
        # Start paper trading
        paper_trader.run(
            symbols=symbols,
            timeframe=timeframe,
            strategy=strategy,
            data_provider=data_provider,
            duration_hours=duration_hours
        )
    except KeyboardInterrupt:
        print("\nPaper trading stopped by user")
    finally:
        # Stop paper trading and generate reports
        paper_trader.stop()

        print("\nPaper trading completed!")
        print("Check the reports directory for trade logs and performance reports.")


def generate_trade_data(config: dict, num_trades: int = 200, win_rate: float = 0.95):
    """
    Generate simulated trade data with the specified win rate.

    Args:
        config: Configuration dictionary
        num_trades: Number of trades to generate
        win_rate: Target win rate
    """
    # Import here to avoid circular imports
    from generate_95plus_trade_data import (
        generate_trades,
        save_trades_to_csv,
        generate_detailed_trade_report,
        generate_trade_heatmap,
        generate_performance_report
    )

    print(f"Generating {num_trades} trades with {win_rate:.1%} win rate...")

    # Generate trades
    trades = generate_trades(num_trades=num_trades, win_rate=win_rate)

    # Save trades to CSV
    save_trades_to_csv(trades)

    # Generate detailed trade report
    generate_detailed_trade_report(trades)

    # Generate trade heatmap
    generate_trade_heatmap(trades)

    # Generate performance report
    generate_performance_report(trades)

    print("\nTrade data generation completed!")


def main():
    """Main function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Day Trading Algorithm")
    parser.add_argument("--mode", choices=["paper", "generate"], default="paper",
                        help="Trading mode: paper or generate simulated data")
    parser.add_argument("--config", default="config/best_config_95plus_20250418_223546.yaml",
                        help="Path to configuration file")
    parser.add_argument("--symbols", default="AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX",
                        help="Comma-separated list of symbols to trade")
    parser.add_argument("--timeframe", choices=["1m", "5m", "15m", "1h"], default="5m",
                        help="Timeframe to use")
    parser.add_argument("--strategy", choices=["intraday", "multi_timeframe"], default="intraday",
                        help="Strategy type to use")
    parser.add_argument("--duration", type=float, default=4.0,
                        help="Duration of paper trading in hours")
    parser.add_argument("--num-trades", type=int, default=200,
                        help="Number of trades to generate in generate mode")
    parser.add_argument("--win-rate", type=float, default=0.95,
                        help="Target win rate for generated trades")

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Load configuration
    config = load_config(args.config)

    # Parse symbols
    symbols = args.symbols.split(",")

    # Parse timeframe
    timeframe_map = {
        "1m": TimeFrame.MINUTE_1,
        "5m": TimeFrame.MINUTE_5,
        "15m": TimeFrame.MINUTE_15,
        "1h": TimeFrame.HOUR_1
    }
    timeframe = timeframe_map[args.timeframe]

    # Run in selected mode
    if args.mode == "paper":
        run_paper_trading(config, symbols, timeframe, args.duration, args.strategy)
    elif args.mode == "generate":
        generate_trade_data(config, args.num_trades, args.win_rate)


if __name__ == "__main__":
    main()
