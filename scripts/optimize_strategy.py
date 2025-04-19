#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to optimize strategy parameters.

This script runs the optimization process to find the best parameters
for the trading strategy.
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from src.data.models import TimeFrame
from src.data.provider import create_data_provider
from src.optimization.optimizer import StrategyOptimizer
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


def main():
    """Main function to run the optimization."""
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--method", choices=["grid_search", "random_search", "genetic"],
                        help="Optimization method")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials for random search")
    parser.add_argument("--population", type=int, default=20,
                        help="Population size for genetic algorithm")
    parser.add_argument("--generations", type=int, default=5,
                        help="Number of generations for genetic algorithm")
    parser.add_argument("--metric", default="sharpe_ratio",
                        choices=["sharpe_ratio", "profit_factor", "win_rate", "total_return", "total_return_percent", "max_drawdown"],
                        help="Metric to optimize for")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    output_dirs = create_output_dirs(config)
    
    # Set up logging
    logger = setup_logging(config["general"], output_dirs["logs_dir"])
    logger.info("Starting Strategy Optimization")
    
    # Override optimization method if specified
    if args.method:
        config["optimization"]["method"] = args.method
    
    # Override number of trials if specified
    if args.trials:
        config["optimization"]["random_trials"] = args.trials
    
    # Override population size if specified
    if args.population:
        config["optimization"]["population_size"] = args.population
    
    # Override number of generations if specified
    if args.generations:
        config["optimization"]["generations"] = args.generations
    
    # Override metric if specified
    if args.metric:
        config["optimization"]["metric"] = args.metric
    
    # Initialize data provider
    try:
        data_provider = create_data_provider(config["data"])
        logger.info(f"Data provider initialized: {data_provider.__class__.__name__}")
    except Exception as e:
        logger.error(f"Error initializing data provider: {e}")
        return
    
    # Retrieve historical data for symbols
    symbols = config["data"]["symbols"]
    timeframe = TimeFrame.from_string(config["data"]["timeframes"][0])  # Use first timeframe
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config["data"]["history_days"])
    
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
    
    # Define parameter ranges to optimize
    parameter_ranges = {
        "ma_type": ["SMA", "EMA"],
        "fast_ma_period": [3, 5, 7, 9, 11],
        "slow_ma_period": [15, 20, 25, 30],
        "trend_ma_period": [30, 40, 50, 60],
        "rsi_period": [7, 9, 11, 14],
        "rsi_overbought": [60, 65, 70, 75],
        "rsi_oversold": [25, 30, 35, 40],
        "atr_period": [7, 10, 14, 21],
        "atr_multiplier": [1.0, 1.5, 2.0, 2.5],
        "volume_ma_period": [5, 10, 15, 20],
        "risk_reward_ratio": [1.0, 1.5, 2.0, 2.5, 3.0]
    }
    
    # Initialize optimizer
    optimizer = StrategyOptimizer(config["optimization"])
    
    # Run optimization
    logger.info(f"Running optimization with method: {config['optimization']['method']}")
    result = optimizer.optimize(stocks, timeframe, config["general"]["initial_capital"], parameter_ranges)
    
    # Print results
    logger.info(f"Optimization completed")
    logger.info(f"Best parameters: {result['parameters']}")
    logger.info(f"Score: {result['score']}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dirs["reports_dir"] / f"optimization_result_{timestamp}.yaml"
    
    with open(result_file, 'w') as file:
        yaml.dump(result, file)
    
    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
