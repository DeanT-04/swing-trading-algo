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
import logging
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.logging_utils import setup_logging
from src.data.models import TimeFrame


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


def run_backtest(config: dict, start_date: str, end_date: str, timeframe: str) -> None:
    """
    Run a backtest of the day trading algorithm.
    
    Args:
        config: Configuration dictionary
        start_date: Start date for the backtest (YYYY-MM-DD)
        end_date: End date for the backtest (YYYY-MM-DD)
        timeframe: Timeframe to use for the backtest
    """
    from src.simulation.backtest import run_backtest
    
    # Convert timeframe string to TimeFrame enum
    tf = TimeFrame.from_string(timeframe)
    
    # Run the backtest
    run_backtest(config, start_date, end_date, tf)


def run_paper_trading(config: dict) -> None:
    """
    Run the day trading algorithm in paper trading mode.
    
    Args:
        config: Configuration dictionary
    """
    # Import here to avoid circular imports
    import run_paper_trading as paper_trading
    
    # Run paper trading
    paper_trading.main()


def run_optimization(config: dict, method: str, metric: str, generations: int, population: int) -> None:
    """
    Run optimization to find the best parameters for the day trading algorithm.
    
    Args:
        config: Configuration dictionary
        method: Optimization method (grid_search, random_search, genetic)
        metric: Metric to optimize for
        generations: Number of generations for genetic algorithm
        population: Population size for genetic algorithm
    """
    from src.optimization.optimizer import run_optimization
    
    # Run optimization
    run_optimization(config, method, metric, generations, population)


def main():
    """Main function to run the day trading algorithm."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the day trading algorithm")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["backtest", "paper", "optimize"], default="paper", help="Mode to run the algorithm in")
    parser.add_argument("--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtest (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="5m", help="Timeframe to use for backtest")
    parser.add_argument("--method", type=str, choices=["grid_search", "random_search", "genetic"], default="genetic", help="Optimization method")
    parser.add_argument("--metric", type=str, default="sharpe_ratio", help="Metric to optimize for")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations for genetic algorithm")
    parser.add_argument("--population", type=int, default=50, help="Population size for genetic algorithm")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run in the specified mode
    if args.mode == "backtest":
        # Set default dates if not provided
        start_date = args.start_date or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
        
        run_backtest(config, start_date, end_date, args.timeframe)
    elif args.mode == "paper":
        run_paper_trading(config)
    elif args.mode == "optimize":
        run_optimization(config, args.method, args.metric, args.generations, args.population)
    else:
        logging.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
