#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the real-time trading system.

This script initializes and runs the real-time trading system,
which continuously retrieves market data and executes trades.
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
from src.data.realtime import RealTimeDataManager, RealTimeTrader
from src.strategy.basic_swing import BasicSwingStrategy
from src.simulation.simulator import TradeSimulator
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


def main():
    """Main function to run the real-time trading system."""
    parser = argparse.ArgumentParser(description="Run the real-time trading system")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--enable-trading", action="store_true",
                        help="Enable actual trading (simulation by default)")
    parser.add_argument("--update-interval", type=int, default=60,
                        help="Update interval in seconds")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    output_dirs = create_output_dirs(config)
    
    # Set up logging
    logger = setup_logging(config["general"], output_dirs["logs_dir"])
    logger.info("Starting Real-Time Trading System")
    
    # Initialize real-time configuration
    if "realtime" not in config:
        config["realtime"] = {
            "update_interval": args.update_interval,
            "max_retries": 3,
            "retry_delay": 5,
            "timeframe": config["data"]["timeframes"][0],
            "enable_trading": args.enable_trading
        }
    else:
        # Override configuration with command-line arguments
        config["realtime"]["update_interval"] = args.update_interval
        config["realtime"]["enable_trading"] = args.enable_trading
    
    # Initialize strategy
    strategy = BasicSwingStrategy(config["strategy"])
    logger.info(f"Strategy initialized: {strategy.name}")
    
    # Initialize simulator
    simulator = TradeSimulator(config["risk"])
    logger.info(f"Simulator initialized with balance: {simulator.account.current_balance} {simulator.account.currency}")
    
    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer(config["performance"]["metrics"])
    logger.info("Performance analyzer initialized")
    
    # Initialize real-time data manager
    data_manager = RealTimeDataManager(config["data"])
    
    # Initialize real-time trader
    trader = RealTimeTrader(config["realtime"], strategy, None, simulator)
    
    # Add trader as callback to data manager
    data_manager.add_callback(trader.process_data_update)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start real-time data manager
    data_manager.start()
    
    # Main loop
    global running
    running = True
    
    try:
        while running:
            # Sleep to avoid high CPU usage
            time.sleep(1)
            
            # Generate performance report every hour
            if datetime.now().minute == 0 and datetime.now().second < 10:
                # Get trade history
                trade_history = simulator.get_trade_history()
                
                # Generate report
                analyzer.generate_report(trade_history, simulator.initial_balance, output_dirs["reports_dir"])
                
                # Sleep to avoid generating multiple reports
                time.sleep(50)
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    
    finally:
        # Stop real-time data manager
        data_manager.stop()
        
        # Generate final performance report
        trade_history = simulator.get_trade_history()
        analyzer.generate_report(trade_history, simulator.initial_balance, output_dirs["reports_dir"])
        
        # Print final summary
        account_summary = simulator.get_account_summary()
        logger.info("Real-Time Trading System stopped")
        logger.info(f"Final balance: {account_summary['current_balance']} {account_summary['currency']}")
        logger.info(f"Total profit/loss: {account_summary['profit_loss']} {account_summary['currency']} ({account_summary['profit_loss_percent']}%)")
        logger.info(f"Total trades: {account_summary['total_trades']}")
        logger.info(f"Open positions: {account_summary['open_positions']}")


if __name__ == "__main__":
    main()
