#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the trading UI.
"""

import os
import sys
import time
from datetime import datetime
import random

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the TradingUI class
from day_trading_algo.src.ui.trading_ui import TradingUI

# If the above import fails, try this alternative path
try:
    from day_trading_algo.src.ui.trading_ui import TradingUI
except ImportError:
    try:
        sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'day-trading-algo'))
        from src.ui.trading_ui import TradingUI
    except ImportError:
        print("Could not import TradingUI. Make sure the path is correct.")
        sys.exit(1)

def main():
    """Main function."""
    print("Starting Trading UI Test...")

    # Create UI
    ui = TradingUI()

    # Add initial log message
    ui.add_message({
        "type": "log",
        "level": "INFO",
        "message": "Day Trading Algorithm initialized"
    })

    # Add initial statistics
    ui.add_message({
        "type": "statistics",
        "statistics": {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_loss": 0.0,
            "balance": 50.0
        }
    })

    # Set status to ONLINE
    ui.add_message({
        "type": "status",
        "status": "ONLINE"
    })

    # Simulate trading activity
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "NFLX"]
    active_trades = {}
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    balance = 50.0

    # Add log messages
    ui.add_message({
        "type": "log",
        "level": "INFO",
        "message": "Starting trading simulation..."
    })

    ui.add_message({
        "type": "log",
        "level": "INFO",
        "message": "Scanning for trading opportunities..."
    })

    # Run the UI
    ui.run()

if __name__ == "__main__":
    main()
