#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct test for the trading UI.
"""

import os
import sys
import time
from datetime import datetime
import random

# Import the TradingUI class directly
from src.ui.trading_ui import TradingUI

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
