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

from src.ui.trading_ui import TradingUI

def main():
    """Main function."""
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
    
    # Start UI in a separate thread
    import threading
    ui_thread = threading.Thread(target=ui.run)
    ui_thread.daemon = True
    ui_thread.start()
    
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
    
    # Simulate trading for 60 seconds
    start_time = time.time()
    while time.time() - start_time < 60:
        # Simulate opening a trade
        if random.random() < 0.2 and len(active_trades) < 5:  # 20% chance to open a trade
            symbol = random.choice(symbols)
            direction = random.choice(["LONG", "SHORT"])
            entry_price = random.uniform(50.0, 200.0)
            trade_id = f"trade_{total_trades}"
            
            # Create trade
            trade = {
                "id": trade_id,
                "symbol": symbol,
                "direction": direction,
                "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "entry_price": entry_price,
                "current_price": entry_price,
                "profit_loss": 0.0,
                "size": 1.0
            }
            
            # Add to active trades
            active_trades[trade_id] = trade
            
            # Update total trades
            total_trades += 1
            
            # Update balance
            balance -= entry_price * 0.01  # Simulate commission
            
            # Add trade to UI
            ui.add_message({
                "type": "trade_opened",
                "trade": trade
            })
            
            # Update statistics
            ui.add_message({
                "type": "statistics",
                "statistics": {
                    "total_trades": total_trades,
                    "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                    "profit_loss": sum(t.get("profit_loss", 0.0) for t in active_trades.values()),
                    "balance": balance
                }
            })
            
            # Add log message
            ui.add_message({
                "type": "log",
                "level": "INFO",
                "message": f"Trade opened: {direction} {symbol} at ${entry_price:.2f}"
            })
        
        # Simulate updating prices for active trades
        for trade_id, trade in list(active_trades.items()):
            # Update price
            price_change = random.uniform(-2.0, 2.0)
            new_price = trade["current_price"] + price_change
            
            # Calculate profit/loss
            if trade["direction"] == "LONG":
                profit_loss = (new_price - trade["entry_price"]) * trade["size"]
            else:  # SHORT
                profit_loss = (trade["entry_price"] - new_price) * trade["size"]
            
            # Update trade
            trade["current_price"] = new_price
            trade["profit_loss"] = profit_loss
            
            # Update UI
            ui.update_active_trade(trade_id, new_price, profit_loss)
        
        # Simulate closing a trade
        if random.random() < 0.1 and active_trades:  # 10% chance to close a trade
            # Select a random trade to close
            trade_id = random.choice(list(active_trades.keys()))
            trade = active_trades.pop(trade_id)
            
            # Determine if it's a win or loss
            is_win = random.random() < 0.95  # 95% win rate
            
            # Calculate profit/loss
            if is_win:
                profit_loss = random.uniform(1.0, 5.0)
                winning_trades += 1
            else:
                profit_loss = -random.uniform(0.5, 2.0)
                losing_trades += 1
            
            # Calculate exit price
            if trade["direction"] == "LONG":
                exit_price = trade["entry_price"] + profit_loss
            else:  # SHORT
                exit_price = trade["entry_price"] - profit_loss
            
            # Update balance
            balance += exit_price * trade["size"]
            
            # Create closed trade
            closed_trade = {
                "id": trade_id,
                "symbol": trade["symbol"],
                "direction": trade["direction"],
                "entry_time": trade["entry_time"],
                "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "profit_loss": profit_loss
            }
            
            # Add to UI
            ui.add_message({
                "type": "trade_closed",
                "trade": closed_trade
            })
            
            # Update statistics
            ui.add_message({
                "type": "statistics",
                "statistics": {
                    "total_trades": total_trades,
                    "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                    "profit_loss": sum(t.get("profit_loss", 0.0) for t in active_trades.values()),
                    "balance": balance
                }
            })
            
            # Add log message
            result = "WIN" if profit_loss > 0 else "LOSE"
            ui.add_message({
                "type": "log",
                "level": "INFO",
                "message": f"Trade closed: {trade['direction']} {trade['symbol']} at ${exit_price:.2f}, {result}, P/L: ${profit_loss:.2f}"
            })
        
        # Sleep for a short time
        time.sleep(1)
    
    # Set status to OFFLINE
    ui.add_message({
        "type": "status",
        "status": "OFFLINE"
    })
    
    # Add final log message
    ui.add_message({
        "type": "log",
        "level": "INFO",
        "message": "Trading simulation completed"
    })
    
    # Wait for UI thread to finish
    ui_thread.join()

if __name__ == "__main__":
    main()
