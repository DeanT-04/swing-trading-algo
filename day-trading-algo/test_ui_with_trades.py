#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the trading UI with simulated trades.
"""

import os
import sys
import time
import threading
from datetime import datetime
import random

# Import the TradingUI class directly
from src.ui.trading_ui import TradingUI

def simulate_trading(ui):
    """
    Simulate trading activity.

    Args:
        ui: TradingUI instance
    """
    # Simulate trading activity
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "NFLX"]
    active_trades = {}
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    balance = 50.0  # Starting balance of £50
    total_profit_loss = 0.0  # Track total profit/loss from all trades
    closed_trades = []  # Keep track of all closed trades

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
        if random.random() < 0.2 and len(active_trades) < 5 and balance > 5.0:  # 20% chance to open a trade if we have enough balance
            symbol = random.choice(symbols)
            direction = random.choice(["LONG", "SHORT"])
            entry_price = random.uniform(20.0, 200.0)  # Stock prices between £20 and £200

            # Calculate position size based on available balance
            # If price > £50, buy partial shares to stay within budget
            max_affordable_size = balance * 0.1 / entry_price  # Use only 10% of balance at most

            if entry_price > 50.0:
                # For expensive stocks, buy partial shares
                position_size = round(max_affordable_size, 2)  # Round to 2 decimal places
                if position_size < 0.1:  # Minimum position size
                    position_size = 0.1
            else:
                # For cheaper stocks, buy whole shares if possible
                position_size = min(int(max_affordable_size), 5)  # Max 5 shares
                if position_size < 1:  # Ensure at least 1 share
                    position_size = 1

            # Calculate cost
            cost = entry_price * position_size

            # Check if we can afford it
            if cost > balance:
                # Skip this trade if we can't afford it
                continue

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
                "size": position_size,
                "cost": cost
            }

            # Add to active trades
            active_trades[trade_id] = trade

            # Update total trades
            total_trades += 1

            # Update balance
            balance -= cost  # Deduct full cost from balance

            # Create statistics
            statistics = {
                "total_trades": total_trades,
                "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                "total_profit_loss": total_profit_loss,
                "active_profit_loss": sum(t.get("profit_loss", 0.0) for t in active_trades.values()),
                "balance": balance,
                "exposure_percent": (sum(t.get("cost", 0.0) for t in active_trades.values()) / balance * 100) if balance > 0 else 0.0,
                "max_position_pct": 10.0,
                "drawdown_percent": 0.0
            }

            # Add trade to UI
            ui.add_message({
                "type": "trade_opened",
                "trade": trade,
                "statistics": statistics
            })

            # Update statistics
            ui.add_message({
                "type": "statistics",
                "statistics": {
                    "total_trades": total_trades,
                    "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                    "total_profit_loss": total_profit_loss,
                    "active_profit_loss": sum(t.get("profit_loss", 0.0) for t in active_trades.values()),
                    "balance": balance
                }
            })

            # Add log message
            ui.add_message({
                "type": "log",
                "level": "INFO",
                "message": f"Trade opened: {direction} {symbol} at £{entry_price:.2f}, Size: {position_size}, Cost: £{cost:.2f}, Balance: £{balance:.2f}"
            })

        # Simulate updating prices for active trades
        for trade_id, trade in list(active_trades.items()):
            # Update price with realistic price movement (0.1% to 1% change)
            price_change_percent = random.uniform(-0.01, 0.01)  # -1% to +1%
            price_change = trade["current_price"] * price_change_percent
            new_price = max(trade["current_price"] + price_change, 0.01)  # Ensure price doesn't go negative

            # Calculate profit/loss
            position_size = trade["size"]
            if trade["direction"] == "LONG":
                profit_loss = (new_price - trade["entry_price"]) * position_size
            else:  # SHORT
                profit_loss = (trade["entry_price"] - new_price) * position_size

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

            # Determine if it's a win or loss based on current profit/loss
            current_profit_loss = trade.get("profit_loss", 0.0)

            # Force a win 95% of the time if currently losing
            if current_profit_loss <= 0 and random.random() < 0.95:
                # Calculate a winning profit (0.5% to 5% of position value)
                position_value = trade["entry_price"] * trade["size"]
                profit_amount = position_value * random.uniform(0.005, 0.05)
                exit_price = trade["entry_price"] + (profit_amount / trade["size"] if trade["direction"] == "LONG" else -profit_amount / trade["size"])
                profit_loss = profit_amount
                winning_trades += 1
            elif current_profit_loss > 0:
                # Already winning, close with current profit
                exit_price = trade["current_price"]
                profit_loss = current_profit_loss
                winning_trades += 1
            else:
                # Take a small loss (0.5% to 2% of position value)
                position_value = trade["entry_price"] * trade["size"]
                loss_amount = position_value * random.uniform(0.005, 0.02)
                exit_price = trade["entry_price"] - (loss_amount / trade["size"] if trade["direction"] == "LONG" else -loss_amount / trade["size"])
                profit_loss = -loss_amount
                losing_trades += 1

            # Ensure exit price is positive
            exit_price = max(exit_price, 0.01)

            # Calculate return amount (cost + profit/loss)
            position_size = trade["size"]
            return_amount = (exit_price * position_size)

            # Update balance
            balance += return_amount

            # Create closed trade
            closed_trade = {
                "id": trade_id,
                "symbol": trade["symbol"],
                "direction": trade["direction"],
                "entry_time": trade["entry_time"],
                "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "profit_loss": profit_loss,
                "size": position_size
            }

            # Create statistics
            statistics = {
                "total_trades": total_trades,
                "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                "total_profit_loss": total_profit_loss,
                "active_profit_loss": sum(t.get("profit_loss", 0.0) for t in active_trades.values()),
                "balance": balance,
                "exposure_percent": (sum(t.get("cost", 0.0) for t in active_trades.values()) / balance * 100) if balance > 0 else 0.0,
                "max_position_pct": 10.0,
                "drawdown_percent": 0.0
            }

            # Add to UI
            ui.add_message({
                "type": "trade_closed",
                "trade": closed_trade,
                "statistics": statistics
            })

            # Add to closed trades list
            closed_trades.append(closed_trade)

            # Update total profit/loss
            total_profit_loss += profit_loss

            # Update statistics
            ui.add_message({
                "type": "statistics",
                "statistics": {
                    "total_trades": total_trades,
                    "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                    "total_profit_loss": total_profit_loss,  # Use the cumulative profit/loss
                    "active_profit_loss": sum(t.get("profit_loss", 0.0) for t in active_trades.values()),
                    "balance": balance
                }
            })

            # Add log message
            result = "WIN" if profit_loss > 0 else "LOSE"
            ui.add_message({
                "type": "log",
                "level": "INFO",
                "message": f"Trade closed: {trade['direction']} {trade['symbol']} at £{exit_price:.2f}, Size: {position_size}, {result}, P/L: £{profit_loss:.2f}, Balance: £{balance:.2f}"
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

def main():
    """Main function."""
    print("Starting Trading UI Test with Simulated Trades...")

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
            "total_profit_loss": 0.0,
            "active_profit_loss": 0.0,
            "balance": 50.0
        }
    })

    # Set status to ONLINE
    ui.add_message({
        "type": "status",
        "status": "ONLINE"
    })

    # Start trading simulation in a separate thread
    trading_thread = threading.Thread(target=simulate_trading, args=(ui,))
    trading_thread.daemon = True
    trading_thread.start()

    # Run the UI
    ui.run()

if __name__ == "__main__":
    main()
