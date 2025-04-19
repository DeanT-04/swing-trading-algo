#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated Day Trading Script

This script automatically starts the day trading algorithm at market open,
monitors multiple stocks, executes trades, and logs all trade details.
It runs autonomously and can be scheduled to start automatically.
Includes a graphical user interface to monitor trading activity.
"""

import os
import sys
import time
import yaml
import logging
import argparse
import schedule
import pandas as pd
import threading
import uuid
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
from src.ui.trading_ui import TradingUI
from src.risk.risk_manager import RiskManager


# Global variables
running = True
paper_trader = None
logger = logging.getLogger(__name__)
ui = None
active_trades = {}
total_trades = 0
winning_trades = 0
losing_trades = 0
balance = 50.0
total_profit_loss = 0.0  # Track total profit/loss from all trades
closed_trades = []  # Keep track of all closed trades
risk_manager = None  # Risk manager instance


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


def on_trade_opened(trade_data):
    """
    Callback function for when a trade is opened.

    Args:
        trade_data: Trade data dictionary
    """
    global ui, active_trades, total_trades, balance, total_profit_loss, risk_manager

    # Generate a unique ID for the trade
    trade_id = str(uuid.uuid4())

    # Calculate position size and cost
    symbol = trade_data.get("symbol", "")
    direction = trade_data.get("direction", "")
    entry_price = trade_data.get("entry_price", 0.0)

    # Check with risk manager if we can open this trade
    can_trade = True
    position_size = trade_data.get("size", 1.0)
    cost = position_size * entry_price

    if risk_manager:
        # Check if we can open this trade based on risk parameters
        can_trade, position_size, cost = risk_manager.can_open_trade(symbol, entry_price, direction, balance)

        if not can_trade:
            logger.warning(f"Risk manager prevented opening trade for {symbol} at ${entry_price:.2f}")
            if ui:
                ui.add_message({
                    "type": "log",
                    "level": "WARNING",
                    "message": f"Risk manager prevented opening trade for {symbol} at ${entry_price:.2f}"
                })
            return

        # Update trade data with new position size and cost
        trade_data["size"] = position_size
        trade_data["cost"] = cost

    # Add trade to active trades
    active_trades[trade_id] = trade_data

    # Update total trades
    total_trades += 1

    # Update balance
    balance -= cost

    # Register trade with risk manager
    if risk_manager:
        risk_manager.register_trade(trade_id, trade_data)

    # Format trade data for UI
    ui_trade = {
        "id": trade_id,
        "symbol": symbol,
        "direction": direction,
        "entry_time": trade_data.get("entry_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "entry_price": entry_price,
        "current_price": entry_price,
        "profit_loss": 0.0,
        "size": position_size,
        "cost": cost
    }

    # Get statistics for UI
    statistics = {
        "total_trades": total_trades,
        "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
        "total_profit_loss": total_profit_loss,
        "active_profit_loss": sum(t.get("profit_loss", 0.0) for t in active_trades.values()),
        "balance": balance,
        "exposure_percent": risk_metrics.get("exposure_pct", 0.0) * 100 if risk_metrics else 0.0,
        "max_position_pct": risk_manager.max_position_pct * 100 if risk_manager else 10.0
    }

    # Send trade to UI
    if ui:
        ui.add_message({
            "type": "trade_opened",
            "trade": ui_trade,
            "statistics": statistics
        })

        # Get risk metrics
        risk_metrics = {}
        if risk_manager:
            risk_metrics = risk_manager.get_risk_metrics()

        # Update statistics
        ui.add_message({
            "type": "statistics",
            "statistics": {
                "total_trades": total_trades,
                "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                "total_profit_loss": total_profit_loss,
                "active_profit_loss": sum(t.get("profit_loss", 0.0) for t in active_trades.values()),
                "balance": balance,
                "exposure_percent": risk_metrics.get("exposure_pct", 0.0) * 100 if risk_metrics else 0.0,
                "max_position_pct": risk_manager.max_position_pct * 100 if risk_manager else 10.0
            }
        })

    # Log trade
    logger.info(f"Trade opened: {direction} {symbol} at ${entry_price:.2f}, Size: {position_size}, Cost: ${cost:.2f}, Balance: ${balance:.2f}")


def on_trade_closed(trade_data):
    """
    Callback function for when a trade is closed.

    Args:
        trade_data: Trade data dictionary
    """
    global ui, active_trades, winning_trades, losing_trades, balance, total_profit_loss, closed_trades, risk_manager

    # Find the trade in active trades
    trade_id = None
    for tid, trade in active_trades.items():
        if trade.get("symbol") == trade_data.get("symbol") and trade.get("direction") == trade_data.get("direction"):
            trade_id = tid
            break

    if trade_id is None:
        logger.warning(f"Trade not found in active trades: {trade_data}")
        return

    # Get the trade from active trades
    trade = active_trades.pop(trade_id, {})

    # Calculate profit/loss
    profit_loss = trade_data.get("profit_loss", 0.0)

    # Calculate return amount (cost + profit/loss)
    position_size = trade_data.get("size", trade.get("size", 1.0))
    exit_price = trade_data.get("exit_price", 0.0)
    return_amount = exit_price * position_size

    # Update balance
    balance += return_amount

    # Update win/loss count
    if profit_loss > 0:
        winning_trades += 1
    else:
        losing_trades += 1

    # Update total profit/loss
    total_profit_loss += profit_loss

    # Unregister trade with risk manager
    if risk_manager:
        risk_manager.unregister_trade(trade_id, profit_loss)

    # Format trade data for UI
    ui_trade = {
        "id": trade_id,
        "symbol": trade_data.get("symbol", ""),
        "direction": trade_data.get("direction", ""),
        "entry_time": trade.get("entry_time", ""),
        "exit_time": trade_data.get("exit_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "entry_price": trade.get("entry_price", 0.0),
        "exit_price": exit_price,
        "profit_loss": profit_loss,
        "size": position_size
    }

    # Add to closed trades list
    closed_trades.append(ui_trade)

    # Get statistics for UI
    statistics = {
        "total_trades": total_trades,
        "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
        "total_profit_loss": total_profit_loss,
        "active_profit_loss": sum(t.get("profit_loss", 0.0) for t in active_trades.values()),
        "balance": balance,
        "exposure_percent": risk_metrics.get("exposure_pct", 0.0) * 100 if risk_metrics else 0.0,
        "drawdown_percent": risk_metrics.get("drawdown", 0.0) * 100 if risk_metrics else 0.0
    }

    # Send trade to UI
    if ui:
        ui.add_message({
            "type": "trade_closed",
            "trade": ui_trade,
            "statistics": statistics
        })

        # Get risk metrics
        risk_metrics = {}
        if risk_manager:
            risk_metrics = risk_manager.get_risk_metrics()

        # Update statistics
        ui.add_message({
            "type": "statistics",
            "statistics": {
                "total_trades": total_trades,
                "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                "total_profit_loss": total_profit_loss,  # Use the cumulative profit/loss
                "active_profit_loss": sum(t.get("profit_loss", 0.0) for t in active_trades.values()),
                "balance": balance,
                "exposure_percent": risk_metrics.get("exposure_pct", 0.0) * 100 if risk_metrics else 0.0,
                "drawdown_percent": risk_metrics.get("drawdown", 0.0) * 100 if risk_metrics else 0.0
            }
        })

    # Log trade
    result = "WIN" if profit_loss > 0 else "LOSE"
    logger.info(f"Trade closed: {ui_trade['direction']} {ui_trade['symbol']} at ${ui_trade['exit_price']:.2f}, Size: {ui_trade['size']}, {result}, P/L: ${profit_loss:.2f}, Balance: ${balance:.2f}")


def on_price_update(symbol, price):
    """
    Callback function for price updates.

    Args:
        symbol: Symbol that was updated
        price: New price
    """
    global ui, active_trades

    # Update active trades with new price
    for trade_id, trade in active_trades.items():
        if trade.get("symbol") == symbol:
            # Calculate profit/loss
            entry_price = trade.get("entry_price", 0.0)
            size = trade.get("size", 1.0)
            direction = trade.get("direction", "")

            if direction == "LONG":
                profit_loss = (price - entry_price) * size
            else:  # SHORT
                profit_loss = (entry_price - price) * size

            # Update trade
            trade["current_price"] = price
            trade["profit_loss"] = profit_loss

            # Update UI
            if ui:
                ui.update_active_trade(trade_id, price, profit_loss)


def start_trading(config: dict, symbols: list, timeframe: TimeFrame):
    """
    Start the day trading algorithm.

    Args:
        config: Configuration dictionary
        symbols: List of symbols to trade
        timeframe: Timeframe to use
    """
    global paper_trader, ui, risk_manager, balance

    logger.info(f"Starting day trading algorithm with {len(symbols)} symbols")

    # Update UI status
    if ui:
        ui.add_message({
            "type": "status",
            "status": "ONLINE"
        })

        ui.add_message({
            "type": "log",
            "level": "INFO",
            "message": f"Starting day trading algorithm with {len(symbols)} symbols"
        })

    # Initialize risk manager
    risk_config = config.get("risk", {})
    if not "initial_balance" in risk_config:
        risk_config["initial_balance"] = balance
    if not "max_position_pct" in risk_config:
        risk_config["max_position_pct"] = 0.1  # Limit to 10% of balance per trade

    config["risk"] = risk_config
    risk_manager = RiskManager(config)

    logger.info(f"Risk Manager initialized with max position: {risk_manager.max_position_pct*100}% of balance")
    if ui:
        ui.add_message({
            "type": "log",
            "level": "INFO",
            "message": f"Risk Manager initialized with max position: {risk_manager.max_position_pct*100}% of balance"
        })

    # Initialize data provider
    data_provider = YahooFinanceProvider()

    # Initialize strategy
    strategy = IntradayStrategy(config)

    # Initialize paper trader with callbacks
    paper_trader = PaperTrader(config)
    paper_trader.set_callbacks({
        "on_trade_opened": on_trade_opened,
        "on_trade_closed": on_trade_closed,
        "on_price_update": on_price_update
    })

    # Start paper trading
    try:
        paper_trader.run(
            symbols=symbols,
            timeframe=timeframe,
            strategy=strategy,
            data_provider=data_provider,
            duration_hours=6.5,  # Standard market hours duration
            risk_manager=risk_manager  # Pass risk manager to paper trader
        )
    except Exception as e:
        logger.error(f"Error in paper trading: {e}")
        if ui:
            ui.add_message({
                "type": "log",
                "level": "ERROR",
                "message": f"Error in paper trading: {e}"
            })
    finally:
        if paper_trader:
            paper_trader.stop()
            logger.info("Paper trading stopped")

            # Update UI status
            if ui:
                ui.add_message({
                    "type": "status",
                    "status": "OFFLINE"
                })

                ui.add_message({
                    "type": "log",
                    "level": "INFO",
                    "message": "Paper trading stopped"
                })


def stop_trading():
    """Stop the day trading algorithm."""
    global paper_trader, ui

    if paper_trader:
        logger.info("Stopping day trading algorithm")
        paper_trader.stop()

        # Update UI status
        if ui:
            ui.add_message({
                "type": "status",
                "status": "OFFLINE"
            })

            ui.add_message({
                "type": "log",
                "level": "INFO",
                "message": "Day trading algorithm stopped"
            })


def signal_handler(sig, frame):
    """Handle interrupt signals."""
    global running, ui

    logger.info("Received interrupt signal, shutting down...")
    running = False
    stop_trading()

    # Update UI
    if ui:
        ui.add_message({
            "type": "log",
            "level": "WARNING",
            "message": "Received interrupt signal, shutting down..."
        })

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


def process_ui_commands():
    """
    Process commands from the UI.
    """
    global ui, running

    if ui and not ui.queue.empty():
        try:
            message = ui.queue.get_nowait()
            if message.get("type") == "command":
                command = message.get("command")

                if command == "start":
                    # Start trading
                    logger.info("Received start command from UI")
                    # This will be handled by the UI button callback
                elif command == "stop":
                    # Stop trading
                    logger.info("Received stop command from UI")
                    stop_trading()
                elif command == "exit":
                    # Exit the application
                    logger.info("Received exit command from UI")
                    running = False

            ui.queue.task_done()
        except Exception as e:
            logger.error(f"Error processing UI command: {e}")


def main():
    """Main function."""
    global running, ui

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
    parser.add_argument("--no-ui", action="store_true",
                        help="Run without the graphical user interface")

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

    # Initialize UI
    if not args.no_ui:
        # Create UI in a separate thread
        ui = TradingUI()
        ui_thread = threading.Thread(target=ui.run)
        ui_thread.daemon = True
        ui_thread.start()

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

                # Update UI
                if ui:
                    ui.add_message({
                        "type": "log",
                        "level": "INFO",
                        "message": f"Market is closed, next open at {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                    })

    # Main loop
    while running:
        # Process UI commands
        if ui:
            process_ui_commands()

        # Run scheduled tasks
        schedule.run_pending()

        # Sleep to avoid high CPU usage
        time.sleep(0.1)


if __name__ == "__main__":
    main()
