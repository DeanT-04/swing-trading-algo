#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time paper trading system for day trading.

This script runs a paper trading simulation using real-time market data,
allowing the algorithm to make trading decisions as if it were trading with real money.
"""

import os
import sys
import time
import json
import logging
import argparse
import yaml
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import threading
import signal
import queue

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.realtime import RealTimeDataProvider
from src.strategy.intraday_momentum import IntradayMomentumStrategy
from src.risk.position_sizing import calculate_position_size
from src.data.models import Stock, TimeFrame, TradeDirection, Trade, Position
from src.performance.metrics import calculate_performance_metrics
from src.utils.logging_utils import setup_logging
from src.utils.notification import send_notification

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Global variables
running = True
data_queue = queue.Queue()
positions = []
trades = []
account_balance = 0.0
initial_balance = 0.0
current_day_trades = 0
max_trades_per_day = 10
trading_hours = {"start": "09:30", "end": "16:00"}
trading_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def signal_handler(sig, frame):
    """Handle interrupt signals to gracefully exit the program."""
    global running
    logger.info("Received interrupt signal. Shutting down...")
    running = False


def load_config(config_path: str) -> Dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


def save_state(state_file: str):
    """
    Save the current state of the paper trading system.
    
    Args:
        state_file: Path to the state file
    """
    state = {
        "timestamp": datetime.now().isoformat(),
        "account_balance": account_balance,
        "positions": [pos.to_dict() for pos in positions],
        "trades": [trade.to_dict() for trade in trades],
        "current_day_trades": current_day_trades
    }
    
    try:
        with open(state_file, 'w') as file:
            json.dump(state, file, indent=4)
        logger.info(f"State saved to {state_file}")
    except Exception as e:
        logger.error(f"Error saving state: {e}")


def load_state(state_file: str) -> bool:
    """
    Load the state of the paper trading system.
    
    Args:
        state_file: Path to the state file
        
    Returns:
        bool: True if state was loaded successfully, False otherwise
    """
    global account_balance, positions, trades, current_day_trades
    
    if not os.path.exists(state_file):
        logger.info(f"No state file found at {state_file}")
        return False
    
    try:
        with open(state_file, 'r') as file:
            state = json.load(file)
        
        account_balance = state.get("account_balance", initial_balance)
        
        # Load positions
        positions = []
        for pos_dict in state.get("positions", []):
            pos = Position.from_dict(pos_dict)
            positions.append(pos)
        
        # Load trades
        trades = []
        for trade_dict in state.get("trades", []):
            trade = Trade.from_dict(trade_dict)
            trades.append(trade)
        
        current_day_trades = state.get("current_day_trades", 0)
        
        logger.info(f"State loaded from {state_file}")
        logger.info(f"Account balance: {account_balance}")
        logger.info(f"Open positions: {len(positions)}")
        logger.info(f"Total trades: {len(trades)}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        return False


def is_trading_time() -> bool:
    """
    Check if the current time is within trading hours.
    
    Returns:
        bool: True if current time is within trading hours, False otherwise
    """
    now = datetime.now()
    
    # Check if today is a trading day
    day_of_week = now.strftime("%A")
    if day_of_week not in trading_days:
        return False
    
    # Parse trading hours
    start_hour, start_minute = map(int, trading_hours["start"].split(":"))
    end_hour, end_minute = map(int, trading_hours["end"].split(":"))
    
    # Create datetime objects for today's trading hours
    market_open = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    market_close = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
    
    # Check if current time is within trading hours
    return market_open <= now <= market_close


def reset_daily_counters():
    """Reset daily counters at the start of a new trading day."""
    global current_day_trades
    
    now = datetime.now()
    day_of_week = now.strftime("%A")
    
    # Only reset if it's a trading day and before market open
    if day_of_week in trading_days:
        start_hour, start_minute = map(int, trading_hours["start"].split(":"))
        market_open = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        
        if now < market_open:
            logger.info("New trading day. Resetting daily counters.")
            current_day_trades = 0


def data_fetcher(data_provider, symbols, timeframe, update_interval):
    """
    Fetch real-time data in a separate thread.
    
    Args:
        data_provider: RealTimeDataProvider instance
        symbols: List of symbols to fetch data for
        timeframe: TimeFrame to fetch data for
        update_interval: Interval in seconds between data updates
    """
    global running, data_queue
    
    logger.info(f"Starting data fetcher thread for {len(symbols)} symbols")
    
    while running:
        try:
            # Only fetch data during trading hours
            if is_trading_time():
                logger.info("Fetching real-time data...")
                
                for symbol in symbols:
                    # Fetch data for the symbol
                    stock = data_provider.get_stock_data(symbol, [timeframe])
                    
                    if stock and stock.data.get(timeframe):
                        # Put the stock data in the queue
                        data_queue.put(stock)
                    else:
                        logger.warning(f"No data available for {symbol}")
            else:
                logger.info("Outside trading hours. Waiting...")
            
            # Sleep for the update interval
            time.sleep(update_interval)
            
        except Exception as e:
            logger.error(f"Error in data fetcher: {e}")
            time.sleep(update_interval)


def process_data(strategy, config):
    """
    Process data from the queue and make trading decisions.
    
    Args:
        strategy: Trading strategy instance
        config: Configuration dictionary
    """
    global running, data_queue, positions, trades, account_balance, current_day_trades
    
    logger.info("Starting data processing thread")
    
    risk_config = config.get("risk", {})
    max_risk_per_trade = risk_config.get("max_risk_per_trade", 0.01)
    max_open_positions = risk_config.get("max_open_positions", 5)
    
    while running:
        try:
            # Get stock data from the queue (with timeout to allow checking running flag)
            try:
                stock = data_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            # Skip processing if outside trading hours
            if not is_trading_time():
                continue
            
            # Skip if we've reached the maximum number of trades for the day
            if current_day_trades >= max_trades_per_day:
                logger.info(f"Maximum daily trades reached ({max_trades_per_day}). Skipping analysis.")
                continue
            
            # Skip if we've reached the maximum number of open positions
            open_positions = [p for p in positions if p.is_open]
            if len(open_positions) >= max_open_positions:
                logger.info(f"Maximum open positions reached ({max_open_positions}). Skipping analysis.")
                
                # Check for exit signals on existing positions
                for position in open_positions:
                    check_exit_conditions(position, stock, strategy)
                
                continue
            
            # Check if we already have a position for this symbol
            symbol_positions = [p for p in open_positions if p.symbol == stock.symbol]
            if symbol_positions:
                logger.info(f"Already have a position for {stock.symbol}. Checking exit conditions.")
                
                # Check for exit signals
                for position in symbol_positions:
                    check_exit_conditions(position, stock, strategy)
                
                continue
            
            # Analyze the stock
            logger.info(f"Analyzing {stock.symbol}...")
            timeframe = TimeFrame.MINUTE_5  # Use 5-minute timeframe for day trading
            analysis = strategy.analyze(stock, timeframe)
            
            if not analysis:
                logger.warning(f"No analysis results for {stock.symbol}")
                continue
            
            # Check for signals
            signals = analysis.get("signals", [])
            
            if signals:
                # Get the latest signal
                signal = signals[-1]
                
                # Set the symbol in the signal
                signal["symbol"] = stock.symbol
                
                # Check if the signal is recent (within the last 5 minutes)
                signal_time = signal["timestamp"]
                if isinstance(signal_time, str):
                    signal_time = pd.to_datetime(signal_time)
                
                now = datetime.now()
                if now - signal_time > timedelta(minutes=5):
                    logger.info(f"Signal for {stock.symbol} is too old ({signal_time}). Skipping.")
                    continue
                
                # Process the signal
                logger.info(f"Found signal for {stock.symbol}: {signal['direction'].name} at {signal['entry_price']}")
                
                # Calculate position size
                position_size = calculate_position_size(
                    account_balance=account_balance,
                    risk_per_trade=max_risk_per_trade,
                    entry_price=signal["entry_price"],
                    stop_loss=signal["stop_loss"],
                    direction=signal["direction"]
                )
                
                # Create a new position
                position = Position(
                    symbol=stock.symbol,
                    direction=signal["direction"],
                    entry_price=signal["entry_price"],
                    stop_loss=signal["stop_loss"],
                    take_profit=signal["take_profit"],
                    size=position_size,
                    entry_time=now,
                    status="open"
                )
                
                # Add the position to the list
                positions.append(position)
                
                # Create a trade record
                trade = Trade(
                    symbol=stock.symbol,
                    direction=signal["direction"],
                    entry_price=signal["entry_price"],
                    entry_time=now,
                    size=position_size,
                    stop_loss=signal["stop_loss"],
                    take_profit=signal["take_profit"],
                    exit_price=None,
                    exit_time=None,
                    profit_loss=None,
                    status="open",
                    reason=signal["reason"]
                )
                
                # Add the trade to the list
                trades.append(trade)
                
                # Increment the daily trade counter
                current_day_trades += 1
                
                # Send notification
                send_notification(
                    f"New {signal['direction'].name} position opened for {stock.symbol} at {signal['entry_price']}"
                )
                
                logger.info(f"Opened {signal['direction'].name} position for {stock.symbol} at {signal['entry_price']}")
                logger.info(f"Position size: {position_size}, Stop loss: {signal['stop_loss']}, Take profit: {signal['take_profit']}")
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")


def check_exit_conditions(position, stock, strategy):
    """
    Check if a position should be closed based on exit conditions.
    
    Args:
        position: Position to check
        stock: Stock data
        strategy: Trading strategy instance
    """
    global account_balance, trades
    
    if not position.is_open:
        return
    
    # Get the latest price
    latest_data = stock.data.get(TimeFrame.MINUTE_5, [])
    if not latest_data:
        logger.warning(f"No data available for {stock.symbol}")
        return
    
    latest_price = latest_data[-1].close
    
    # Check stop loss
    if position.direction == TradeDirection.LONG and latest_price <= position.stop_loss:
        close_position(position, latest_price, "stop_loss")
        return
    
    if position.direction == TradeDirection.SHORT and latest_price >= position.stop_loss:
        close_position(position, latest_price, "stop_loss")
        return
    
    # Check take profit
    if position.direction == TradeDirection.LONG and latest_price >= position.take_profit:
        close_position(position, latest_price, "take_profit")
        return
    
    if position.direction == TradeDirection.SHORT and latest_price <= position.take_profit:
        close_position(position, latest_price, "take_profit")
        return
    
    # Check for exit signals from the strategy
    timeframe = TimeFrame.MINUTE_5
    analysis = strategy.analyze(stock, timeframe)
    
    if not analysis:
        return
    
    # Check for exit signals based on the strategy
    # This is a simplified example - you would implement more sophisticated exit logic here
    if position.direction == TradeDirection.LONG:
        # Exit long if trend turns bearish
        if analysis.get("trend", [""])[0] == "downtrend":
            close_position(position, latest_price, "trend_change")
            return
    else:  # SHORT
        # Exit short if trend turns bullish
        if analysis.get("trend", [""])[0] == "uptrend":
            close_position(position, latest_price, "trend_change")
            return


def close_position(position, exit_price, reason):
    """
    Close a position and update account balance.
    
    Args:
        position: Position to close
        exit_price: Exit price
        reason: Reason for closing the position
    """
    global account_balance, trades
    
    if not position.is_open:
        return
    
    # Calculate profit/loss
    if position.direction == TradeDirection.LONG:
        profit_loss = (exit_price - position.entry_price) * position.size
    else:  # SHORT
        profit_loss = (position.entry_price - exit_price) * position.size
    
    # Update account balance
    account_balance += profit_loss
    
    # Update position
    position.exit_price = exit_price
    position.exit_time = datetime.now()
    position.profit_loss = profit_loss
    position.status = "closed"
    
    # Update corresponding trade
    for trade in trades:
        if (trade.symbol == position.symbol and 
            trade.direction == position.direction and 
            trade.entry_time == position.entry_time and
            trade.status == "open"):
            
            trade.exit_price = exit_price
            trade.exit_time = position.exit_time
            trade.profit_loss = profit_loss
            trade.status = "closed"
            trade.exit_reason = reason
            break
    
    # Send notification
    send_notification(
        f"Closed {position.direction.name} position for {position.symbol} at {exit_price}. "
        f"P/L: {profit_loss:.2f} ({reason})"
    )
    
    logger.info(f"Closed {position.direction.name} position for {position.symbol} at {exit_price}")
    logger.info(f"Profit/Loss: {profit_loss:.2f} ({reason})")
    logger.info(f"New account balance: {account_balance:.2f}")


def generate_performance_report():
    """Generate a performance report for the paper trading session."""
    global trades, account_balance, initial_balance
    
    if not trades:
        logger.info("No trades to report.")
        return
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(trades, initial_balance, account_balance)
    
    # Print report
    logger.info("=== Performance Report ===")
    logger.info(f"Initial Balance: {initial_balance:.2f}")
    logger.info(f"Current Balance: {account_balance:.2f}")
    logger.info(f"Total Return: {metrics['total_return']:.2f} ({metrics['total_return_percent']:.2f}%)")
    logger.info(f"Number of Trades: {metrics['num_trades']}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"Average Win: {metrics['avg_win']:.2f}")
    logger.info(f"Average Loss: {metrics['avg_loss']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info("==========================")
    
    # Save report to file
    report_file = f"reports/paper_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as file:
        json.dump(metrics, file, indent=4)
    
    logger.info(f"Performance report saved to {report_file}")


def main():
    """Main function to run the paper trading system."""
    global running, account_balance, initial_balance, max_trades_per_day, trading_hours, trading_days
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run paper trading simulation")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--state", type=str, help="Path to state file (default: from config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize account balance
    risk_config = config.get("risk", {})
    initial_balance = risk_config.get("initial_balance", 50.0)
    account_balance = initial_balance
    
    # Get paper trading configuration
    paper_config = config.get("paper_trading", {})
    update_interval = paper_config.get("update_interval", 30)
    state_file = args.state or paper_config.get("state_file", "paper_trading_state.json")
    max_trades_per_day = paper_config.get("max_trades_per_day", 10)
    trading_hours = paper_config.get("trading_hours", {"start": "09:30", "end": "16:00"})
    trading_days = paper_config.get("trading_days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    
    # Load state if available
    if load_state(state_file):
        logger.info("Continuing from previous state")
    else:
        logger.info(f"Starting with initial balance: {initial_balance}")
    
    # Initialize data provider
    data_config = config.get("data", {})
    provider_name = data_config.get("provider", "yahoo_finance")
    symbols = data_config.get("symbols", [])
    timeframes = [TimeFrame.from_string(tf) for tf in data_config.get("timeframes", ["1h", "15m", "5m"])]
    
    logger.info(f"Initializing data provider: {provider_name}")
    data_provider = RealTimeDataProvider(config)
    
    # Initialize strategy
    strategy_config = config.get("strategy", {})
    strategy = IntradayMomentumStrategy(strategy_config)
    
    # Start data fetcher thread
    timeframe = TimeFrame.MINUTE_5  # Use 5-minute timeframe for day trading
    fetcher_thread = threading.Thread(
        target=data_fetcher,
        args=(data_provider, symbols, timeframe, update_interval),
        daemon=True
    )
    fetcher_thread.start()
    
    # Start data processing thread
    processor_thread = threading.Thread(
        target=process_data,
        args=(strategy, config),
        daemon=True
    )
    processor_thread.start()
    
    # Main loop
    try:
        while running:
            # Reset daily counters if needed
            reset_daily_counters()
            
            # Save state periodically
            save_state(state_file)
            
            # Sleep for a while
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Set running flag to False to stop threads
        running = False
        
        # Wait for threads to finish
        fetcher_thread.join(timeout=5)
        processor_thread.join(timeout=5)
        
        # Generate performance report
        generate_performance_report()
        
        # Save final state
        save_state(state_file)
        
        logger.info("Paper trading system shutdown complete")


if __name__ == "__main__":
    main()
