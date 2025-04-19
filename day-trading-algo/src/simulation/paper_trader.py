#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paper trading module for the day trading algorithm.

This module implements a paper trading system to simulate real trading.
"""

import os
import logging
import json
import pandas as pd
import csv
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import time
import threading
import queue

from src.data.models import Stock, TimeFrame, TradeDirection, Trade, Position
from src.data.provider import YahooFinanceProvider
from src.strategy.intraday_strategy import IntradayStrategy
from src.risk.position_sizing import calculate_position_size
from src.ui.console_ui import ConsoleUI

# Set up logging
logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper trading system for the day trading algorithm.

    This class simulates real trading by executing trades based on
    strategy signals using real-time market data.
    """

    def __init__(self, config: Dict):
        """
        Initialize the paper trading system.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paper_config = config.get("paper_trading", {})

        # Initialize account
        risk_config = config.get("risk", {})
        self.initial_balance = risk_config.get("initial_balance", 50.0)
        self.account_balance = self.initial_balance
        self.currency = risk_config.get("currency", "GBP")

        # Risk parameters
        self.max_risk_per_trade = risk_config.get("max_risk_per_trade", 0.01)
        self.max_open_positions = risk_config.get("max_open_positions", 5)

        # Trading parameters
        self.update_interval = self.paper_config.get("update_interval", 30)  # seconds
        self.state_file = self.paper_config.get("state_file", "paper_trading_state.json")
        self.timeframe = TimeFrame.from_string(self.paper_config.get("timeframe", "5m"))
        self.max_trades_per_day = self.paper_config.get("max_trades_per_day", 10)

        # Trading hours
        trading_hours = self.paper_config.get("trading_hours", {})
        self.trading_start = trading_hours.get("start", "09:30")
        self.trading_end = trading_hours.get("end", "16:00")

        # Trading days
        self.trading_days = self.paper_config.get("trading_days",
                                                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

        # Initialize data provider
        self.data_provider = YahooFinanceProvider(config)

        # Initialize strategy
        self.strategy = IntradayStrategy(config)

        # Initialize state
        self.positions = []
        self.trades = []
        self.current_day_trades = 0

        # Initialize data queue and running flag
        self.data_queue = queue.Queue()
        self.running = True

        # Initialize callbacks
        self.callbacks = {
            "on_trade_opened": None,
            "on_trade_closed": None,
            "on_price_update": None
        }

        # Initialize console UI
        self.console_ui = None
        self.use_console_ui = self.paper_config.get("use_console_ui", True)

        if self.use_console_ui:
            try:
                self.console_ui = ConsoleUI({
                    "update_interval": 1,
                    "initial_balance": self.initial_balance
                })
                self.console_ui.start()
            except Exception as e:
                logger.error(f"Error initializing console UI: {e}")
                self.console_ui = None

        logger.info(f"Initialized paper trading system with {self.initial_balance} {self.currency}")

    def set_callbacks(self, callbacks: Dict):
        """
        Set callbacks for trade events.

        Args:
            callbacks: Dictionary of callback functions
        """
        for key, callback in callbacks.items():
            if key in self.callbacks:
                self.callbacks[key] = callback
                logger.info(f"Set callback for {key}")
            else:
                logger.warning(f"Unknown callback type: {key}")

    def run(self, symbols: List[str], timeframe: TimeFrame, strategy=None, data_provider=None,
            duration_hours: float = 6.5, risk_manager=None, test_mode: bool = False):
        """
        Run the paper trading system with the specified parameters.

        Args:
            symbols: List of symbols to trade
            timeframe: Timeframe to use
            strategy: Strategy to use (optional, uses the one from constructor if not provided)
            data_provider: Data provider to use (optional, uses the one from constructor if not provided)
            duration_hours: Duration in hours to run the simulation
            risk_manager: Risk manager to use (optional)
            test_mode: Whether to run in test mode (ignoring market hours)
        """
        logger.info(f"Running paper trading for {len(symbols)} symbols with {timeframe} timeframe")

        # Set test mode flag
        self.test_mode = test_mode
        if test_mode:
            logger.info("Running in test mode - ignoring market hours")

        # Override strategy and data provider if provided
        if strategy:
            self.strategy = strategy
        if data_provider:
            self.data_provider = data_provider

        # Store risk manager if provided
        self.risk_manager = risk_manager

        # Start the paper trading system
        self.start(symbols, timeframe, duration_hours)

    def stop(self):
        """
        Stop the paper trading system.
        """
        logger.info("Stopping paper trading system")
        self.running = False

        # Stop console UI if it's running
        if self.console_ui:
            self.console_ui.stop()

    def start(self, symbols: List[str], timeframe: TimeFrame, duration_hours: float = 6.5):
        """Start the paper trading system."""
        logger.info("Starting paper trading system")

        # Load state if available
        self.load_state()

        # Use provided symbols or get from config
        if not symbols:
            symbols = self.config.get("data", {}).get("symbols", [])
            if not symbols:
                logger.error("No symbols configured for trading")
                return False

        # Store the timeframe
        self.timeframe = timeframe

        # Calculate end time based on duration
        self.end_time = datetime.now() + timedelta(hours=duration_hours)

        # Start data fetcher thread
        fetcher_thread = threading.Thread(
            target=self._data_fetcher,
            args=(symbols, self.timeframe, self.update_interval),
            daemon=True
        )
        fetcher_thread.start()

        # Store the symbols for later use
        self.symbols = symbols

        # Start processor thread
        processor_thread = threading.Thread(
            target=self._process_data,
            daemon=True
        )
        processor_thread.start()

        # Main loop
        try:
            while self.running:
                # Reset daily counters if needed
                self._reset_daily_counters()

                # Save state periodically
                self.save_state()

                # Sleep for a while
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Set running flag to False to stop threads
            self.running = False

            # Wait for threads to finish
            fetcher_thread.join(timeout=5)
            processor_thread.join(timeout=5)

            # Save final state
            self.save_state()

            # Generate performance report
            self._generate_performance_report()

            logger.info("Paper trading system shutdown complete")

            return True

    def _data_fetcher(self, symbols: List[str], timeframe: TimeFrame, update_interval: int):
        """
        Fetch real-time data in a separate thread.

        Args:
            symbols: List of symbols to fetch data for
            timeframe: TimeFrame to fetch data for
            update_interval: Interval in seconds between data updates
        """
        logger.info(f"Starting data fetcher thread for {len(symbols)} symbols")

        while self.running:
            try:
                # Only fetch data during trading hours
                if self._is_trading_time():
                    logger.info("Fetching real-time data...")

                    # Update console UI with algorithm status
                    if self.console_ui:
                        self.console_ui.update_data("status", {"status": "ONLINE"})

                    for symbol in symbols:
                        # Fetch data for the symbol
                        stock = self.data_provider.get_stock_data(symbol, [timeframe])

                        if stock and stock.data.get(timeframe):
                            # Put the stock data in the queue
                            self.data_queue.put(stock)

                            # Call the on_price_update callback if set
                            if self.callbacks["on_price_update"]:
                                latest_data = stock.data.get(timeframe, [])
                                if latest_data:
                                    latest_price = latest_data[-1].close
                                    self.callbacks["on_price_update"](symbol, latest_price)

                                    # Update any active trades with the latest price
                                    active_trades = self.console_ui.active_trades if self.console_ui else {}
                                    for trade_id, trade in active_trades.items():
                                        if trade.get("symbol") == symbol:
                                            trade["current_price"] = latest_price
                                            profit_loss = self._calculate_profit_loss(trade)
                                            trade["profit_loss"] = profit_loss
                        else:
                            logger.warning(f"No data available for {symbol}")
                else:
                    logger.info("Outside trading hours. Waiting...")

                    # Update console UI with algorithm status
                    if self.console_ui:
                        self.console_ui.update_data("status", {"status": "WAITING"})

                # Sleep for the update interval
                time.sleep(update_interval)

            except Exception as e:
                logger.error(f"Error in data fetcher: {e}")
                time.sleep(update_interval)

    def _process_data(self):
        """Process data from the queue and make trading decisions."""
        logger.info("Starting data processing thread")

        while self.running:
            try:
                # Get stock data from the queue (with timeout to allow checking running flag)
                try:
                    stock = self.data_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Skip processing if outside trading hours
                if not self._is_trading_time():
                    continue

                # Skip if we've reached the maximum number of trades for the day
                if self.current_day_trades >= self.max_trades_per_day:
                    logger.info(f"Maximum daily trades reached ({self.max_trades_per_day}). Skipping analysis.")
                    continue

                # Skip if we've reached the maximum number of open positions
                open_positions = [p for p in self.positions if p.is_open]
                if len(open_positions) >= self.max_open_positions:
                    logger.info(f"Maximum open positions reached ({self.max_open_positions}). Skipping analysis.")

                    # Check for exit signals on existing positions
                    for position in open_positions:
                        self._check_exit_conditions(position, stock)

                    continue

                # Check if we already have a position for this symbol
                symbol_positions = [p for p in open_positions if p.symbol == stock.symbol]
                if symbol_positions:
                    logger.info(f"Already have a position for {stock.symbol}. Checking exit conditions.")

                    # Check for exit signals
                    for position in symbol_positions:
                        self._check_exit_conditions(position, stock)

                    continue

                # Analyze the stock
                logger.info(f"Analyzing {stock.symbol}...")
                analysis = self.strategy.analyze(stock, self.timeframe)

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

                    # Make sure both times are timezone-aware or timezone-naive
                    import pytz
                    london = pytz.timezone('Europe/London')
                    now = datetime.now(london)

                    # Convert signal_time to timezone-aware if it's naive
                    if signal_time.tzinfo is None:
                        signal_time = london.localize(signal_time)

                    # Calculate time difference
                    time_diff = now - signal_time
                    if time_diff > timedelta(minutes=5):
                        logger.info(f"Signal for {stock.symbol} is too old ({signal_time}). Skipping.")
                        continue

                    # Process the signal
                    logger.info(f"Found signal for {stock.symbol}: {signal['direction'].name} at {signal['entry_price']}")

                    # Calculate position size
                    position_size = calculate_position_size(
                        account_balance=self.account_balance,
                        risk_per_trade=self.max_risk_per_trade,
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
                        entry_time=now.replace(tzinfo=None),  # Remove timezone info to avoid issues
                        status="open"
                    )

                    # Add the position to the list
                    self.positions.append(position)

                    # Create a trade record
                    trade = Trade(
                        symbol=stock.symbol,
                        direction=signal["direction"],
                        entry_price=signal["entry_price"],
                        entry_time=now.replace(tzinfo=None),  # Remove timezone info to avoid issues
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
                    self.trades.append(trade)

                    # Increment the daily trade counter
                    self.current_day_trades += 1

                    # Create trade data dictionary
                    trade_data = {
                        "id": f"trade_{len(self.trades)}",
                        "symbol": stock.symbol,
                        "direction": signal["direction"].name,
                        "entry_price": signal["entry_price"],
                        "current_price": signal["entry_price"],  # Initially the same as entry price
                        "entry_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "size": position_size,
                        "stop_loss": signal["stop_loss"],
                        "take_profit": signal["take_profit"],
                        "cost": position_size * signal["entry_price"],
                        "profit_loss": 0.0  # Initially zero
                    }

                    # Call the on_trade_opened callback if set
                    if self.callbacks["on_trade_opened"]:
                        self.callbacks["on_trade_opened"](trade_data)

                    # Update console UI
                    if self.console_ui:
                        self.console_ui.update_data("trade_opened", {"trade": trade_data})

                        # Update statistics
                        self.console_ui.update_data("statistics", {
                            "statistics": {
                                "balance": self.account_balance,
                                "total_trades": len(self.trades),
                                "win_rate": self._calculate_win_rate()
                            }
                        })

                    logger.info(f"Opened {signal['direction'].name} position for {stock.symbol} at {signal['entry_price']}")
                    logger.info(f"Position size: {position_size}, Stop loss: {signal['stop_loss']}, Take profit: {signal['take_profit']}")

            except Exception as e:
                logger.error(f"Error processing data: {e}")

    def _check_exit_conditions(self, position: Position, stock: Stock):
        """
        Check if a position should be closed based on exit conditions.

        Args:
            position: Position to check
            stock: Stock data
        """
        if not position.is_open:
            return

        # Get the latest price
        latest_data = stock.data.get(self.timeframe, [])
        if not latest_data:
            logger.warning(f"No data available for {stock.symbol}")
            return

        latest_price = latest_data[-1].close

        # Check stop loss
        if position.direction == TradeDirection.LONG and latest_price <= position.stop_loss:
            self._close_position(position, latest_price, "stop_loss")
            return

        if position.direction == TradeDirection.SHORT and latest_price >= position.stop_loss:
            self._close_position(position, latest_price, "stop_loss")
            return

        # Check take profit
        if position.direction == TradeDirection.LONG and latest_price >= position.take_profit:
            self._close_position(position, latest_price, "take_profit")
            return

        if position.direction == TradeDirection.SHORT and latest_price <= position.take_profit:
            self._close_position(position, latest_price, "take_profit")
            return

    def _close_position(self, position: Position, exit_price: float, reason: str):
        """
        Close a position and update account balance.

        Args:
            position: Position to close
            exit_price: Exit price
            reason: Reason for closing the position
        """
        if not position.is_open:
            return

        # Calculate profit/loss
        if position.direction == TradeDirection.LONG:
            profit_loss = (exit_price - position.entry_price) * position.size
        else:  # SHORT
            profit_loss = (position.entry_price - exit_price) * position.size

        # Update account balance
        self.account_balance += profit_loss

        # Get current time in US Eastern Time
        import pytz
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        # Update position
        position.exit_price = exit_price
        position.exit_time = now
        position.profit_loss = profit_loss
        position.status = "closed"

        # Update corresponding trade
        for trade in self.trades:
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

        # Log trade details
        logger.info(f"Closed {position.direction.name} position for {position.symbol} at {exit_price}")
        logger.info(f"Profit/Loss: {profit_loss:.2f} ({reason})")
        logger.info(f"New account balance: {self.account_balance:.2f}")

        # Create trade data dictionary
        trade_data = {
            "id": f"trade_{len([t for t in self.trades if t.status == 'closed'])}",
            "symbol": position.symbol,
            "direction": position.direction.name,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "entry_time": position.entry_time.strftime("%Y-%m-%d %H:%M:%S") if position.entry_time else "",
            "exit_time": position.exit_time.strftime("%Y-%m-%d %H:%M:%S") if position.exit_time else "",
            "size": position.size,
            "profit_loss": profit_loss,
            "exit_reason": reason
        }

        # Call the on_trade_closed callback if set
        if self.callbacks["on_trade_closed"]:
            self.callbacks["on_trade_closed"](trade_data)

        # Update console UI
        if self.console_ui:
            # Find the trade ID in active trades
            trade_id = None
            for tid, trade in self.console_ui.active_trades.items():
                if (trade.get("symbol") == position.symbol and
                    trade.get("direction") == position.direction.name and
                    trade.get("entry_time") == position.entry_time.strftime("%Y-%m-%d %H:%M:%S") if position.entry_time else ""):
                    trade_id = tid
                    break

            if trade_id:
                trade_data["id"] = trade_id
                self.console_ui.update_data("trade_closed", {"trade": trade_data})

                # Update statistics
                self.console_ui.update_data("statistics", {
                    "statistics": {
                        "balance": self.account_balance,
                        "total_trades": len(self.trades),
                        "win_rate": self._calculate_win_rate()
                    }
                })

        # Save trade details to CSV file
        self._log_trade_to_csv(position, exit_price, profit_loss, reason)

    def _is_trading_time(self) -> bool:
        """
        Check if the current time is within trading hours.

        Returns:
            bool: True if current time is within trading hours, False otherwise
        """
        # Check if test mode is enabled
        if hasattr(self, 'test_mode') and self.test_mode:
            return True

        # Get current time in US Eastern Time
        import pytz
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        # Check if today is a trading day
        day_of_week = now.strftime("%A")
        if day_of_week not in self.trading_days:
            return False

        # Parse trading hours
        start_hour, start_minute = map(int, self.trading_start.split(":"))
        end_hour, end_minute = map(int, self.trading_end.split(":"))

        # Create datetime objects for today's trading hours
        market_open = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        market_close = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        # Check if current time is within trading hours
        return market_open <= now <= market_close

    def _reset_daily_counters(self):
        """Reset daily counters at the start of a new trading day."""
        # Get current time in US Eastern Time
        import pytz
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        day_of_week = now.strftime("%A")

        # Only reset if it's a trading day and before market open
        if day_of_week in self.trading_days:
            start_hour, start_minute = map(int, self.trading_start.split(":"))
            market_open = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)

            if now < market_open:
                logger.info("New trading day. Resetting daily counters.")
                self.current_day_trades = 0

    def save_state(self):
        """Save the current state of the paper trading system."""
        # Get current time in US Eastern Time
        import pytz
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        state = {
            "timestamp": now.isoformat(),
            "account_balance": self.account_balance,
            "positions": [pos.to_dict() for pos in self.positions],
            "trades": [trade.to_dict() for trade in self.trades],
            "current_day_trades": self.current_day_trades
        }

        try:
            with open(self.state_file, 'w') as file:
                json.dump(state, file, indent=4)
            logger.info(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def load_state(self) -> bool:
        """
        Load the state of the paper trading system.

        Returns:
            bool: True if state was loaded successfully, False otherwise
        """
        if not os.path.exists(self.state_file):
            logger.info(f"No state file found at {self.state_file}")
            return False

        try:
            with open(self.state_file, 'r') as file:
                state = json.load(file)

            self.account_balance = state.get("account_balance", self.initial_balance)

            # Load positions
            self.positions = []
            for pos_dict in state.get("positions", []):
                pos = Position.from_dict(pos_dict)
                self.positions.append(pos)

            # Load trades
            self.trades = []
            for trade_dict in state.get("trades", []):
                trade = Trade.from_dict(trade_dict)
                self.trades.append(trade)

            self.current_day_trades = state.get("current_day_trades", 0)

            logger.info(f"State loaded from {self.state_file}")
            logger.info(f"Account balance: {self.account_balance}")
            logger.info(f"Open positions: {len([p for p in self.positions if p.is_open])}")
            logger.info(f"Total trades: {len(self.trades)}")

            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False

    def _log_trade_to_csv(self, position: Position, exit_price: float, profit_loss: float, reason: str):
        """
        Log trade details to a CSV file.

        Args:
            position: Closed position
            exit_price: Exit price
            profit_loss: Profit or loss from the trade
            reason: Reason for closing the position
        """
        # Create directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)

        # CSV file path
        csv_file = "reports/trade_log.csv"

        # Check if file exists
        file_exists = os.path.isfile(csv_file)

        # Prepare trade data
        trade_data = {
            "symbol": position.symbol,
            "direction": position.direction.name,
            "entry_time": position.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time": position.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "profit_loss": profit_loss,
            "exit_reason": reason,
            "account_balance": self.account_balance
        }

        # Write to CSV
        with open(csv_file, 'a', newline='') as file:
            fieldnames = list(trade_data.keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()

            writer.writerow(trade_data)

        logger.info(f"Trade details logged to {csv_file}")

    def _calculate_win_rate(self) -> float:
        """Calculate the win rate for closed trades."""
        closed_trades = [t for t in self.trades if t.status == "closed"]
        winning_trades = [t for t in closed_trades if t.profit_loss and t.profit_loss > 0]

        return len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0

    def _calculate_profit_loss(self, trade: Dict) -> float:
        """Calculate profit/loss for a trade based on current price."""
        direction = trade.get("direction")
        entry_price = trade.get("entry_price", 0.0)
        current_price = trade.get("current_price", entry_price)
        size = trade.get("size", 0.0)

        if direction == "LONG":
            return (current_price - entry_price) * size
        elif direction == "SHORT":
            return (entry_price - current_price) * size
        else:
            return 0.0

    def _generate_performance_report(self):
        """Generate a performance report for the paper trading session."""
        if not self.trades:
            logger.info("No trades to report.")
            return

        # Calculate performance metrics
        total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if t.status == "closed"]
        winning_trades = [t for t in closed_trades if t.profit_loss and t.profit_loss > 0]
        losing_trades = [t for t in closed_trades if t.profit_loss and t.profit_loss <= 0]

        win_rate = self._calculate_win_rate()

        total_profit = sum(t.profit_loss for t in winning_trades) if winning_trades else 0
        total_loss = sum(t.profit_loss for t in losing_trades) if losing_trades else 0

        profit_factor = abs(total_profit / total_loss) if total_loss else float('inf')

        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0

        total_return = self.account_balance - self.initial_balance
        total_return_percent = (total_return / self.initial_balance) * 100

        # Print report
        logger.info("=== Performance Report ===")
        logger.info(f"Initial Balance: {self.initial_balance:.2f}")
        logger.info(f"Final Balance: {self.account_balance:.2f}")
        logger.info(f"Total Return: {total_return:.2f} ({total_return_percent:.2f}%)")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Closed Trades: {len(closed_trades)}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Average Win: {avg_win:.2f}")
        logger.info(f"Average Loss: {avg_loss:.2f}")
        logger.info("==========================")

        # Get current time in US Eastern Time
        import pytz
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        # Save report to file
        report_file = f"reports/paper_trading_report_{now.strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        report = {
            "initial_balance": self.initial_balance,
            "final_balance": self.account_balance,
            "total_return": total_return,
            "total_return_percent": total_return_percent,
            "total_trades": total_trades,
            "closed_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "trades": [t.to_dict() for t in self.trades]
        }

        with open(report_file, 'w') as file:
            json.dump(report, file, indent=4)

        # Generate trade log CSV if it doesn't exist
        csv_file = "reports/trade_log.csv"
        if not os.path.isfile(csv_file) and closed_trades:
            for trade in closed_trades:
                self._log_trade_to_csv(
                    Position(
                        symbol=trade.symbol,
                        direction=trade.direction,
                        entry_price=trade.entry_price,
                        stop_loss=trade.stop_loss,
                        take_profit=trade.take_profit,
                        size=trade.size,
                        entry_time=trade.entry_time,
                        exit_price=trade.exit_price,
                        exit_time=trade.exit_time,
                        profit_loss=trade.profit_loss
                    ),
                    trade.exit_price,
                    trade.profit_loss,
                    trade.exit_reason
                )

        # Generate heatmap of trades
        self._generate_trade_heatmap()

        logger.info(f"Performance report saved to {report_file}")

    def _generate_trade_heatmap(self):
        """
        Generate a heatmap of trades showing profit/loss by symbol and time.
        """
        # Check if we have closed trades
        closed_trades = [t for t in self.trades if t.status == "closed"]
        if not closed_trades:
            logger.info("No closed trades to generate heatmap")
            return

        try:
            # Create a DataFrame from trades
            trades_df = pd.DataFrame([
                {
                    "symbol": t.symbol,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "profit_loss": t.profit_loss,
                    "direction": t.direction.name
                }
                for t in closed_trades
            ])

            # Convert times to datetime if they're strings
            if isinstance(trades_df["entry_time"].iloc[0], str):
                trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
            if isinstance(trades_df["exit_time"].iloc[0], str):
                trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])

            # Add date and hour columns
            trades_df["date"] = trades_df["entry_time"].dt.date
            trades_df["hour"] = trades_df["entry_time"].dt.hour

            # Group by symbol and hour, and calculate sum of profit/loss
            heatmap_data = trades_df.groupby(["symbol", "hour"])["profit_loss"].sum().reset_index()

            # Pivot the data for the heatmap
            pivot_data = heatmap_data.pivot(index="symbol", columns="hour", values="profit_loss")

            # Create directory if it doesn't exist
            os.makedirs("reports", exist_ok=True)

            # Save to CSV
            pivot_data.to_csv("reports/trade_heatmap.csv")

            # Create a heatmap using matplotlib and seaborn
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_data, cmap="RdYlGn", center=0, annot=True, fmt=".2f")
            plt.title("Trade Profit/Loss Heatmap by Symbol and Hour")
            plt.xlabel("Hour of Day")
            plt.ylabel("Symbol")

            # Get current time in US Eastern Time
            import pytz
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)

            # Save the heatmap
            heatmap_file = f"reports/trade_heatmap_{now.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(heatmap_file)
            plt.close()

            logger.info(f"Trade heatmap saved to {heatmap_file}")

        except Exception as e:
            logger.error(f"Error generating trade heatmap: {e}")
