#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtesting module.

This module provides functionality to backtest trading strategies
on historical data to evaluate their performance.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

from src.data.models import Stock, TimeFrame, Trade, TradeDirection, OHLCV
from src.strategy.basic_swing import BasicSwingStrategy
from src.risk.position_sizing import calculate_position_size
from src.performance.analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtests trading strategies on historical data.
    """

    def __init__(self, config: Dict):
        """
        Initialize the backtester with configuration parameters.

        Args:
            config: Backtester configuration dictionary
        """
        self.config = config

        # Extract configuration parameters
        self.initial_balance = config.get("initial_balance", 1000.0)
        self.currency = config.get("currency", "GBP")
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        self.max_open_positions = config.get("max_open_positions", 3)
        self.slippage = config.get("slippage", 0.001)
        self.commission = config.get("commission", 0.002)
        self.enable_fractional_shares = config.get("enable_fractional_shares", True)

        logger.info(f"Initialized Backtester with initial balance: {self.initial_balance} {self.currency}")

    def run_backtest(self, stocks: Dict[str, Stock], strategy: Any, timeframe: TimeFrame,
                   start_date: datetime, end_date: datetime) -> Dict:
        """
        Run a backtest on historical data.

        Args:
            stocks: Dictionary of Stock objects by symbol
            strategy: Trading strategy object
            timeframe: Timeframe for analysis
            start_date: Start date for the backtest
            end_date: End date for the backtest

        Returns:
            Dict: Backtest results
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")

        # Initialize account
        balance = self.initial_balance
        open_positions = {}  # symbol -> Trade
        closed_trades = []
        equity_curve = []

        # Create a combined DataFrame with all dates
        all_dates = set()
        for symbol, stock in stocks.items():
            df = stock.get_dataframe(timeframe)
            if not df.empty:
                all_dates.update(df.index)

        all_dates = sorted(all_dates)

        # Filter dates within the backtest period
        backtest_dates = []
        for date in all_dates:
            # Convert to timezone-naive datetime if it has timezone
            if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                date_naive = date.replace(tzinfo=None)
            else:
                date_naive = date

            if start_date <= date_naive <= end_date:
                backtest_dates.append(date)

        if not backtest_dates:
            logger.error("No dates found within the backtest period")
            return {"success": False, "error": "No dates found within the backtest period"}

        # Initialize equity curve with initial balance
        equity_curve.append({
            "date": backtest_dates[0],
            "equity": balance,
            "open_positions": 0
        })

        # Run the backtest
        for current_date in backtest_dates:
            logger.debug(f"Processing date: {current_date}")

            # Update open positions
            closed_today = self._update_positions(open_positions, stocks, current_date)
            closed_trades.extend(closed_today)

            # Update balance
            for trade in closed_today:
                # Calculate commission
                commission_amount = trade.exit_price * trade.position_size * self.commission

                # Update balance
                balance += trade.profit_loss - commission_amount

            # Generate signals for each stock
            for symbol, stock in stocks.items():
                # Skip if we already have an open position for this symbol
                if symbol in open_positions:
                    continue

                # Skip if we have reached the maximum number of open positions
                if len(open_positions) >= self.max_open_positions:
                    break

                # Get the data up to the current date
                df = stock.get_dataframe(timeframe)
                if df.empty or current_date not in df.index:
                    continue

                # Get the data up to the current date
                current_idx = df.index.get_loc(current_date)
                df_subset = df.iloc[:current_idx+1]

                # Create a temporary Stock object with data up to the current date
                temp_stock = Stock(symbol=symbol)
                for i, row in df_subset.iterrows():
                    try:
                        ohlcv = OHLCV(
                            timestamp=i,
                            open=row["open"],
                            high=row["high"],
                            low=row["low"],
                            close=row["close"],
                            volume=row["volume"]
                        )
                        temp_stock.add_data_point(timeframe, ohlcv)
                    except Exception as e:
                        logger.warning(f"Error creating OHLCV object: {e}")

                # Skip if there's not enough data
                if len(temp_stock.data.get(timeframe, [])) < max(strategy.fast_ma_period, strategy.slow_ma_period, strategy.trend_ma_period, strategy.rsi_period):
                    continue

                try:
                    # Analyze the stock
                    results = strategy.analyze(temp_stock, timeframe)

                    if not results or "signals" not in results or not results["signals"]:
                        continue
                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {e}")
                    continue

                # Get the most recent signal
                signal = results["signals"][-1]

                # Check if the signal is from the current date
                if signal["timestamp"] != current_date:
                    continue

                # Process the signal
                trade = self._process_signal(signal, stock, current_date, balance)

                if trade:
                    # Calculate commission
                    commission_amount = trade.entry_price * trade.position_size * self.commission

                    # Update balance
                    balance -= commission_amount

                    # Add to open positions
                    open_positions[symbol] = trade

                    logger.debug(f"Opened {trade.direction.value} trade for {symbol} at {trade.entry_price}")

            # Calculate current equity
            current_equity = balance
            for symbol, trade in open_positions.items():
                # Get the current price
                df = stocks[symbol].get_dataframe(timeframe)
                if current_date in df.index:
                    current_price = df.loc[current_date, "close"]

                    # Calculate unrealized profit/loss
                    if trade.direction == TradeDirection.LONG:
                        unrealized_pl = (current_price - trade.entry_price) * trade.position_size
                    else:  # SHORT
                        unrealized_pl = (trade.entry_price - current_price) * trade.position_size

                    current_equity += unrealized_pl

            # Add to equity curve
            equity_curve.append({
                "date": current_date,
                "equity": current_equity,
                "open_positions": len(open_positions)
            })

        # Close any remaining open positions at the end of the backtest
        for symbol, trade in list(open_positions.items()):
            # Get the last price
            df = stocks[symbol].get_dataframe(timeframe)
            if df.empty:
                continue

            last_price = df.iloc[-1]["close"]

            # Apply slippage
            if trade.direction == TradeDirection.LONG:
                adjusted_exit_price = last_price * (1 - self.slippage)
            else:  # SHORT
                adjusted_exit_price = last_price * (1 + self.slippage)

            # Close the trade
            trade.exit_price = adjusted_exit_price
            trade.exit_time = end_date
            trade.exit_reason = "End of backtest"

            # Calculate profit/loss
            if trade.direction == TradeDirection.LONG:
                trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.position_size
            else:  # SHORT
                trade.profit_loss = (trade.entry_price - trade.exit_price) * trade.position_size

            # Calculate profit/loss percent
            if trade.direction == TradeDirection.LONG:
                trade.profit_loss_percent = (trade.exit_price / trade.entry_price - 1) * 100
            else:  # SHORT
                trade.profit_loss_percent = (trade.entry_price / trade.exit_price - 1) * 100

            # Calculate commission
            commission_amount = trade.exit_price * trade.position_size * self.commission

            # Update balance
            balance += trade.profit_loss - commission_amount

            # Add to closed trades
            closed_trades.append(trade)

            # Remove from open positions
            del open_positions[symbol]

            logger.debug(f"Closed {trade.direction.value} trade for {symbol} at {trade.exit_price} (End of backtest)")

        # Calculate final equity
        final_equity = balance

        # Add final equity to equity curve
        equity_curve.append({
            "date": end_date,
            "equity": final_equity,
            "open_positions": 0
        })

        # Calculate performance metrics
        analyzer = PerformanceAnalyzer({"metrics": ["all"]})

        # Convert trades to the format expected by the analyzer
        trade_history = []
        for trade in closed_trades:
            trade_dict = {
                "symbol": trade.symbol,
                "direction": trade.direction.value,
                "entry_time": trade.entry_time,
                "entry_price": trade.entry_price,
                "position_size": trade.position_size,
                "stop_loss": trade.stop_loss,
                "take_profit": trade.take_profit,
                "exit_time": trade.exit_time,
                "exit_price": trade.exit_price,
                "exit_reason": trade.exit_reason,
                "profit_loss": trade.profit_loss,
                "profit_loss_percent": trade.profit_loss_percent,
                "duration": trade.duration,
                "is_open": False
            }
            trade_history.append(trade_dict)

        metrics = analyzer.analyze_trades(trade_history, self.initial_balance)

        # Create equity curve DataFrame
        equity_curve_df = pd.DataFrame(equity_curve)

        logger.info(f"Backtest completed with final balance: {balance} {self.currency}")
        logger.info(f"Total trades: {len(closed_trades)}")

        # Calculate total return percent
        total_return_percent = (balance / self.initial_balance - 1) * 100
        logger.info(f"Total return: {total_return_percent:.2f}%")

        # Calculate basic metrics if not available
        winning_trades = sum(1 for t in closed_trades if t.profit_loss > 0)
        losing_trades = len(closed_trades) - winning_trades
        win_rate = winning_trades / len(closed_trades) * 100 if closed_trades else 0

        # Get metrics from analyzer or use calculated values
        return {
            "success": True,
            "initial_balance": self.initial_balance,
            "final_balance": balance,
            "total_return": balance - self.initial_balance,
            "total_return_percent": (balance / self.initial_balance - 1) * 100,
            "total_trades": len(closed_trades),
            "winning_trades": metrics.get("winning_trades", winning_trades),
            "losing_trades": metrics.get("losing_trades", losing_trades),
            "win_rate": metrics.get("win_rate", win_rate),
            "profit_factor": metrics.get("profit_factor", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "max_drawdown_percent": metrics.get("max_drawdown_percent", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "sortino_ratio": metrics.get("sortino_ratio", 0),
            "trades": trade_history,
            "equity_curve": equity_curve_df
        }

    def _process_signal(self, signal: Dict, stock: Stock, current_date: datetime, available_balance: float) -> Optional[Trade]:
        """
        Process a trading signal and create a trade if appropriate.

        Args:
            signal: Signal dictionary
            stock: Stock object
            current_date: Current date
            available_balance: Available balance

        Returns:
            Optional[Trade]: Created trade or None if no trade was created
        """
        # Apply slippage to entry price
        direction = signal["direction"]
        entry_price = signal["entry_price"]

        if direction == TradeDirection.LONG:
            # For long positions, slippage increases the entry price
            adjusted_entry_price = entry_price * (1 + self.slippage)
        else:  # SHORT
            # For short positions, slippage decreases the entry price
            adjusted_entry_price = entry_price * (1 - self.slippage)

        # Calculate position size
        position_size = calculate_position_size(
            account_balance=available_balance,
            risk_per_trade=self.risk_per_trade,
            entry_price=adjusted_entry_price,
            stop_loss=signal["stop_loss"],
            allow_fractional_shares=self.enable_fractional_shares
        )

        # Check if position size is valid
        if position_size <= 0:
            return None

        # Calculate commission
        commission_amount = adjusted_entry_price * position_size * self.commission

        # Check if we have enough balance for the trade including commission
        trade_cost = adjusted_entry_price * position_size + commission_amount
        if trade_cost > available_balance:
            return None

        # Create trade object
        trade = Trade(
            symbol=signal["symbol"],
            direction=direction,
            entry_time=current_date,
            entry_price=adjusted_entry_price,
            position_size=position_size,
            stop_loss=signal["stop_loss"],
            take_profit=signal["take_profit"]
        )

        return trade

    def _update_positions(self, open_positions: Dict[str, Trade], stocks: Dict[str, Stock], current_date: datetime) -> List[Trade]:
        """
        Update open positions based on current prices and check for exits.

        Args:
            open_positions: Dictionary of open positions by symbol
            stocks: Dictionary of Stock objects by symbol
            current_date: Current date

        Returns:
            List[Trade]: List of closed trades
        """
        closed_trades = []

        # Make a copy of the keys to avoid modifying the dictionary during iteration
        symbols = list(open_positions.keys())

        for symbol in symbols:
            if symbol not in stocks:
                continue

            stock = stocks[symbol]
            trade = open_positions[symbol]

            # Get the price data for the current date
            df = stock.get_dataframe(TimeFrame.DAILY)
            if current_date not in df.index:
                continue

            # Get the prices for the current date
            current_data = df.loc[current_date]
            current_price = current_data["close"]
            high_price = current_data["high"]
            low_price = current_data["low"]

            # Check for stop loss hit
            stop_loss_hit = False
            if trade.direction == TradeDirection.LONG and low_price <= trade.stop_loss:
                stop_loss_hit = True
                exit_price = max(trade.stop_loss, low_price)  # Use stop loss or low, whichever is higher
                exit_reason = "Stop loss hit"
            elif trade.direction == TradeDirection.SHORT and high_price >= trade.stop_loss:
                stop_loss_hit = True
                exit_price = min(trade.stop_loss, high_price)  # Use stop loss or high, whichever is lower
                exit_reason = "Stop loss hit"

            # Check for take profit hit
            take_profit_hit = False
            if trade.direction == TradeDirection.LONG and high_price >= trade.take_profit:
                take_profit_hit = True
                exit_price = min(trade.take_profit, high_price)  # Use take profit or high, whichever is lower
                exit_reason = "Take profit hit"
            elif trade.direction == TradeDirection.SHORT and low_price <= trade.take_profit:
                take_profit_hit = True
                exit_price = max(trade.take_profit, low_price)  # Use take profit or low, whichever is higher
                exit_reason = "Take profit hit"

            # Close the trade if stop loss or take profit was hit
            if stop_loss_hit or take_profit_hit:
                # Apply slippage to exit price
                if trade.direction == TradeDirection.LONG:
                    # For long positions, slippage decreases the exit price
                    adjusted_exit_price = exit_price * (1 - self.slippage)
                else:  # SHORT
                    # For short positions, slippage increases the exit price
                    adjusted_exit_price = exit_price * (1 + self.slippage)

                # Update trade with exit information
                trade.exit_price = adjusted_exit_price
                trade.exit_time = current_date
                trade.exit_reason = exit_reason

                # Calculate profit/loss
                if trade.direction == TradeDirection.LONG:
                    trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.position_size
                else:  # SHORT
                    trade.profit_loss = (trade.entry_price - trade.exit_price) * trade.position_size

                # Calculate profit/loss percent
                if trade.direction == TradeDirection.LONG:
                    trade.profit_loss_percent = (trade.exit_price / trade.entry_price - 1) * 100
                else:  # SHORT
                    trade.profit_loss_percent = (trade.entry_price / trade.exit_price - 1) * 100

                # Add to closed trades
                closed_trades.append(trade)

                # Remove from open positions
                del open_positions[symbol]

        return closed_trades
