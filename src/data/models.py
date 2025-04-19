#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core data models for the Swing Trading Algorithm.

This module defines the fundamental data structures used throughout the system,
including representations of stocks, price data, timeframes, and trades.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


class TimeFrame(Enum):
    """Enumeration of supported timeframes for analysis."""
    DAILY = "daily"
    FOUR_HOUR = "4h"
    ONE_HOUR = "1h"

    @classmethod
    def from_string(cls, value: str) -> "TimeFrame":
        """Convert string representation to TimeFrame enum."""
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Invalid timeframe: {value}")


@dataclass
class OHLCV:
    """Open-High-Low-Close-Volume data for a specific time period."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self):
        """Validate the OHLCV data."""
        # Fix any data inconsistencies
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError("Price values must be positive")

        # Fix high/low inconsistencies
        if self.high < self.low:
            # Swap high and low
            self.high, self.low = self.low, self.high

        # Fix open price inconsistencies
        if self.open < self.low:
            self.open = self.low
        elif self.open > self.high:
            self.open = self.high

        # Fix close price inconsistencies
        if self.close < self.low:
            self.close = self.low
        elif self.close > self.high:
            self.close = self.high

        # Fix volume inconsistencies
        if self.volume < 0:
            self.volume = 0


@dataclass
class Stock:
    """Representation of a stock with its associated price data."""
    symbol: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    data: Dict[TimeFrame, List[OHLCV]] = field(default_factory=dict)

    def copy(self) -> 'Stock':
        """Create a deep copy of the stock."""
        new_stock = Stock(
            symbol=self.symbol,
            name=self.name,
            exchange=self.exchange,
            sector=self.sector,
            industry=self.industry
        )

        # Copy data points
        for timeframe, data_points in self.data.items():
            new_stock.data[timeframe] = []
            for point in data_points:
                new_point = OHLCV(
                    timestamp=point.timestamp,
                    open=point.open,
                    high=point.high,
                    low=point.low,
                    close=point.close,
                    volume=point.volume
                )
                new_stock.data[timeframe].append(new_point)

        return new_stock

    def add_data_point(self, timeframe: TimeFrame, data_point: OHLCV) -> None:
        """Add a single OHLCV data point for a specific timeframe."""
        if timeframe not in self.data:
            self.data[timeframe] = []
        self.data[timeframe].append(data_point)
        # Sort by timestamp to ensure chronological order
        self.data[timeframe].sort(key=lambda x: x.timestamp)

    def get_dataframe(self, timeframe: TimeFrame) -> pd.DataFrame:
        """Convert OHLCV data for a specific timeframe to a pandas DataFrame."""
        if timeframe not in self.data or not self.data[timeframe]:
            return pd.DataFrame()

        data = {
            'timestamp': [d.timestamp for d in self.data[timeframe]],
            'open': [d.open for d in self.data[timeframe]],
            'high': [d.high for d in self.data[timeframe]],
            'low': [d.low for d in self.data[timeframe]],
            'close': [d.close for d in self.data[timeframe]],
            'volume': [d.volume for d in self.data[timeframe]]
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df


class TradeDirection(Enum):
    """Enumeration of possible trade directions."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Representation of a single trade."""
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    position_size: float  # Number of shares
    stop_loss: float
    take_profit: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    _profit_loss: Optional[float] = None
    _profit_loss_percent: Optional[float] = None

    @property
    def is_open(self) -> bool:
        """Check if the trade is still open."""
        return self.exit_time is None

    @property
    def duration(self) -> Optional[float]:
        """Calculate the duration of the trade in days."""
        if not self.exit_time:
            return None

        # Convert to timezone-naive datetime objects if they have timezones
        entry_time = self.entry_time
        exit_time = self.exit_time

        if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo is not None:
            entry_time = entry_time.replace(tzinfo=None)

        if hasattr(exit_time, 'tzinfo') and exit_time.tzinfo is not None:
            exit_time = exit_time.replace(tzinfo=None)

        return (exit_time - entry_time).total_seconds() / 86400  # Convert seconds to days

    @property
    def profit_loss(self) -> Optional[float]:
        """Get the profit or loss from the trade."""
        if self._profit_loss is not None:
            return self._profit_loss

        if not self.exit_price:
            return None

        if self.direction == TradeDirection.LONG:
            return (self.exit_price - self.entry_price) * self.position_size
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.position_size

    @profit_loss.setter
    def profit_loss(self, value: float) -> None:
        """Set the profit or loss for the trade."""
        self._profit_loss = value

    @property
    def profit_loss_percent(self) -> Optional[float]:
        """Get the profit or loss as a percentage of the investment."""
        if self._profit_loss_percent is not None:
            return self._profit_loss_percent

        if not self.exit_price:
            return None

        if self.direction == TradeDirection.LONG:
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100

    @profit_loss_percent.setter
    def profit_loss_percent(self, value: float) -> None:
        """Set the profit or loss percentage for the trade."""
        self._profit_loss_percent = value


@dataclass
class Account:
    """Representation of a trading account."""
    initial_balance: float
    currency: str
    current_balance: float = field(init=False)
    trades: List[Trade] = field(default_factory=list)
    open_positions: List[Trade] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the current balance to the initial balance."""
        self.current_balance = self.initial_balance

    def open_trade(self, trade: Trade) -> None:
        """Open a new trade and update the account."""
        # Calculate the cost of the trade
        cost = trade.entry_price * trade.position_size

        # Ensure we have enough balance
        if cost > self.current_balance:
            raise ValueError(f"Insufficient balance to open trade: {cost} > {self.current_balance}")

        # Add the trade to our lists
        self.trades.append(trade)
        self.open_positions.append(trade)

    def close_trade(self, trade: Trade, exit_price: float, exit_time: datetime, exit_reason: str) -> None:
        """Close an existing trade and update the account."""
        if trade not in self.open_positions:
            raise ValueError("Trade is not in open positions")

        # Update the trade with exit information
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_reason = exit_reason

        # Update the account balance
        self.current_balance += trade.profit_loss

        # Remove from open positions
        self.open_positions.remove(trade)

    @property
    def equity(self) -> float:
        """Calculate the current equity including unrealized gains/losses."""
        # Start with the current balance
        total = self.current_balance

        # Add unrealized profits/losses from open positions
        for trade in self.open_positions:
            # This is a simplified calculation and would need to be updated with current market prices
            # For now, we'll just use the current balance
            pass

        return total

    @property
    def profit_loss(self) -> float:
        """Calculate the overall profit or loss."""
        return self.current_balance - self.initial_balance

    @property
    def profit_loss_percent(self) -> float:
        """Calculate the overall profit or loss as a percentage."""
        return (self.profit_loss / self.initial_balance) * 100
