#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data models for the day trading algorithm.

This module defines the data structures used throughout the algorithm.
"""

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from datetime import datetime


class TimeFrame(enum.Enum):
    """Enum representing different timeframes for analysis."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    
    @classmethod
    def from_string(cls, value: str) -> 'TimeFrame':
        """Convert string to TimeFrame enum."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown timeframe: {value}")


class TradeDirection(enum.Enum):
    """Enum representing trade direction (long or short)."""
    LONG = "long"
    SHORT = "short"


@dataclass
class OHLCV:
    """Class representing OHLCV (Open, High, Low, Close, Volume) data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OHLCV':
        """Create from dictionary."""
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            timestamp=timestamp,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"]
        )


@dataclass
class Stock:
    """Class representing a stock with its data."""
    symbol: str
    name: Optional[str] = None
    data: Dict[TimeFrame, List[OHLCV]] = field(default_factory=dict)
    
    def add_data(self, timeframe: TimeFrame, data: List[OHLCV]):
        """Add OHLCV data for a specific timeframe."""
        self.data[timeframe] = data
    
    def get_latest_price(self, timeframe: TimeFrame) -> Optional[float]:
        """Get the latest price for a specific timeframe."""
        if timeframe not in self.data or not self.data[timeframe]:
            return None
        
        return self.data[timeframe][-1].close
    
    def get_dataframe(self, timeframe: TimeFrame) -> 'pd.DataFrame':
        """Convert data to pandas DataFrame."""
        import pandas as pd
        
        if timeframe not in self.data or not self.data[timeframe]:
            return pd.DataFrame()
        
        data = self.data[timeframe]
        df = pd.DataFrame([{
            "timestamp": point.timestamp,
            "open": point.open,
            "high": point.high,
            "low": point.low,
            "close": point.close,
            "volume": point.volume
        } for point in data])
        
        if not df.empty:
            df.set_index("timestamp", inplace=True)
        
        return df


@dataclass
class Trade:
    """Class representing a trade."""
    symbol: str
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float
    take_profit: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    profit_loss: Optional[float] = None
    status: str = "open"  # open, closed
    reason: str = ""
    exit_reason: Optional[str] = None
    
    def close(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the trade."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.status = "closed"
        
        # Calculate profit/loss
        if self.direction == TradeDirection.LONG:
            self.profit_loss = (exit_price - self.entry_price) * self.size
        else:  # SHORT
            self.profit_loss = (self.entry_price - exit_price) * self.size
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "size": self.size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "profit_loss": self.profit_loss,
            "status": self.status,
            "reason": self.reason,
            "exit_reason": self.exit_reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trade':
        """Create from dictionary."""
        entry_time = data["entry_time"]
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        
        exit_time = data.get("exit_time")
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)
        
        direction = data["direction"]
        if isinstance(direction, str):
            direction = TradeDirection(direction)
        
        return cls(
            symbol=data["symbol"],
            direction=direction,
            entry_price=data["entry_price"],
            entry_time=entry_time,
            size=data["size"],
            stop_loss=data["stop_loss"],
            take_profit=data["take_profit"],
            exit_price=data.get("exit_price"),
            exit_time=exit_time,
            profit_loss=data.get("profit_loss"),
            status=data.get("status", "open"),
            reason=data.get("reason", ""),
            exit_reason=data.get("exit_reason")
        )


@dataclass
class Position:
    """Class representing a position in a stock."""
    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    entry_time: datetime
    status: str = "open"  # open, closed
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    profit_loss: Optional[float] = None
    
    @property
    def is_open(self) -> bool:
        """Check if the position is open."""
        return self.status == "open"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "size": self.size,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "status": self.status,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "profit_loss": self.profit_loss
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create from dictionary."""
        entry_time = data["entry_time"]
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        
        exit_time = data.get("exit_time")
        if isinstance(exit_time, str) and exit_time:
            exit_time = datetime.fromisoformat(exit_time)
        
        direction = data["direction"]
        if isinstance(direction, str):
            direction = TradeDirection(direction)
        
        return cls(
            symbol=data["symbol"],
            direction=direction,
            entry_price=data["entry_price"],
            stop_loss=data["stop_loss"],
            take_profit=data["take_profit"],
            size=data["size"],
            entry_time=entry_time,
            status=data.get("status", "open"),
            exit_price=data.get("exit_price"),
            exit_time=exit_time,
            profit_loss=data.get("profit_loss")
        )
