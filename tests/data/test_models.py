#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for data models.
"""

import pytest
from datetime import datetime
import pandas as pd
from src.data.models import OHLCV, Stock, TimeFrame, TradeDirection, Trade, Account


class TestTimeFrame:
    """Test suite for TimeFrame enum."""
    
    def test_from_string(self):
        """Test conversion from string to TimeFrame enum."""
        assert TimeFrame.from_string("daily") == TimeFrame.DAILY
        assert TimeFrame.from_string("4h") == TimeFrame.FOUR_HOUR
        assert TimeFrame.from_string("1h") == TimeFrame.ONE_HOUR
        
        # Test case insensitivity
        assert TimeFrame.from_string("DAILY") == TimeFrame.DAILY
        
        # Test invalid value
        with pytest.raises(ValueError):
            TimeFrame.from_string("invalid")


class TestOHLCV:
    """Test suite for OHLCV data class."""
    
    def test_valid_ohlcv(self):
        """Test creating a valid OHLCV object."""
        timestamp = datetime.now()
        ohlcv = OHLCV(
            timestamp=timestamp,
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000
        )
        
        assert ohlcv.timestamp == timestamp
        assert ohlcv.open == 100.0
        assert ohlcv.high == 110.0
        assert ohlcv.low == 95.0
        assert ohlcv.close == 105.0
        assert ohlcv.volume == 1000
    
    def test_invalid_ohlcv(self):
        """Test creating invalid OHLCV objects."""
        timestamp = datetime.now()
        
        # Test negative price
        with pytest.raises(ValueError):
            OHLCV(timestamp=timestamp, open=-100.0, high=110.0, low=95.0, close=105.0, volume=1000)
        
        # Test high < low
        with pytest.raises(ValueError):
            OHLCV(timestamp=timestamp, open=100.0, high=90.0, low=95.0, close=105.0, volume=1000)
        
        # Test open outside range
        with pytest.raises(ValueError):
            OHLCV(timestamp=timestamp, open=120.0, high=110.0, low=95.0, close=105.0, volume=1000)
        
        # Test close outside range
        with pytest.raises(ValueError):
            OHLCV(timestamp=timestamp, open=100.0, high=110.0, low=95.0, close=90.0, volume=1000)
        
        # Test negative volume
        with pytest.raises(ValueError):
            OHLCV(timestamp=timestamp, open=100.0, high=110.0, low=95.0, close=105.0, volume=-1000)


class TestStock:
    """Test suite for Stock class."""
    
    def test_add_data_point(self):
        """Test adding data points to a stock."""
        stock = Stock(symbol="AAPL")
        
        # Create some data points
        timestamp1 = datetime(2023, 1, 1)
        ohlcv1 = OHLCV(timestamp=timestamp1, open=100.0, high=110.0, low=95.0, close=105.0, volume=1000)
        
        timestamp2 = datetime(2023, 1, 2)
        ohlcv2 = OHLCV(timestamp=timestamp2, open=105.0, high=115.0, low=100.0, close=110.0, volume=1200)
        
        # Add data points
        stock.add_data_point(TimeFrame.DAILY, ohlcv1)
        stock.add_data_point(TimeFrame.DAILY, ohlcv2)
        
        # Check that data points were added
        assert len(stock.data[TimeFrame.DAILY]) == 2
        assert stock.data[TimeFrame.DAILY][0] == ohlcv1
        assert stock.data[TimeFrame.DAILY][1] == ohlcv2
    
    def test_get_dataframe(self):
        """Test converting stock data to a DataFrame."""
        stock = Stock(symbol="AAPL")
        
        # Create some data points
        timestamp1 = datetime(2023, 1, 1)
        ohlcv1 = OHLCV(timestamp=timestamp1, open=100.0, high=110.0, low=95.0, close=105.0, volume=1000)
        
        timestamp2 = datetime(2023, 1, 2)
        ohlcv2 = OHLCV(timestamp=timestamp2, open=105.0, high=115.0, low=100.0, close=110.0, volume=1200)
        
        # Add data points
        stock.add_data_point(TimeFrame.DAILY, ohlcv1)
        stock.add_data_point(TimeFrame.DAILY, ohlcv2)
        
        # Get DataFrame
        df = stock.get_dataframe(TimeFrame.DAILY)
        
        # Check DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.index[0] == timestamp1
        assert df.index[1] == timestamp2
        assert df["open"][0] == 100.0
        assert df["high"][0] == 110.0
        assert df["low"][0] == 95.0
        assert df["close"][0] == 105.0
        assert df["volume"][0] == 1000


class TestTrade:
    """Test suite for Trade class."""
    
    def test_trade_properties(self):
        """Test trade properties."""
        # Create a trade
        entry_time = datetime(2023, 1, 1)
        trade = Trade(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_time=entry_time,
            entry_price=100.0,
            position_size=10,
            stop_loss=95.0,
            take_profit=110.0
        )
        
        # Check initial state
        assert trade.is_open
        assert trade.duration is None
        assert trade.profit_loss is None
        assert trade.profit_loss_percent is None
        
        # Close the trade
        exit_time = datetime(2023, 1, 2)
        trade.exit_time = exit_time
        trade.exit_price = 105.0
        trade.exit_reason = "Take profit hit"
        
        # Check closed state
        assert not trade.is_open
        assert trade.duration == 1.0  # 1 day
        assert trade.profit_loss == 50.0  # (105 - 100) * 10
        assert trade.profit_loss_percent == 5.0  # ((105 - 100) / 100) * 100


class TestAccount:
    """Test suite for Account class."""
    
    def test_account_operations(self):
        """Test account operations."""
        # Create an account
        account = Account(initial_balance=100.0, currency="GBP")
        
        # Check initial state
        assert account.current_balance == 100.0
        assert account.profit_loss == 0.0
        assert account.profit_loss_percent == 0.0
        assert len(account.trades) == 0
        assert len(account.open_positions) == 0
        
        # Create a trade
        entry_time = datetime(2023, 1, 1)
        trade = Trade(
            symbol="AAPL",
            direction=TradeDirection.LONG,
            entry_time=entry_time,
            entry_price=10.0,
            position_size=5,
            stop_loss=9.0,
            take_profit=12.0
        )
        
        # Open the trade
        account.open_trade(trade)
        
        # Check state after opening trade
        assert len(account.trades) == 1
        assert len(account.open_positions) == 1
        assert account.open_positions[0] == trade
        
        # Close the trade
        exit_time = datetime(2023, 1, 2)
        account.close_trade(trade, 11.0, exit_time, "Take profit hit")
        
        # Check state after closing trade
        assert len(account.trades) == 1
        assert len(account.open_positions) == 0
        assert trade.exit_price == 11.0
        assert trade.exit_time == exit_time
        assert trade.exit_reason == "Take profit hit"
        assert account.current_balance == 105.0  # 100 + (11 - 10) * 5
        assert account.profit_loss == 5.0
        assert account.profit_loss_percent == 5.0
        
        # Test insufficient balance
        expensive_trade = Trade(
            symbol="GOOGL",
            direction=TradeDirection.LONG,
            entry_time=entry_time,
            entry_price=1000.0,
            position_size=1,
            stop_loss=950.0,
            take_profit=1100.0
        )
        
        with pytest.raises(ValueError):
            account.open_trade(expensive_trade)
        
        # Test closing non-existent trade
        non_existent_trade = Trade(
            symbol="MSFT",
            direction=TradeDirection.LONG,
            entry_time=entry_time,
            entry_price=10.0,
            position_size=5,
            stop_loss=9.0,
            take_profit=12.0
        )
        
        with pytest.raises(ValueError):
            account.close_trade(non_existent_trade, 11.0, exit_time, "Take profit hit")
