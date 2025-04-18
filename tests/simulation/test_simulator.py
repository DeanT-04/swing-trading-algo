#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the trade simulator.
"""

import pytest
from datetime import datetime, timedelta

from src.data.models import OHLCV, Stock, TimeFrame, TradeDirection
from src.simulation.simulator import TradeSimulator


class TestTradeSimulator:
    """Test suite for TradeSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create a trade simulator instance."""
        config = {
            "initial_balance": 1000.0,
            "currency": "GBP",
            "risk_per_trade": 0.02,
            "max_open_positions": 3,
            "slippage": 0.001,
            "commission": 0.002,
            "enable_fractional_shares": True
        }
        return TradeSimulator(config)

    @pytest.fixture
    def stock(self):
        """Create a stock with sample data."""
        stock = Stock(symbol="AAPL")

        # Create 10 days of sample data
        base_date = datetime(2023, 1, 1)

        for i in range(10):
            date = base_date + timedelta(days=i)

            # Create OHLCV data with an uptrend
            open_price = 100.0 + i
            high = open_price + 2.0
            low = open_price - 1.0
            close = open_price + 1.0
            volume = 10000 + i * 100

            ohlcv = OHLCV(
                timestamp=date,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume
            )

            stock.add_data_point(TimeFrame.DAILY, ohlcv)

        return stock

    @pytest.fixture
    def signal(self):
        """Create a sample trading signal."""
        return {
            "timestamp": datetime(2023, 1, 1),
            "symbol": "AAPL",
            "direction": TradeDirection.LONG,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "reason": "Test signal"
        }

    def test_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.initial_balance == 1000.0
        assert simulator.currency == "GBP"
        assert simulator.risk_per_trade == 0.02
        assert simulator.max_open_positions == 3
        assert simulator.slippage == 0.001
        assert simulator.commission == 0.002
        assert simulator.enable_fractional_shares == True

        assert simulator.account.initial_balance == 1000.0
        assert simulator.account.current_balance == 1000.0
        assert len(simulator.open_trades_by_symbol) == 0

    def test_process_signal(self, simulator, stock, signal):
        """Test processing a trading signal."""
        # Process the signal
        current_time = datetime(2023, 1, 1)
        current_price = 100.0

        trade = simulator.process_signal(signal, stock, current_price, current_time)

        # Check that a trade was created
        assert trade is not None
        assert trade.symbol == "AAPL"
        assert trade.direction == TradeDirection.LONG
        assert trade.entry_time == current_time

        # Check that entry price includes slippage
        assert trade.entry_price == 100.1  # 100 * (1 + 0.001)

        # Check that the trade was added to the account
        assert len(simulator.account.trades) == 1
        assert len(simulator.account.open_positions) == 1
        assert simulator.account.open_positions[0] == trade

        # Check that the trade was added to open trades by symbol
        assert "AAPL" in simulator.open_trades_by_symbol
        assert simulator.open_trades_by_symbol["AAPL"] == trade

        # Check that commission was deducted
        expected_commission = trade.entry_price * trade.position_size * simulator.commission
        expected_balance = 1000.0 - expected_commission
        assert simulator.account.current_balance == pytest.approx(expected_balance)

        # Try to process another signal for the same symbol
        trade2 = simulator.process_signal(signal, stock, current_price, current_time)

        # Check that no trade was created
        assert trade2 is None
        assert len(simulator.account.trades) == 1
        assert len(simulator.account.open_positions) == 1

    def test_update_positions(self, simulator, stock, signal):
        """Test updating positions."""
        # Process a signal to create a trade
        current_time = datetime(2023, 1, 1)
        current_price = 100.0

        # Create a modified stock with lower high price to avoid triggering take profit
        modified_stock = Stock(symbol="AAPL")

        # Add a data point with high price below take profit
        ohlcv = OHLCV(
            timestamp=current_time,
            open=100.0,
            high=105.0,  # Below take profit of 110.0
            low=98.0,    # Above stop loss of 95.0
            close=102.0,
            volume=10000
        )

        modified_stock.add_data_point(TimeFrame.DAILY, ohlcv)

        # Process the signal
        trade = simulator.process_signal(signal, modified_stock, current_price, current_time)

        # Create a dictionary of stocks
        stocks = {"AAPL": modified_stock}

        # Update positions
        closed_trades = simulator.update_positions(stocks, current_time)

        # Check that no trades were closed
        assert len(closed_trades) == 0
        assert len(simulator.account.open_positions) == 1

        # Modify the stock data to trigger a stop loss
        # The last data point has low = 98.0, which is above the stop loss of 95.0
        # Let's add a new data point with low below the stop loss
        new_date = datetime(2023, 1, 11)
        new_ohlcv = OHLCV(
            timestamp=new_date,
            open=96.0,
            high=97.0,
            low=94.0,  # Below stop loss of 95.0
            close=95.0,
            volume=10000
        )

        modified_stock.add_data_point(TimeFrame.DAILY, new_ohlcv)

        # Update positions again
        closed_trades = simulator.update_positions(stocks, new_date)

        # Check that the trade was closed
        assert len(closed_trades) == 1
        assert closed_trades[0] == trade
        assert len(simulator.account.open_positions) == 0
        assert "AAPL" not in simulator.open_trades_by_symbol

        # Check that the trade was closed at the stop loss
        assert trade.exit_price == pytest.approx(95.0 * (1 - simulator.slippage))  # Stop loss with slippage
        assert trade.exit_time == new_date
        assert trade.exit_reason == "Stop loss hit"

        # Check that commission was deducted
        expected_commission = trade.exit_price * trade.position_size * simulator.commission
        assert simulator.account.current_balance < 1000.0  # Balance should be reduced

    def test_get_account_summary(self, simulator, stock, signal):
        """Test getting account summary."""
        # Process a signal to create a trade
        current_time = datetime(2023, 1, 1)
        current_price = 100.0

        trade = simulator.process_signal(signal, stock, current_price, current_time)

        # Get account summary
        summary = simulator.get_account_summary()

        # Check summary
        assert summary["initial_balance"] == 1000.0
        assert summary["current_balance"] < 1000.0  # Reduced by commission
        assert summary["profit_loss"] < 0  # Negative due to commission
        assert summary["profit_loss_percent"] < 0  # Negative due to commission
        assert summary["open_positions"] == 1
        assert summary["total_trades"] == 1
        assert summary["currency"] == "GBP"

    def test_get_trade_history(self, simulator, stock, signal):
        """Test getting trade history."""
        # Process a signal to create a trade
        current_time = datetime(2023, 1, 1)
        current_price = 100.0

        trade = simulator.process_signal(signal, stock, current_price, current_time)

        # Close the trade
        exit_time = datetime(2023, 1, 2)
        simulator.account.close_trade(trade, 105.0, exit_time, "Take profit hit")

        # Remove from open trades
        del simulator.open_trades_by_symbol["AAPL"]

        # Get trade history
        history = simulator.get_trade_history()

        # Check history
        assert len(history) == 1
        assert history[0]["symbol"] == "AAPL"
        assert history[0]["direction"] == "long"
        assert history[0]["entry_time"] == current_time
        assert history[0]["exit_time"] == exit_time
        assert history[0]["entry_price"] == 100.1  # With slippage
        assert history[0]["exit_price"] == 105.0
        assert history[0]["exit_reason"] == "Take profit hit"
        assert history[0]["is_open"] == False
