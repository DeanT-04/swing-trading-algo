#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pytest configuration and fixtures for the Swing Trading Algorithm tests.
"""

import os
import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.data.models import OHLCV, Stock, TimeFrame, TradeDirection, Trade, Account


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary."""
    return {
        "general": {
            "name": "Swing Trading Algorithm",
            "version": "0.1.0",
            "initial_capital": 1000.0,
            "currency": "GBP",
            "log_level": "INFO"
        },
        "data": {
            "provider": "csv",
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "timeframes": ["daily", "4h", "1h"],
            "cache_dir": "data/cache",
            "history_days": 365
        },
        "analysis": {
            "indicators": {
                "moving_averages": [
                    {"type": "SMA", "periods": [20, 50, 200]},
                    {"type": "EMA", "periods": [9, 21]}
                ],
                "rsi": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                },
                "atr": {
                    "period": 14
                }
            }
        },
        "strategy": {
            "name": "basic_swing",
            "ma_type": "EMA",
            "fast_ma_period": 9,
            "slow_ma_period": 21,
            "trend_ma_period": 50,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "volume_ma_period": 20,
            "risk_reward_ratio": 2.0
        },
        "risk": {
            "max_risk_per_trade": 0.02,
            "risk_reward_ratio": 2.0,
            "max_open_positions": 3,
            "stop_loss": {
                "method": "atr",
                "atr_multiplier": 2.0,
                "percent": 0.05
            },
            "initial_balance": 1000.0,
            "currency": "GBP",
            "risk_per_trade": 0.02,
            "slippage": 0.001,
            "commission": 0.002,
            "enable_fractional_shares": True
        },
        "simulation": {
            "slippage": 0.001,
            "commission": 0.002,
            "enable_fractional_shares": True
        },
        "performance": {
            "metrics": {
                "win_rate": True,
                "profit_factor": True,
                "max_drawdown": True,
                "sharpe_ratio": True
            },
            "reporting": {
                "frequency": "daily",
                "save_to_file": True,
                "plot_equity_curve": True
            }
        }
    }


@pytest.fixture
def sample_stock():
    """Return a sample stock with price data."""
    stock = Stock(symbol="AAPL", name="Apple Inc.", exchange="NASDAQ", sector="Technology")
    
    # Create 100 days of sample data
    base_date = datetime(2023, 1, 1)
    
    # Start with a price of 100
    price = 100.0
    
    # Create an uptrend followed by a downtrend
    for i in range(100):
        date = base_date + timedelta(days=i)
        
        # Add some randomness to the price
        random_factor = np.random.normal(0, 1)
        
        # Create an uptrend with some noise
        if i < 50:
            # First 50 days: uptrend
            price_change = 0.5 + random_factor * 0.2
        else:
            # Next 50 days: downtrend
            price_change = -0.3 + random_factor * 0.2
        
        price += price_change
        
        # Ensure price is positive
        price = max(price, 50.0)
        
        # Create OHLCV data
        high = price + abs(random_factor)
        low = price - abs(random_factor)
        open_price = price - price_change * 0.5
        close = price
        volume = int(10000 + random_factor * 1000)
        
        # Ensure high is the highest and low is the lowest
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
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
def sample_account():
    """Return a sample account."""
    return Account(initial_balance=1000.0, currency="GBP")


@pytest.fixture
def sample_trade():
    """Return a sample trade."""
    return Trade(
        symbol="AAPL",
        direction=TradeDirection.LONG,
        entry_time=datetime(2023, 1, 1),
        entry_price=100.0,
        position_size=10,
        stop_loss=95.0,
        take_profit=110.0
    )
