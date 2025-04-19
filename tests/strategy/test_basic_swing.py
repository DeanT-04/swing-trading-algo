#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the basic swing trading strategy.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.data.models import OHLCV, Stock, TimeFrame, TradeDirection
from src.strategy.basic_swing import BasicSwingStrategy


class TestBasicSwingStrategy:
    """Test suite for BasicSwingStrategy class."""
    
    @pytest.fixture
    def strategy(self):
        """Create a basic swing strategy instance."""
        config = {
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
        }
        return BasicSwingStrategy(config)
    
    @pytest.fixture
    def stock(self):
        """Create a stock with sample data."""
        stock = Stock(symbol="AAPL")
        
        # Create 100 days of sample data
        base_date = datetime(2023, 1, 1)
        
        # Start with a price of 100
        price = 100.0
        
        # Create an uptrend
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
    
    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "Basic Swing Strategy"
        assert strategy.ma_type == "EMA"
        assert strategy.fast_ma_period == 9
        assert strategy.slow_ma_period == 21
        assert strategy.trend_ma_period == 50
        assert strategy.rsi_period == 14
        assert strategy.rsi_overbought == 70
        assert strategy.rsi_oversold == 30
        assert strategy.atr_period == 14
        assert strategy.atr_multiplier == 2.0
        assert strategy.volume_ma_period == 20
        assert strategy.risk_reward_ratio == 2.0
    
    def test_analyze(self, strategy, stock):
        """Test strategy analysis."""
        # Analyze the stock
        results = strategy.analyze(stock, TimeFrame.DAILY)
        
        # Check that all expected indicators are calculated
        assert "fast_ma" in results
        assert "slow_ma" in results
        assert "trend_ma" in results
        assert "rsi" in results
        assert "atr" in results
        assert "volume_ma" in results
        assert "trend" in results
        assert "signals" in results
        
        # Check that indicators have the correct length
        df = stock.get_dataframe(TimeFrame.DAILY)
        assert len(results["fast_ma"]) == len(df)
        assert len(results["slow_ma"]) == len(df)
        assert len(results["trend_ma"]) == len(df)
        assert len(results["rsi"]) == len(df)
        assert len(results["atr"]) == len(df)
        assert len(results["volume_ma"]) == len(df)
        assert len(results["trend"]) == len(df)
        
        # Check that signals is a list
        assert isinstance(results["signals"], list)
    
    def test_determine_trend(self, strategy, stock):
        """Test trend determination."""
        # Analyze the stock
        results = strategy.analyze(stock, TimeFrame.DAILY)
        
        # Check that trend is determined for each data point
        assert len(results["trend"]) == len(stock.data[TimeFrame.DAILY])
        
        # Check that trend values are valid
        for trend in results["trend"]:
            assert trend in ["uptrend", "downtrend", "sideways"]
    
    def test_generate_signals(self, strategy, stock):
        """Test signal generation."""
        # Analyze the stock
        results = strategy.analyze(stock, TimeFrame.DAILY)
        
        # Check that signals is a list
        assert isinstance(results["signals"], list)
        
        # If there are signals, check their structure
        for signal in results["signals"]:
            assert "timestamp" in signal
            assert "direction" in signal
            assert "entry_price" in signal
            assert "stop_loss" in signal
            assert "take_profit" in signal
            assert "reason" in signal
            
            # Check that direction is a valid TradeDirection
            assert signal["direction"] in [TradeDirection.LONG, TradeDirection.SHORT]
            
            # Check that prices are valid
            assert signal["entry_price"] > 0
            assert signal["stop_loss"] > 0
            assert signal["take_profit"] > 0
            
            # Check that stop loss and take profit are consistent with direction
            if signal["direction"] == TradeDirection.LONG:
                assert signal["stop_loss"] < signal["entry_price"]
                assert signal["take_profit"] > signal["entry_price"]
            else:  # SHORT
                assert signal["stop_loss"] > signal["entry_price"]
                assert signal["take_profit"] < signal["entry_price"]
