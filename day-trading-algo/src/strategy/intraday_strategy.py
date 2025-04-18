#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Intraday trading strategy for the day trading algorithm.

This module implements a basic intraday trading strategy using technical indicators.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, time

from src.data.models import Stock, TimeFrame, TradeDirection

# Set up logging
logger = logging.getLogger(__name__)


class IntradayStrategy:
    """
    Basic intraday trading strategy.
    
    This class implements a simple intraday trading strategy using
    moving averages, RSI, and other technical indicators.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the strategy with configuration parameters.
        
        Args:
            config: Strategy configuration dictionary
        """
        self.config = config
        self.strategy_config = config.get("strategy", {})
        
        # Extract strategy parameters
        self.name = self.strategy_config.get("name", "intraday_strategy")
        self.ma_type = self.strategy_config.get("ma_type", "EMA")
        self.fast_ma_period = self.strategy_config.get("fast_ma_period", 8)
        self.slow_ma_period = self.strategy_config.get("slow_ma_period", 21)
        self.trend_ma_period = self.strategy_config.get("trend_ma_period", 50)
        self.rsi_period = self.strategy_config.get("rsi_period", 14)
        self.rsi_overbought = self.strategy_config.get("rsi_overbought", 70)
        self.rsi_oversold = self.strategy_config.get("rsi_oversold", 30)
        self.volume_ma_period = self.strategy_config.get("volume_ma_period", 20)
        
        # Day trading specific parameters
        self.session_start = self.strategy_config.get("session_start", "09:30")
        self.session_end = self.strategy_config.get("session_end", "16:00")
        self.avoid_first_minutes = self.strategy_config.get("avoid_first_minutes", 15)
        self.avoid_last_minutes = self.strategy_config.get("avoid_last_minutes", 15)
        
        # Convert session times to datetime.time objects
        try:
            h, m = map(int, self.session_start.split(':'))
            self.session_start_time = time(h, m)
            
            h, m = map(int, self.session_end.split(':'))
            self.session_end_time = time(h, m)
        except (ValueError, AttributeError):
            logger.warning("Invalid session time format. Using default market hours.")
            self.session_start_time = time(9, 30)
            self.session_end_time = time(16, 0)
        
        logger.info(f"Initialized {self.name} strategy with parameters: {self.strategy_config}")
    
    def analyze(self, stock: Stock, timeframe: TimeFrame) -> Dict:
        """
        Analyze the stock data and generate trading signals.
        
        Args:
            stock: Stock object with price data
            timeframe: Timeframe to analyze
            
        Returns:
            Dict: Analysis results including indicators and signals
        """
        # Get the price data as a DataFrame
        df = stock.get_dataframe(timeframe)
        if df.empty:
            logger.warning(f"No data available for {stock.symbol} on {timeframe.value} timeframe")
            return {}
        
        # Calculate indicators
        results = {}
        
        # Calculate moving averages
        if self.ma_type.upper() == "SMA":
            results["fast_ma"] = self._calculate_sma(df["close"], self.fast_ma_period)
            results["slow_ma"] = self._calculate_sma(df["close"], self.slow_ma_period)
            results["trend_ma"] = self._calculate_sma(df["close"], self.trend_ma_period)
        else:  # Default to EMA
            results["fast_ma"] = self._calculate_ema(df["close"], self.fast_ma_period)
            results["slow_ma"] = self._calculate_ema(df["close"], self.slow_ma_period)
            results["trend_ma"] = self._calculate_ema(df["close"], self.trend_ma_period)
        
        # Calculate RSI
        results["rsi"] = self._calculate_rsi(df["close"], self.rsi_period)
        
        # Calculate volume moving average
        results["volume_ma"] = self._calculate_sma(df["volume"], self.volume_ma_period)
        
        # Add time-based features
        self._add_time_features(df, results)
        
        # Determine trend
        results["trend"] = self._determine_trend(df, results)
        
        # Generate signals
        results["signals"] = self._generate_signals(df, results)
        
        return results
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            series: Price or volume series
            period: MA period
            
        Returns:
            pd.Series: SMA values
        """
        return series.rolling(window=period).mean()
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            series: Price or volume series
            period: MA period
            
        Returns:
            pd.Series: EMA values
        """
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            series: Price series
            period: RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = series.diff()
        
        # Make two series: one for gains and one for losses
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Calculate the EWMA
        roll_up = up.ewm(com=period-1, adjust=False).mean()
        roll_down = down.ewm(com=period-1, adjust=False).mean()
        
        # Calculate the RSI based on EWMA
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def _add_time_features(self, df: pd.DataFrame, results: Dict) -> None:
        """
        Add time-based features to the results.
        
        Args:
            df: Price data as a DataFrame
            results: Dictionary of calculated indicators
        """
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        
        # Extract hour of day
        results["hour_of_day"] = df.index.hour
        
        # Extract day of week (0=Monday, 6=Sunday)
        results["day_of_week"] = df.index.dayofweek
        
        # Check if market is open
        results["is_market_open"] = [
            1 if (self.session_start_time <= idx.time() <= self.session_end_time and idx.dayofweek < 5)
            else 0
            for idx in df.index
        ]
        
        # Calculate time to market close (in minutes)
        results["time_to_close"] = [
            (self.session_end_time.hour - idx.hour) * 60 + (self.session_end_time.minute - idx.minute)
            if (idx.time() <= self.session_end_time and idx.dayofweek < 5)
            else 0
            for idx in df.index
        ]
    
    def _determine_trend(self, df: pd.DataFrame, indicators: Dict) -> List[str]:
        """
        Determine the trend for each data point.
        
        Args:
            df: Price data as a DataFrame
            indicators: Dictionary of calculated indicators
            
        Returns:
            List[str]: Trend for each data point ('uptrend', 'downtrend', or 'sideways')
        """
        trend = ["sideways"] * len(df)
        
        for i in range(self.trend_ma_period, len(df)):
            # Price above trend MA indicates uptrend
            if df["close"].iloc[i] > indicators["trend_ma"].iloc[i] and \
               indicators["fast_ma"].iloc[i] > indicators["slow_ma"].iloc[i]:
                trend[i] = "uptrend"
            # Price below trend MA indicates downtrend
            elif df["close"].iloc[i] < indicators["trend_ma"].iloc[i] and \
                 indicators["fast_ma"].iloc[i] < indicators["slow_ma"].iloc[i]:
                trend[i] = "downtrend"
            # Otherwise, consider it sideways
            else:
                trend[i] = "sideways"
        
        return trend
    
    def _generate_signals(self, df: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """
        Generate trading signals based on the calculated indicators.
        
        Args:
            df: Price data as a DataFrame
            indicators: Dictionary of calculated indicators
            
        Returns:
            List[Dict]: List of signal dictionaries
        """
        signals = []
        
        # We need enough data for all our indicators
        min_periods = max(
            self.fast_ma_period,
            self.slow_ma_period,
            self.trend_ma_period,
            self.rsi_period,
            self.volume_ma_period
        )
        
        # Start analyzing from the point where all indicators are available
        for i in range(min_periods, len(df)):
            # Skip if not within trading hours
            if "is_market_open" in indicators and not indicators["is_market_open"][i]:
                continue
                
            # Skip if too close to market open or close
            if "time_to_close" in indicators:
                if indicators["time_to_close"][i] < self.avoid_last_minutes:
                    continue
                    
                # Check if within first minutes of market open
                if isinstance(df.index, pd.DatetimeIndex):
                    current_time = df.index[i].time()
                    minutes_since_open = (current_time.hour - self.session_start_time.hour) * 60 + \
                                         (current_time.minute - self.session_start_time.minute)
                    if 0 <= minutes_since_open < self.avoid_first_minutes:
                        continue
            
            signal = self._check_for_signal(df, indicators, i)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _check_for_signal(self, df: pd.DataFrame, indicators: Dict, index: int) -> Optional[Dict]:
        """
        Check for a trading signal at the specified index.
        
        Args:
            df: Price data as a DataFrame
            indicators: Dictionary of calculated indicators
            index: Index to check for a signal
            
        Returns:
            Optional[Dict]: Signal dictionary if a signal is found, None otherwise
        """
        # Extract current values
        current_close = df["close"].iloc[index]
        current_volume = df["volume"].iloc[index]
        current_trend = indicators["trend"][index]
        current_rsi = indicators["rsi"].iloc[index]
        current_fast_ma = indicators["fast_ma"].iloc[index]
        current_slow_ma = indicators["slow_ma"].iloc[index]
        current_trend_ma = indicators["trend_ma"].iloc[index]
        current_volume_ma = indicators["volume_ma"].iloc[index]
        
        # Previous values
        prev_close = df["close"].iloc[index - 1]
        prev_rsi = indicators["rsi"].iloc[index - 1]
        prev_fast_ma = indicators["fast_ma"].iloc[index - 1]
        prev_slow_ma = indicators["slow_ma"].iloc[index - 1]
        
        signal = None
        
        # Check for long entry signal
        if (
            # Uptrend condition
            (current_trend == "uptrend" or current_close > current_trend_ma) and
            # RSI condition
            (prev_rsi < self.rsi_oversold and current_rsi > self.rsi_oversold or
             current_rsi < 45 and current_rsi > prev_rsi) and
            # Moving average condition
            current_close > current_slow_ma and
            # Volume condition
            current_volume > 0.8 * current_volume_ma and
            # MA crossover condition
            (prev_fast_ma <= prev_slow_ma and current_fast_ma > current_slow_ma or
             current_fast_ma > prev_fast_ma)
        ):
            # Calculate stop loss and take profit
            stop_loss = current_close * 0.99  # 1% stop loss
            take_profit = current_close * 1.015  # 1.5% take profit
            
            signal = {
                "timestamp": df.index[index],
                "symbol": None,  # To be filled by the caller
                "direction": TradeDirection.LONG,
                "entry_price": current_close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reason": "Long entry: Uptrend, RSI oversold bounce, MA crossover, volume confirmation"
            }
        
        # Check for short entry signal
        elif (
            # Downtrend condition
            (current_trend == "downtrend" or current_close < current_trend_ma) and
            # RSI condition
            (prev_rsi > self.rsi_overbought and current_rsi < self.rsi_overbought or
             current_rsi > 55 and current_rsi < prev_rsi) and
            # Moving average condition
            current_close < current_slow_ma and
            # Volume condition
            current_volume > 0.8 * current_volume_ma and
            # MA crossover condition
            (prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma or
             current_fast_ma < prev_fast_ma)
        ):
            # Calculate stop loss and take profit
            stop_loss = current_close * 1.01  # 1% stop loss
            take_profit = current_close * 0.985  # 1.5% take profit
            
            signal = {
                "timestamp": df.index[index],
                "symbol": None,  # To be filled by the caller
                "direction": TradeDirection.SHORT,
                "entry_price": current_close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reason": "Short entry: Downtrend, RSI overbought drop, MA crossover, volume confirmation"
            }
        
        return signal
