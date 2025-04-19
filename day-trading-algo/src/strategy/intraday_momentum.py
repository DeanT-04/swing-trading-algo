#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Intraday momentum trading strategy implementation.

This module defines a day trading strategy based on momentum indicators,
MACD, RSI, and volume patterns to identify potential intraday entry and exit points.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, time

from src.analysis.indicators import (
    simple_moving_average,
    exponential_moving_average,
    relative_strength_index,
    average_true_range,
    macd,
    bollinger_bands,
    volume_oscillator,
    vwap
)
from src.data.models import Stock, TimeFrame, TradeDirection
from src.risk.position_sizing import calculate_stop_loss, calculate_take_profit
from src.analysis.market_regime import MarketRegimeDetector, MarketRegime

logger = logging.getLogger(__name__)


class IntradayMomentumStrategy:
    """
    An intraday momentum trading strategy that uses MACD, RSI, and volume
    to identify potential entry and exit points within the trading day.
    """

    def __init__(self, config: Dict):
        """
        Initialize the strategy with configuration parameters.

        Args:
            config: Strategy configuration dictionary
        """
        self.name = "Intraday Momentum Strategy"
        self.config = config

        # Extract configuration parameters
        self.ma_type = config.get("ma_type", "EMA")
        self.fast_ma_period = config.get("fast_ma_period", 8)
        self.slow_ma_period = config.get("slow_ma_period", 21)
        self.trend_ma_period = config.get("trend_ma_period", 50)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 1.5)
        self.volume_ma_period = config.get("volume_ma_period", 10)
        self.risk_reward_ratio = config.get("risk_reward_ratio", 1.5)
        
        # MACD parameters
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)
        
        # Bollinger Bands parameters
        self.bb_period = config.get("bb_period", 20)
        self.bb_std_dev = config.get("bb_std_dev", 2.0)
        
        # Day trading specific parameters
        self.session_start = config.get("session_start", "09:30")
        self.session_end = config.get("session_end", "16:00")
        self.avoid_first_minutes = config.get("avoid_first_minutes", 15)
        self.avoid_last_minutes = config.get("avoid_last_minutes", 15)
        self.max_trades_per_day = config.get("max_trades_per_day", 10)
        self.profit_taking_threshold = config.get("profit_taking_threshold", 0.5)
        self.stop_loss_threshold = config.get("stop_loss_threshold", 0.3)
        
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
        
        # Initialize market regime detector
        regime_config = {
            "lookback_period": 50,
            "trend_threshold": 0.1,
            "volatility_threshold": 0.15,
            "sideways_threshold": 0.05,
            "rsi_period": self.rsi_period,
            "atr_period": self.atr_period,
            "ma_period": self.trend_ma_period
        }
        self.regime_detector = MarketRegimeDetector(regime_config)

        logger.info(f"Initialized {self.name} with parameters: {config}")

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
            results["fast_ma"] = simple_moving_average(df["close"], self.fast_ma_period)
            results["slow_ma"] = simple_moving_average(df["close"], self.slow_ma_period)
            results["trend_ma"] = simple_moving_average(df["close"], self.trend_ma_period)
        else:  # Default to EMA
            results["fast_ma"] = exponential_moving_average(df["close"], self.fast_ma_period)
            results["slow_ma"] = exponential_moving_average(df["close"], self.slow_ma_period)
            results["trend_ma"] = exponential_moving_average(df["close"], self.trend_ma_period)

        # Calculate RSI
        results["rsi"] = relative_strength_index(df["close"], self.rsi_period)

        # Calculate ATR
        results["atr"] = average_true_range(df["high"], df["low"], df["close"], self.atr_period)

        # Calculate volume moving average
        results["volume_ma"] = simple_moving_average(df["volume"], self.volume_ma_period)
        
        # Calculate MACD
        macd_line, signal_line, histogram = macd(
            df["close"], 
            fast_period=self.macd_fast, 
            slow_period=self.macd_slow, 
            signal_period=self.macd_signal
        )
        results["macd"] = macd_line
        results["macd_signal"] = signal_line
        results["macd_hist"] = histogram
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = bollinger_bands(
            df["close"], 
            period=self.bb_period, 
            std_dev=self.bb_std_dev
        )
        results["bb_upper"] = upper_band
        results["bb_middle"] = middle_band
        results["bb_lower"] = lower_band
        
        # Calculate Volume Oscillator
        results["volume_osc"] = volume_oscillator(
            df["volume"], 
            short_period=5, 
            long_period=10
        )
        
        # Calculate VWAP (Volume Weighted Average Price)
        if "timestamp" in df.columns:
            results["vwap"] = vwap(df)
        
        # Add time-based features
        self._add_time_features(df, results)

        # Determine trend
        results["trend"] = self._determine_trend(df, results)
        
        # Detect market regime
        regime_result = self.regime_detector.detect_regime(stock, timeframe)
        results["market_regime"] = regime_result.primary_regime
        results["secondary_regime"] = regime_result.secondary_regime
        results["regime_confidence"] = regime_result.confidence

        # Generate signals
        results["signals"] = self._generate_signals(df, results)

        return results
    
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
            if df["close"].iloc[i] > indicators["trend_ma"][i] and \
               indicators["fast_ma"][i] > indicators["slow_ma"][i]:
                trend[i] = "uptrend"
            # Price below trend MA indicates downtrend
            elif df["close"].iloc[i] < indicators["trend_ma"][i] and \
                 indicators["fast_ma"][i] < indicators["slow_ma"][i]:
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
            self.atr_period,
            self.volume_ma_period,
            self.macd_slow + self.macd_signal,
            self.bb_period
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
        current_high = df["high"].iloc[index]
        current_low = df["low"].iloc[index]
        current_volume = df["volume"].iloc[index]
        current_trend = indicators["trend"][index]
        current_rsi = indicators["rsi"][index]
        current_atr = indicators["atr"][index]
        current_fast_ma = indicators["fast_ma"][index]
        current_slow_ma = indicators["slow_ma"][index]
        current_trend_ma = indicators["trend_ma"][index]
        current_volume_ma = indicators["volume_ma"][index]
        current_macd = indicators["macd"][index]
        current_macd_signal = indicators["macd_signal"][index]
        current_macd_hist = indicators["macd_hist"][index]
        current_bb_upper = indicators["bb_upper"][index]
        current_bb_lower = indicators["bb_lower"][index]
        current_volume_osc = indicators["volume_osc"][index] if "volume_osc" in indicators else 0

        # Previous values
        prev_close = df["close"].iloc[index - 1]
        prev_rsi = indicators["rsi"][index - 1]
        prev_fast_ma = indicators["fast_ma"][index - 1]
        prev_slow_ma = indicators["slow_ma"][index - 1]
        prev_macd = indicators["macd"][index - 1]
        prev_macd_signal = indicators["macd_signal"][index - 1]
        prev_macd_hist = indicators["macd_hist"][index - 1]
        
        # Get market regime
        market_regime = indicators.get("market_regime", MarketRegime.UNKNOWN)
        secondary_regime = indicators.get("secondary_regime")
        regime_confidence = indicators.get("regime_confidence", 0.0)

        signal = None
        
        # Adjust parameters based on market regime
        rsi_oversold = self.rsi_oversold
        rsi_overbought = self.rsi_overbought
        atr_multiplier = self.atr_multiplier
        risk_reward_ratio = self.risk_reward_ratio
        volume_threshold = 0.8
        
        # Adjust parameters based on market regime
        if market_regime == MarketRegime.STRONG_UPTREND:
            # In strong uptrend, be more aggressive with long entries
            rsi_oversold = min(self.rsi_oversold + 10, 40)  # Higher RSI threshold for oversold
            atr_multiplier = self.atr_multiplier * 0.8  # Tighter stops
            risk_reward_ratio = self.risk_reward_ratio * 1.2  # Higher profit targets
            volume_threshold = 0.7  # Lower volume requirement
        elif market_regime == MarketRegime.WEAK_UPTREND:
            # In weak uptrend, be slightly more aggressive with long entries
            rsi_oversold = min(self.rsi_oversold + 5, 35)  # Slightly higher RSI threshold
            atr_multiplier = self.atr_multiplier * 0.9  # Slightly tighter stops
        elif market_regime == MarketRegime.STRONG_DOWNTREND:
            # In strong downtrend, be more aggressive with short entries
            rsi_overbought = max(self.rsi_overbought - 10, 60)  # Lower RSI threshold for overbought
            atr_multiplier = self.atr_multiplier * 0.8  # Tighter stops
            risk_reward_ratio = self.risk_reward_ratio * 1.2  # Higher profit targets
            volume_threshold = 0.7  # Lower volume requirement
        elif market_regime == MarketRegime.WEAK_DOWNTREND:
            # In weak downtrend, be slightly more aggressive with short entries
            rsi_overbought = max(self.rsi_overbought - 5, 65)  # Slightly lower RSI threshold
            atr_multiplier = self.atr_multiplier * 0.9  # Slightly tighter stops
        elif market_regime == MarketRegime.SIDEWAYS:
            # In sideways market, be more conservative
            rsi_oversold = max(self.rsi_oversold - 5, 25)  # Lower RSI threshold for oversold
            rsi_overbought = min(self.rsi_overbought + 5, 75)  # Higher RSI threshold for overbought
            atr_multiplier = self.atr_multiplier * 1.1  # Wider stops
            risk_reward_ratio = self.risk_reward_ratio * 0.9  # Lower profit targets
            volume_threshold = 0.9  # Higher volume requirement
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            # In high volatility, be more conservative
            atr_multiplier = self.atr_multiplier * 1.2  # Wider stops
            risk_reward_ratio = self.risk_reward_ratio * 1.3  # Higher profit targets
            volume_threshold = 1.0  # Higher volume requirement
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            # In low volatility, be more aggressive
            atr_multiplier = self.atr_multiplier * 0.8  # Tighter stops
            risk_reward_ratio = self.risk_reward_ratio * 0.8  # Lower profit targets
            volume_threshold = 0.7  # Lower volume requirement

        # Check for long entry signal - Momentum strategy
        long_conditions = [
            # Base trend condition
            current_trend == "uptrend" or current_close > current_trend_ma,
            
            # MACD bullish crossover or positive histogram
            (prev_macd < prev_macd_signal and current_macd > current_macd_signal) or 
            (current_macd_hist > 0 and current_macd_hist > prev_macd_hist),
            
            # RSI condition - not overbought and rising
            current_rsi < rsi_overbought and current_rsi > prev_rsi,
            
            # Price near lower Bollinger Band or bouncing off support
            current_close < current_bb_lower * 1.05 or 
            (current_low <= current_bb_lower and current_close > current_bb_lower),
            
            # Volume confirmation
            current_volume > volume_threshold * current_volume_ma,
            
            # Volume oscillator positive or increasing
            current_volume_osc > 0 or (current_volume_osc > prev_macd_hist)
        ]
        
        # Additional conditions based on market regime
        if market_regime in [MarketRegime.STRONG_DOWNTREND, MarketRegime.WEAK_DOWNTREND]:
            # In downtrend, require stronger signals for long entries
            long_conditions.append(current_rsi < 30 and current_rsi > prev_rsi)
            long_conditions.append(current_volume > 1.2 * current_volume_ma)
        
        # Check if all long conditions are met
        if all(long_conditions) and market_regime != MarketRegime.STRONG_DOWNTREND:
            # Calculate stop loss and take profit with adjusted parameters
            stop_loss = calculate_stop_loss(
                entry_price=current_close,
                direction="long",
                method="atr",
                atr_value=current_atr,
                atr_multiplier=atr_multiplier
            )

            take_profit = calculate_take_profit(
                entry_price=current_close,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward_ratio
            )

            signal = {
                "timestamp": df.index[index],
                "symbol": None,  # To be filled by the caller
                "direction": TradeDirection.LONG,
                "entry_price": current_close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reason": f"Long entry: {market_regime.value}, MACD crossover, RSI: {current_rsi:.1f}, near BB lower, volume confirmation"
            }

        # Check for short entry signal - Momentum strategy
        short_conditions = [
            # Base trend condition
            current_trend == "downtrend" or current_close < current_trend_ma,
            
            # MACD bearish crossover or negative histogram
            (prev_macd > prev_macd_signal and current_macd < current_macd_signal) or 
            (current_macd_hist < 0 and current_macd_hist < prev_macd_hist),
            
            # RSI condition - not oversold and falling
            current_rsi > rsi_oversold and current_rsi < prev_rsi,
            
            # Price near upper Bollinger Band or bouncing off resistance
            current_close > current_bb_upper * 0.95 or 
            (current_high >= current_bb_upper and current_close < current_bb_upper),
            
            # Volume confirmation
            current_volume > volume_threshold * current_volume_ma,
            
            # Volume oscillator negative or decreasing
            current_volume_osc < 0 or (current_volume_osc < prev_macd_hist)
        ]
        
        # Additional conditions based on market regime
        if market_regime in [MarketRegime.STRONG_UPTREND, MarketRegime.WEAK_UPTREND]:
            # In uptrend, require stronger signals for short entries
            short_conditions.append(current_rsi > 70 and current_rsi < prev_rsi)
            short_conditions.append(current_volume > 1.2 * current_volume_ma)
        
        # Check if all short conditions are met and we don't already have a long signal
        if all(short_conditions) and not signal and market_regime != MarketRegime.STRONG_UPTREND:
            # Calculate stop loss and take profit with adjusted parameters
            stop_loss = calculate_stop_loss(
                entry_price=current_close,
                direction="short",
                method="atr",
                atr_value=current_atr,
                atr_multiplier=atr_multiplier
            )

            take_profit = calculate_take_profit(
                entry_price=current_close,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward_ratio
            )

            signal = {
                "timestamp": df.index[index],
                "symbol": None,  # To be filled by the caller
                "direction": TradeDirection.SHORT,
                "entry_price": current_close,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reason": f"Short entry: {market_regime.value}, MACD crossover, RSI: {current_rsi:.1f}, near BB upper, volume confirmation"
            }

        return signal
