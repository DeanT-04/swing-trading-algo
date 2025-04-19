#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical indicators for market analysis.

This module provides functions to calculate various technical indicators
used in the trading strategy, such as moving averages, RSI, MACD, and VWAP.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union


def simple_moving_average(data: Union[List[float], np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Calculate the Simple Moving Average (SMA) for a series of prices.

    Args:
        data: Price data as a list, numpy array, or pandas Series
        period: Number of periods to average

    Returns:
        numpy.ndarray: SMA values (with NaN for the first period-1 positions)
    """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.Series):
        data = data.values

    if period <= 0:
        raise ValueError("Period must be positive")
    if len(data) < period:
        raise ValueError(f"Data length ({len(data)}) must be at least as long as period ({period})")

    # Initialize result array with NaNs
    result = np.full_like(data, np.nan, dtype=float)

    # Calculate SMA for each valid position
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])

    return result


def exponential_moving_average(data: Union[List[float], np.ndarray, pd.Series], period: int,
                              alpha: Optional[float] = None) -> np.ndarray:
    """
    Calculate the Exponential Moving Average (EMA) for a series of prices.

    Args:
        data: Price data as a list, numpy array, or pandas Series
        period: Number of periods for the EMA calculation
        alpha: Smoothing factor (if None, calculated as 2/(period+1))

    Returns:
        numpy.ndarray: EMA values (with NaN for the first period-1 positions)
    """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.Series):
        data = data.values

    if period <= 0:
        raise ValueError("Period must be positive")
    if len(data) < period:
        raise ValueError(f"Data length ({len(data)}) must be at least as long as period ({period})")

    # Initialize result array with NaNs
    result = np.full_like(data, np.nan, dtype=float)

    # Set the smoothing factor
    if alpha is None:
        alpha = 2 / (period + 1)

    # Initialize EMA with SMA for the first valid position
    result[period - 1] = np.mean(data[:period])

    # Calculate EMA for remaining positions
    for i in range(period, len(data)):
        result[i] = data[i] * alpha + result[i - 1] * (1 - alpha)

    return result


def relative_strength_index(data: Union[List[float], np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Calculate the Relative Strength Index (RSI) for a series of prices.

    Args:
        data: Price data as a list, numpy array, or pandas Series
        period: Number of periods for the RSI calculation

    Returns:
        numpy.ndarray: RSI values (with NaN for the first period positions)
    """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.Series):
        data = data.values

    if period <= 1:
        raise ValueError("Period must be greater than 1")
    if len(data) <= period:
        raise ValueError(f"Data length ({len(data)}) must be greater than period ({period})")

    # Initialize result array with NaNs
    result = np.full_like(data, np.nan, dtype=float)

    # Calculate price changes
    deltas = np.diff(data)

    # Separate gains (positive) and losses (negative)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Initialize average gains and losses
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate first RSI value
    if avg_loss == 0:
        result[period] = 100
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))

    # Calculate RSI for remaining positions
    for i in range(period + 1, len(data)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            result[i] = 100
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))

    return result


def average_true_range(high: Union[List[float], np.ndarray, pd.Series],
                      low: Union[List[float], np.ndarray, pd.Series],
                      close: Union[List[float], np.ndarray, pd.Series],
                      period: int) -> np.ndarray:
    """
    Calculate the Average True Range (ATR) for a series of prices.

    Args:
        high: High prices as a list, numpy array, or pandas Series
        low: Low prices as a list, numpy array, or pandas Series
        close: Close prices as a list, numpy array, or pandas Series
        period: Number of periods for the ATR calculation

    Returns:
        numpy.ndarray: ATR values (with NaN for the first period positions)
    """
    if isinstance(high, list):
        high = np.array(high)
    elif isinstance(high, pd.Series):
        high = high.values

    if isinstance(low, list):
        low = np.array(low)
    elif isinstance(low, pd.Series):
        low = low.values

    if isinstance(close, list):
        close = np.array(close)
    elif isinstance(close, pd.Series):
        close = close.values

    if period <= 0:
        raise ValueError("Period must be positive")
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must have the same length")
    if len(high) <= period:
        raise ValueError(f"Data length ({len(high)}) must be greater than period ({period})")

    # Initialize result array with NaNs
    result = np.full_like(high, np.nan, dtype=float)

    # Calculate true range
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]  # First TR is simply the first day's range

    for i in range(1, len(high)):
        # True range is the greatest of:
        # 1. Current high - current low
        # 2. Current high - previous close (absolute)
        # 3. Current low - previous close (absolute)
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    # Calculate first ATR as simple average of TR
    result[period] = np.mean(tr[1:period+1])

    # Calculate ATR for remaining positions using smoothing
    for i in range(period + 1, len(high)):
        result[i] = (result[i-1] * (period - 1) + tr[i]) / period

    return result


def bollinger_bands(data: Union[List[float], np.ndarray, pd.Series],
                   period: int,
                   std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands for a series of prices.

    Args:
        data: Price data as a list, numpy array, or pandas Series
        period: Number of periods for the moving average
        std_dev: Number of standard deviations for the bands

    Returns:
        Tuple of numpy.ndarray: (upper band, middle band, lower band)
    """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.Series):
        data = data.values

    if period <= 0:
        raise ValueError("Period must be positive")
    if len(data) < period:
        raise ValueError(f"Data length ({len(data)}) must be at least as long as period ({period})")

    # Calculate middle band (SMA)
    middle_band = simple_moving_average(data, period)

    # Initialize upper and lower bands with NaNs
    upper_band = np.full_like(middle_band, np.nan, dtype=float)
    lower_band = np.full_like(middle_band, np.nan, dtype=float)

    # Calculate standard deviation and bands
    for i in range(period - 1, len(data)):
        std_dev_val = np.std(data[i - period + 1:i + 1])
        upper_band[i] = middle_band[i] + (std_dev_val * std_dev)
        lower_band[i] = middle_band[i] - (std_dev_val * std_dev)

    return upper_band, middle_band, lower_band


def macd(data: Union[List[float], np.ndarray, pd.Series],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Moving Average Convergence Divergence (MACD) for a series of prices.

    Args:
        data: Price data as a list, numpy array, or pandas Series
        fast_period: Number of periods for the fast EMA
        slow_period: Number of periods for the slow EMA
        signal_period: Number of periods for the signal line

    Returns:
        Tuple of numpy.ndarray: (MACD line, signal line, histogram)
    """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.Series):
        data = data.values

    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("All periods must be positive")
    if fast_period >= slow_period:
        raise ValueError("Fast period must be less than slow period")
    if len(data) <= slow_period:
        raise ValueError(f"Data length ({len(data)}) must be greater than slow period ({slow_period})")

    # Calculate fast and slow EMAs
    fast_ema = exponential_moving_average(data, fast_period)
    slow_ema = exponential_moving_average(data, slow_period)

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line
    signal_line = exponential_moving_average(macd_line, signal_period)

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def volume_oscillator(volume: Union[List[float], np.ndarray, pd.Series],
                     short_period: int = 5,
                     long_period: int = 10) -> np.ndarray:
    """
    Calculate Volume Oscillator for a series of volumes.

    Args:
        volume: Volume data as a list, numpy array, or pandas Series
        short_period: Number of periods for the short-term MA
        long_period: Number of periods for the long-term MA

    Returns:
        numpy.ndarray: Volume Oscillator values
    """
    if isinstance(volume, list):
        volume = np.array(volume)
    elif isinstance(volume, pd.Series):
        volume = volume.values

    if short_period <= 0 or long_period <= 0:
        raise ValueError("All periods must be positive")
    if short_period >= long_period:
        raise ValueError("Short period must be less than long period")
    if len(volume) <= long_period:
        raise ValueError(f"Data length ({len(volume)}) must be greater than long period ({long_period})")

    # Calculate short and long-term MAs
    short_ma = simple_moving_average(volume, short_period)
    long_ma = simple_moving_average(volume, long_period)

    # Calculate volume oscillator as percentage difference
    result = np.full_like(volume, np.nan, dtype=float)
    
    for i in range(long_period - 1, len(volume)):
        if long_ma[i] == 0:  # Avoid division by zero
            result[i] = 0
        else:
            result[i] = 100 * (short_ma[i] - long_ma[i]) / long_ma[i]

    return result


def vwap(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate Volume Weighted Average Price (VWAP) for a DataFrame with OHLCV data.

    Args:
        df: DataFrame with 'high', 'low', 'close', and 'volume' columns

    Returns:
        numpy.ndarray: VWAP values
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    required_columns = ['high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate VWAP
    vwap_values = np.full(len(df), np.nan, dtype=float)
    
    # Check if DataFrame has a datetime index to reset VWAP daily
    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    
    if has_datetime_index:
        # Group by date to reset VWAP daily
        df['date'] = df.index.date
        grouped = df.groupby('date')
        
        for date, group in grouped:
            # Get indices in the original DataFrame
            indices = group.index
            
            # Calculate cumulative (price * volume) and cumulative volume
            cum_tp_vol = (typical_price.loc[indices] * df['volume'].loc[indices]).cumsum()
            cum_vol = df['volume'].loc[indices].cumsum()
            
            # Calculate VWAP
            vwap_values[df.index.get_indexer(indices)] = cum_tp_vol / cum_vol
    else:
        # Calculate cumulative (price * volume) and cumulative volume for the entire dataset
        cum_tp_vol = (typical_price * df['volume']).cumsum()
        cum_vol = df['volume'].cumsum()
        
        # Calculate VWAP
        vwap_values = cum_tp_vol / cum_vol
    
    return vwap_values


def money_flow_index(high: Union[List[float], np.ndarray, pd.Series],
                    low: Union[List[float], np.ndarray, pd.Series],
                    close: Union[List[float], np.ndarray, pd.Series],
                    volume: Union[List[float], np.ndarray, pd.Series],
                    period: int = 14) -> np.ndarray:
    """
    Calculate Money Flow Index (MFI) for a series of prices and volumes.

    Args:
        high: High prices as a list, numpy array, or pandas Series
        low: Low prices as a list, numpy array, or pandas Series
        close: Close prices as a list, numpy array, or pandas Series
        volume: Volume data as a list, numpy array, or pandas Series
        period: Number of periods for the MFI calculation

    Returns:
        numpy.ndarray: MFI values (with NaN for the first period positions)
    """
    if isinstance(high, list):
        high = np.array(high)
    elif isinstance(high, pd.Series):
        high = high.values

    if isinstance(low, list):
        low = np.array(low)
    elif isinstance(low, pd.Series):
        low = low.values

    if isinstance(close, list):
        close = np.array(close)
    elif isinstance(close, pd.Series):
        close = close.values

    if isinstance(volume, list):
        volume = np.array(volume)
    elif isinstance(volume, pd.Series):
        volume = volume.values

    if period <= 0:
        raise ValueError("Period must be positive")
    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("High, low, close, and volume arrays must have the same length")
    if len(high) <= period:
        raise ValueError(f"Data length ({len(high)}) must be greater than period ({period})")

    # Initialize result array with NaNs
    result = np.full_like(close, np.nan, dtype=float)

    # Calculate typical price
    typical_price = (high + low + close) / 3

    # Calculate raw money flow
    raw_money_flow = typical_price * volume

    # Calculate money flow ratio
    money_flow_positive = np.zeros(len(close))
    money_flow_negative = np.zeros(len(close))

    # First value has no price change
    for i in range(1, len(close)):
        if typical_price[i] > typical_price[i-1]:  # Positive money flow
            money_flow_positive[i] = raw_money_flow[i]
            money_flow_negative[i] = 0
        elif typical_price[i] < typical_price[i-1]:  # Negative money flow
            money_flow_positive[i] = 0
            money_flow_negative[i] = raw_money_flow[i]
        else:  # No change
            money_flow_positive[i] = 0
            money_flow_negative[i] = 0

    # Calculate MFI for each valid position
    for i in range(period, len(close)):
        positive_flow_sum = np.sum(money_flow_positive[i-period+1:i+1])
        negative_flow_sum = np.sum(money_flow_negative[i-period+1:i+1])

        if negative_flow_sum == 0:  # Avoid division by zero
            result[i] = 100
        else:
            money_ratio = positive_flow_sum / negative_flow_sum
            result[i] = 100 - (100 / (1 + money_ratio))

    return result
