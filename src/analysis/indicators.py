#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical indicators for market analysis.

This module provides functions to calculate various technical indicators
used in the trading strategy, such as moving averages, RSI, and ATR.
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
                   num_std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands for a series of prices.

    Args:
        data: Price data as a list, numpy array, or pandas Series
        period: Number of periods for the moving average
        num_std_dev: Number of standard deviations for the bands

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
        std_dev = np.std(data[i - period + 1:i + 1])
        upper_band[i] = middle_band[i] + (std_dev * num_std_dev)
        lower_band[i] = middle_band[i] - (std_dev * num_std_dev)

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


def stochastic_oscillator(high: Union[List[float], np.ndarray, pd.Series],
                         low: Union[List[float], np.ndarray, pd.Series],
                         close: Union[List[float], np.ndarray, pd.Series],
                         k_period: int = 14,
                         d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Stochastic Oscillator for a series of prices.

    Args:
        high: High prices as a list, numpy array, or pandas Series
        low: Low prices as a list, numpy array, or pandas Series
        close: Close prices as a list, numpy array, or pandas Series
        k_period: Number of periods for %K calculation
        d_period: Number of periods for %D calculation (simple moving average of %K)

    Returns:
        Tuple of numpy.ndarray: (%K, %D)
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

    if k_period <= 0 or d_period <= 0:
        raise ValueError("All periods must be positive")
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must have the same length")
    if len(high) <= k_period:
        raise ValueError(f"Data length ({len(high)}) must be greater than k_period ({k_period})")

    # Initialize result arrays with NaNs
    k = np.full_like(close, np.nan, dtype=float)
    d = np.full_like(close, np.nan, dtype=float)

    # Calculate %K
    for i in range(k_period - 1, len(close)):
        highest_high = np.max(high[i - k_period + 1:i + 1])
        lowest_low = np.min(low[i - k_period + 1:i + 1])

        if highest_high == lowest_low:  # Avoid division by zero
            k[i] = 50.0
        else:
            k[i] = 100.0 * (close[i] - lowest_low) / (highest_high - lowest_low)

    # Calculate %D (SMA of %K)
    d = simple_moving_average(k, d_period)

    return k, d


def on_balance_volume(close: Union[List[float], np.ndarray, pd.Series],
                     volume: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Calculate On-Balance Volume (OBV) for a series of prices and volumes.

    Args:
        close: Close prices as a list, numpy array, or pandas Series
        volume: Volume data as a list, numpy array, or pandas Series

    Returns:
        numpy.ndarray: OBV values
    """
    if isinstance(close, list):
        close = np.array(close)
    elif isinstance(close, pd.Series):
        close = close.values

    if isinstance(volume, list):
        volume = np.array(volume)
    elif isinstance(volume, pd.Series):
        volume = volume.values

    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must have the same length")
    if len(close) < 2:
        raise ValueError("Data length must be at least 2")

    # Initialize result array
    obv = np.zeros_like(close)

    # First OBV value is the first volume value
    obv[0] = volume[0]

    # Calculate OBV for remaining positions
    for i in range(1, len(close)):
        if close[i] > close[i-1]:  # Price up
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:  # Price down
            obv[i] = obv[i-1] - volume[i]
        else:  # Price unchanged
            obv[i] = obv[i-1]

    return obv


def average_directional_index(high: Union[List[float], np.ndarray, pd.Series],
                             low: Union[List[float], np.ndarray, pd.Series],
                             close: Union[List[float], np.ndarray, pd.Series],
                             period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Average Directional Index (ADX) for a series of prices.

    Args:
        high: High prices as a list, numpy array, or pandas Series
        low: Low prices as a list, numpy array, or pandas Series
        close: Close prices as a list, numpy array, or pandas Series
        period: Number of periods for the ADX calculation

    Returns:
        Tuple of numpy.ndarray: (ADX, +DI, -DI, DX)
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
    if len(high) <= period + 1:
        raise ValueError(f"Data length ({len(high)}) must be greater than period+1 ({period+1})")

    # Initialize result arrays with NaNs
    adx = np.full_like(close, np.nan, dtype=float)
    plus_di = np.full_like(close, np.nan, dtype=float)
    minus_di = np.full_like(close, np.nan, dtype=float)
    dx = np.full_like(close, np.nan, dtype=float)

    # Calculate True Range
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]  # First TR is simply the first day's range

    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    # Calculate +DM and -DM
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))

    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0

    # Calculate smoothed TR, +DM, and -DM
    smoothed_tr = np.zeros(len(high))
    smoothed_plus_dm = np.zeros(len(high))
    smoothed_minus_dm = np.zeros(len(high))

    # Initialize with simple averages
    smoothed_tr[period] = np.sum(tr[1:period+1])
    smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])

    # Calculate smoothed values for remaining positions
    for i in range(period + 1, len(high)):
        smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1] / period) + tr[i]
        smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / period) + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / period) + minus_dm[i]

    # Calculate +DI and -DI
    for i in range(period, len(high)):
        if smoothed_tr[i] == 0:  # Avoid division by zero
            plus_di[i] = 0
            minus_di[i] = 0
        else:
            plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]

    # Calculate DX
    for i in range(period, len(high)):
        if plus_di[i] + minus_di[i] == 0:  # Avoid division by zero
            dx[i] = 0
        else:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])

    # Calculate ADX
    # First ADX value is the average of DX values
    adx[2*period-1] = np.mean(dx[period:2*period])

    # Calculate ADX for remaining positions
    for i in range(2*period, len(high)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di, dx


def ichimoku_cloud(high: Union[List[float], np.ndarray, pd.Series],
                  low: Union[List[float], np.ndarray, pd.Series],
                  close: Union[List[float], np.ndarray, pd.Series],
                  tenkan_period: int = 9,
                  kijun_period: int = 26,
                  senkou_span_b_period: int = 52,
                  displacement: int = 26) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Ichimoku Cloud for a series of prices.

    Args:
        high: High prices as a list, numpy array, or pandas Series
        low: Low prices as a list, numpy array, or pandas Series
        close: Close prices as a list, numpy array, or pandas Series
        tenkan_period: Number of periods for Tenkan-sen (Conversion Line)
        kijun_period: Number of periods for Kijun-sen (Base Line)
        senkou_span_b_period: Number of periods for Senkou Span B (Leading Span B)
        displacement: Number of periods for displacement (Chikou Span and Senkou Spans)

    Returns:
        Tuple of numpy.ndarray: (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)
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

    if tenkan_period <= 0 or kijun_period <= 0 or senkou_span_b_period <= 0 or displacement <= 0:
        raise ValueError("All periods must be positive")
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("High, low, and close arrays must have the same length")
    if len(high) <= max(tenkan_period, kijun_period, senkou_span_b_period, displacement):
        raise ValueError(f"Data length ({len(high)}) must be greater than the maximum period")

    # Initialize result arrays with NaNs
    tenkan_sen = np.full_like(close, np.nan, dtype=float)
    kijun_sen = np.full_like(close, np.nan, dtype=float)
    senkou_span_a = np.full_like(close, np.nan, dtype=float)
    senkou_span_b = np.full_like(close, np.nan, dtype=float)
    chikou_span = np.full_like(close, np.nan, dtype=float)

    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
    for i in range(tenkan_period - 1, len(high)):
        highest_high = np.max(high[i - tenkan_period + 1:i + 1])
        lowest_low = np.min(low[i - tenkan_period + 1:i + 1])
        tenkan_sen[i] = (highest_high + lowest_low) / 2

    # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
    for i in range(kijun_period - 1, len(high)):
        highest_high = np.max(high[i - kijun_period + 1:i + 1])
        lowest_low = np.min(low[i - kijun_period + 1:i + 1])
        kijun_sen[i] = (highest_high + lowest_low) / 2

    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 displaced forward by displacement periods
    for i in range(kijun_period - 1, len(high)):
        if i + displacement < len(high):
            senkou_span_a[i + displacement] = (tenkan_sen[i] + kijun_sen[i]) / 2

    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_span_b_period, displaced forward by displacement periods
    for i in range(senkou_span_b_period - 1, len(high)):
        highest_high = np.max(high[i - senkou_span_b_period + 1:i + 1])
        lowest_low = np.min(low[i - senkou_span_b_period + 1:i + 1])
        if i + displacement < len(high):
            senkou_span_b[i + displacement] = (highest_high + lowest_low) / 2

    # Calculate Chikou Span (Lagging Span): Close price displaced backward by displacement periods
    for i in range(displacement, len(high)):
        chikou_span[i - displacement] = close[i]

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
