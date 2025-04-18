#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market regime detection module.

This module provides functionality to detect different market regimes
(trending, ranging, volatile, etc.) to adapt trading strategies accordingly.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from src.data.models import Stock, TimeFrame, OHLCV


class MarketRegime(Enum):
    """Enum representing different market regimes."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeDetectionResult:
    """Result of market regime detection."""
    primary_regime: MarketRegime
    secondary_regime: Optional[MarketRegime] = None
    confidence: float = 0.0
    metrics: Dict[str, float] = None


class MarketRegimeDetector:
    """
    Detects market regimes based on various technical indicators.
    
    This class analyzes price data to determine the current market regime,
    which can be used to adapt trading strategies to different market conditions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the market regime detector.
        
        Args:
            config: Configuration dictionary with parameters for regime detection
        """
        self.config = config
        self.lookback_period = config.get("lookback_period", 50)
        self.trend_threshold = config.get("trend_threshold", 0.1)
        self.volatility_threshold = config.get("volatility_threshold", 0.15)
        self.sideways_threshold = config.get("sideways_threshold", 0.05)
        self.rsi_period = config.get("rsi_period", 14)
        self.atr_period = config.get("atr_period", 14)
        self.ma_period = config.get("ma_period", 50)
    
    def detect_regime(self, stock: Stock, timeframe: TimeFrame, 
                     end_idx: int = -1) -> RegimeDetectionResult:
        """
        Detect the current market regime for a stock.
        
        Args:
            stock: Stock object with price data
            timeframe: TimeFrame to analyze
            end_idx: Index of the last data point to consider (default: -1, the latest)
            
        Returns:
            RegimeDetectionResult object with detected regime and confidence
        """
        if timeframe not in stock.data or not stock.data[timeframe]:
            return RegimeDetectionResult(
                primary_regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                metrics={}
            )
        
        # Get data for analysis
        data = stock.data[timeframe]
        if end_idx < 0:
            end_idx = len(data) + end_idx
        
        start_idx = max(0, end_idx - self.lookback_period)
        if end_idx - start_idx < 10:  # Need at least 10 data points
            return RegimeDetectionResult(
                primary_regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                metrics={}
            )
        
        analysis_data = data[start_idx:end_idx+1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(analysis_data)
        
        # Determine primary regime
        primary_regime, secondary_regime, confidence = self._determine_regime(metrics)
        
        return RegimeDetectionResult(
            primary_regime=primary_regime,
            secondary_regime=secondary_regime,
            confidence=confidence,
            metrics=metrics
        )
    
    def _calculate_metrics(self, data: List[OHLCV]) -> Dict[str, float]:
        """
        Calculate various metrics for regime detection.
        
        Args:
            data: List of OHLCV data points
            
        Returns:
            Dictionary of calculated metrics
        """
        # Extract close prices
        closes = np.array([point.close for point in data])
        highs = np.array([point.high for point in data])
        lows = np.array([point.low for point in data])
        
        # Calculate trend metrics
        price_change = (closes[-1] / closes[0]) - 1
        
        # Calculate moving average
        ma_period = min(self.ma_period, len(closes) - 1)
        ma = np.mean(closes[-ma_period:])
        ma_slope = (closes[-1] - closes[-ma_period]) / (ma_period * closes[-ma_period])
        
        # Calculate volatility metrics
        daily_returns = np.diff(closes) / closes[:-1]
        volatility = np.std(daily_returns)
        
        # Calculate ATR-based volatility
        true_ranges = []
        for i in range(1, len(data)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        atr = np.mean(true_ranges[-self.atr_period:]) if true_ranges else 0
        atr_pct = atr / closes[-1] if closes[-1] > 0 else 0
        
        # Calculate RSI
        rsi = self._calculate_rsi(closes, self.rsi_period)
        
        # Calculate range metrics
        price_range = (np.max(closes) - np.min(closes)) / np.min(closes)
        
        # Calculate linear regression
        x = np.arange(len(closes))
        slope, intercept = np.polyfit(x, closes, 1)
        r_squared = self._calculate_r_squared(x, closes, slope, intercept)
        
        return {
            "price_change": price_change,
            "ma_slope": ma_slope,
            "volatility": volatility,
            "atr_pct": atr_pct,
            "rsi": rsi,
            "price_range": price_range,
            "r_squared": r_squared,
            "slope": slope,
            "current_price": closes[-1],
            "ma": ma
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            prices: Array of price data
            period: RSI period
            
        Returns:
            RSI value
        """
        if len(prices) <= period:
            return 50.0  # Default to neutral if not enough data
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Calculate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, 
                           slope: float, intercept: float) -> float:
        """
        Calculate R-squared value for linear regression.
        
        Args:
            x: X values
            y: Y values
            slope: Slope of regression line
            intercept: Intercept of regression line
            
        Returns:
            R-squared value
        """
        y_pred = slope * x + intercept
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        
        if ss_total == 0:
            return 0.0
        
        return 1 - (ss_residual / ss_total)
    
    def _determine_regime(self, metrics: Dict[str, float]) -> Tuple[MarketRegime, Optional[MarketRegime], float]:
        """
        Determine the market regime based on calculated metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Tuple of (primary_regime, secondary_regime, confidence)
        """
        price_change = metrics["price_change"]
        ma_slope = metrics["ma_slope"]
        volatility = metrics["volatility"]
        atr_pct = metrics["atr_pct"]
        rsi = metrics["rsi"]
        r_squared = metrics["r_squared"]
        current_price = metrics["current_price"]
        ma = metrics["ma"]
        
        # Initialize scores for each regime
        regime_scores = {
            MarketRegime.STRONG_UPTREND: 0,
            MarketRegime.WEAK_UPTREND: 0,
            MarketRegime.SIDEWAYS: 0,
            MarketRegime.WEAK_DOWNTREND: 0,
            MarketRegime.STRONG_DOWNTREND: 0,
            MarketRegime.HIGH_VOLATILITY: 0,
            MarketRegime.LOW_VOLATILITY: 0
        }
        
        # Score based on price change
        if price_change > self.trend_threshold * 2:
            regime_scores[MarketRegime.STRONG_UPTREND] += 2
        elif price_change > self.trend_threshold:
            regime_scores[MarketRegime.WEAK_UPTREND] += 2
        elif price_change < -self.trend_threshold * 2:
            regime_scores[MarketRegime.STRONG_DOWNTREND] += 2
        elif price_change < -self.trend_threshold:
            regime_scores[MarketRegime.WEAK_DOWNTREND] += 2
        else:
            regime_scores[MarketRegime.SIDEWAYS] += 2
        
        # Score based on MA slope
        if ma_slope > self.trend_threshold / 10:
            regime_scores[MarketRegime.STRONG_UPTREND] += 1
        elif ma_slope > self.trend_threshold / 20:
            regime_scores[MarketRegime.WEAK_UPTREND] += 1
        elif ma_slope < -self.trend_threshold / 10:
            regime_scores[MarketRegime.STRONG_DOWNTREND] += 1
        elif ma_slope < -self.trend_threshold / 20:
            regime_scores[MarketRegime.WEAK_DOWNTREND] += 1
        else:
            regime_scores[MarketRegime.SIDEWAYS] += 1
        
        # Score based on price vs MA
        if current_price > ma * 1.05:
            regime_scores[MarketRegime.STRONG_UPTREND] += 1
        elif current_price > ma:
            regime_scores[MarketRegime.WEAK_UPTREND] += 1
        elif current_price < ma * 0.95:
            regime_scores[MarketRegime.STRONG_DOWNTREND] += 1
        elif current_price < ma:
            regime_scores[MarketRegime.WEAK_DOWNTREND] += 1
        else:
            regime_scores[MarketRegime.SIDEWAYS] += 1
        
        # Score based on RSI
        if rsi > 70:
            regime_scores[MarketRegime.STRONG_UPTREND] += 1
        elif rsi > 60:
            regime_scores[MarketRegime.WEAK_UPTREND] += 1
        elif rsi < 30:
            regime_scores[MarketRegime.STRONG_DOWNTREND] += 1
        elif rsi < 40:
            regime_scores[MarketRegime.WEAK_DOWNTREND] += 1
        else:
            regime_scores[MarketRegime.SIDEWAYS] += 1
        
        # Score based on volatility
        if atr_pct > self.volatility_threshold:
            regime_scores[MarketRegime.HIGH_VOLATILITY] += 2
        elif atr_pct < self.volatility_threshold / 2:
            regime_scores[MarketRegime.LOW_VOLATILITY] += 2
        
        # Score based on R-squared (linearity of trend)
        if r_squared > 0.7:
            if price_change > 0:
                regime_scores[MarketRegime.STRONG_UPTREND] += 1
            else:
                regime_scores[MarketRegime.STRONG_DOWNTREND] += 1
        elif r_squared < 0.3:
            regime_scores[MarketRegime.SIDEWAYS] += 1
        
        # Find primary and secondary regimes
        sorted_regimes = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)
        primary_regime = sorted_regimes[0][0]
        secondary_regime = sorted_regimes[1][0] if sorted_regimes[1][1] > 0 else None
        
        # Calculate confidence
        total_score = sum(regime_scores.values())
        confidence = sorted_regimes[0][1] / total_score if total_score > 0 else 0.0
        
        # Special case: if high volatility score is high, make it secondary regime
        if (primary_regime != MarketRegime.HIGH_VOLATILITY and 
            regime_scores[MarketRegime.HIGH_VOLATILITY] >= 2):
            secondary_regime = MarketRegime.HIGH_VOLATILITY
        
        return primary_regime, secondary_regime, confidence
