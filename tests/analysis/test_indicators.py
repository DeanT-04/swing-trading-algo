#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for technical indicators.
"""

import numpy as np
import pytest
from src.analysis.indicators import (
    simple_moving_average,
    exponential_moving_average,
    relative_strength_index,
    average_true_range
)


class TestIndicators:
    """Test suite for technical indicators."""
    
    def test_simple_moving_average(self):
        """Test simple moving average calculation."""
        # Test with a simple case
        data = [1, 2, 3, 4, 5]
        period = 3
        expected = [np.nan, np.nan, 2, 3, 4]
        result = simple_moving_average(data, period)
        
        # Check NaN values
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Check calculated values
        assert result[2] == expected[2]
        assert result[3] == expected[3]
        assert result[4] == expected[4]
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            simple_moving_average(data, 0)  # Period must be positive
        
        with pytest.raises(ValueError):
            simple_moving_average(data, 6)  # Data length must be at least as long as period
    
    def test_exponential_moving_average(self):
        """Test exponential moving average calculation."""
        # Test with a simple case
        data = [1, 2, 3, 4, 5]
        period = 3
        expected_first = 2  # First value is SMA
        result = exponential_moving_average(data, period)
        
        # Check NaN values
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Check first calculated value (SMA)
        assert result[2] == expected_first
        
        # Check that subsequent values are calculated
        assert not np.isnan(result[3])
        assert not np.isnan(result[4])
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            exponential_moving_average(data, 0)  # Period must be positive
        
        with pytest.raises(ValueError):
            exponential_moving_average(data, 6)  # Data length must be at least as long as period
    
    def test_relative_strength_index(self):
        """Test relative strength index calculation."""
        # Test with a simple case
        data = [10, 11, 10, 12, 11, 14, 13, 12, 13, 15]
        period = 3
        result = relative_strength_index(data, period)
        
        # Check NaN values
        for i in range(period):
            assert np.isnan(result[i])
        
        # Check that subsequent values are calculated
        for i in range(period, len(data)):
            assert not np.isnan(result[i])
            assert 0 <= result[i] <= 100  # RSI should be between 0 and 100
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            relative_strength_index(data, 1)  # Period must be greater than 1
        
        with pytest.raises(ValueError):
            relative_strength_index(data, 11)  # Data length must be greater than period
    
    def test_average_true_range(self):
        """Test average true range calculation."""
        # Test with a simple case
        high = [10, 11, 12, 11, 13]
        low = [8, 9, 10, 9, 11]
        close = [9, 10, 11, 10, 12]
        period = 3
        result = average_true_range(high, low, close, period)
        
        # Check NaN values
        for i in range(period):
            assert np.isnan(result[i])
        
        # Check that subsequent values are calculated
        for i in range(period, len(high)):
            assert not np.isnan(result[i])
            assert result[i] >= 0  # ATR should be non-negative
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            average_true_range(high, low, close, 0)  # Period must be positive
        
        with pytest.raises(ValueError):
            average_true_range(high, low, close, 6)  # Data length must be greater than period
        
        with pytest.raises(ValueError):
            average_true_range(high, low[:-1], close, 3)  # Arrays must have the same length
