#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for position sizing and risk management functions.
"""

import pytest
from src.risk.position_sizing import (
    calculate_position_size,
    calculate_stop_loss,
    calculate_take_profit
)


class TestPositionSizing:
    """Test suite for position sizing functions."""
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Test long position
        account_balance = 1000.0
        risk_per_trade = 0.02  # 2%
        entry_price = 100.0
        stop_loss = 95.0  # 5% stop loss
        
        # Expected position size: (1000 * 0.02) / (100 - 95) = 20 / 5 = 4 shares
        expected_size = 4.0
        
        result = calculate_position_size(
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        assert result == expected_size
        
        # Test short position
        entry_price = 100.0
        stop_loss = 105.0  # 5% stop loss for short
        
        # Expected position size: (1000 * 0.02) / (105 - 100) = 20 / 5 = 4 shares
        expected_size = 4.0
        
        result = calculate_position_size(
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        
        assert result == expected_size
        
        # Test with minimum position size
        result = calculate_position_size(
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            entry_price=entry_price,
            stop_loss=stop_loss,
            min_position_size=5.0
        )
        
        assert result == 5.0
        
        # Test with maximum position size
        result = calculate_position_size(
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            entry_price=entry_price,
            stop_loss=stop_loss,
            max_position_size=3.0
        )
        
        assert result == 3.0
        
        # Test with fractional shares disabled
        result = calculate_position_size(
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            entry_price=entry_price,
            stop_loss=stop_loss,
            allow_fractional_shares=False
        )
        
        assert result == 4.0  # Should be floored to 4
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            calculate_position_size(
                account_balance=0,
                risk_per_trade=risk_per_trade,
                entry_price=entry_price,
                stop_loss=stop_loss
            )
        
        with pytest.raises(ValueError):
            calculate_position_size(
                account_balance=account_balance,
                risk_per_trade=0,
                entry_price=entry_price,
                stop_loss=stop_loss
            )
        
        with pytest.raises(ValueError):
            calculate_position_size(
                account_balance=account_balance,
                risk_per_trade=risk_per_trade,
                entry_price=0,
                stop_loss=stop_loss
            )
        
        with pytest.raises(ValueError):
            calculate_position_size(
                account_balance=account_balance,
                risk_per_trade=risk_per_trade,
                entry_price=entry_price,
                stop_loss=0
            )
    
    def test_calculate_stop_loss(self):
        """Test stop loss calculation."""
        # Test ATR method for long position
        entry_price = 100.0
        atr_value = 2.0
        atr_multiplier = 2.0
        
        # Expected stop loss: 100 - (2 * 2) = 96
        expected_stop_loss = 96.0
        
        result = calculate_stop_loss(
            entry_price=entry_price,
            direction="long",
            method="atr",
            atr_value=atr_value,
            atr_multiplier=atr_multiplier
        )
        
        assert result == expected_stop_loss
        
        # Test ATR method for short position
        # Expected stop loss: 100 + (2 * 2) = 104
        expected_stop_loss = 104.0
        
        result = calculate_stop_loss(
            entry_price=entry_price,
            direction="short",
            method="atr",
            atr_value=atr_value,
            atr_multiplier=atr_multiplier
        )
        
        assert result == expected_stop_loss
        
        # Test percent method for long position
        percent = 0.05  # 5%
        
        # Expected stop loss: 100 * (1 - 0.05) = 95
        expected_stop_loss = 95.0
        
        result = calculate_stop_loss(
            entry_price=entry_price,
            direction="long",
            method="percent",
            percent=percent
        )
        
        assert result == expected_stop_loss
        
        # Test percent method for short position
        # Expected stop loss: 100 * (1 + 0.05) = 105
        expected_stop_loss = 105.0
        
        result = calculate_stop_loss(
            entry_price=entry_price,
            direction="short",
            method="percent",
            percent=percent
        )
        
        assert result == expected_stop_loss
        
        # Test support/resistance method for long position
        support_level = 95.0
        
        result = calculate_stop_loss(
            entry_price=entry_price,
            direction="long",
            method="support_resistance",
            support_resistance_level=support_level
        )
        
        assert result == support_level
        
        # Test support/resistance method for short position
        resistance_level = 105.0
        
        result = calculate_stop_loss(
            entry_price=entry_price,
            direction="short",
            method="support_resistance",
            support_resistance_level=resistance_level
        )
        
        assert result == resistance_level
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            calculate_stop_loss(
                entry_price=0,
                direction="long",
                method="atr",
                atr_value=atr_value
            )
        
        with pytest.raises(ValueError):
            calculate_stop_loss(
                entry_price=entry_price,
                direction="invalid",
                method="atr",
                atr_value=atr_value
            )
        
        with pytest.raises(ValueError):
            calculate_stop_loss(
                entry_price=entry_price,
                direction="long",
                method="invalid",
                atr_value=atr_value
            )
        
        with pytest.raises(ValueError):
            calculate_stop_loss(
                entry_price=entry_price,
                direction="long",
                method="atr",
                atr_value=None
            )
        
        with pytest.raises(ValueError):
            calculate_stop_loss(
                entry_price=entry_price,
                direction="long",
                method="support_resistance",
                support_resistance_level=None
            )
        
        with pytest.raises(ValueError):
            calculate_stop_loss(
                entry_price=entry_price,
                direction="long",
                method="support_resistance",
                support_resistance_level=105.0  # Support level above entry for long
            )
    
    def test_calculate_take_profit(self):
        """Test take profit calculation."""
        # Test for long position
        entry_price = 100.0
        stop_loss = 95.0  # 5 points risk
        risk_reward_ratio = 2.0
        
        # Expected take profit: 100 + (5 * 2) = 110
        expected_take_profit = 110.0
        
        result = calculate_take_profit(
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward_ratio
        )
        
        assert result == expected_take_profit
        
        # Test for short position
        entry_price = 100.0
        stop_loss = 105.0  # 5 points risk
        
        # Expected take profit: 100 - (5 * 2) = 90
        expected_take_profit = 90.0
        
        result = calculate_take_profit(
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward_ratio
        )
        
        assert result == expected_take_profit
        
        # Test with different risk-reward ratio
        risk_reward_ratio = 3.0
        
        # Expected take profit: 100 - (5 * 3) = 85
        expected_take_profit = 85.0
        
        result = calculate_take_profit(
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward_ratio
        )
        
        assert result == expected_take_profit
        
        # Test with invalid inputs
        with pytest.raises(ValueError):
            calculate_take_profit(
                entry_price=0,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward_ratio
            )
        
        with pytest.raises(ValueError):
            calculate_take_profit(
                entry_price=entry_price,
                stop_loss=0,
                risk_reward_ratio=risk_reward_ratio
            )
        
        with pytest.raises(ValueError):
            calculate_take_profit(
                entry_price=entry_price,
                stop_loss=stop_loss,
                risk_reward_ratio=0
            )
