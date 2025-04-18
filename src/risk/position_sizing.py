#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Position sizing and risk management functions.

This module provides functions to calculate appropriate position sizes
based on account balance, risk tolerance, and market conditions.
"""

from typing import Dict, Optional, Union
import numpy as np


def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float,
    min_position_size: Optional[float] = None,
    max_position_size: Optional[float] = None,
    allow_fractional_shares: bool = True
) -> float:
    """
    Calculate the appropriate position size based on risk parameters.
    
    Args:
        account_balance: Current account balance
        risk_per_trade: Maximum risk per trade as a decimal (e.g., 0.02 for 2%)
        entry_price: Planned entry price
        stop_loss: Planned stop loss price
        min_position_size: Minimum position size (optional)
        max_position_size: Maximum position size (optional)
        allow_fractional_shares: Whether to allow fractional shares
        
    Returns:
        float: Calculated position size (number of shares)
    """
    if account_balance <= 0:
        raise ValueError("Account balance must be positive")
    if risk_per_trade <= 0 or risk_per_trade > 0.1:
        raise ValueError("Risk per trade must be between 0 and 0.1 (0% to 10%)")
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")
    if stop_loss <= 0:
        raise ValueError("Stop loss must be positive")
    
    # Calculate risk amount in currency
    risk_amount = account_balance * risk_per_trade
    
    # Calculate risk per share
    if entry_price > stop_loss:  # Long position
        risk_per_share = entry_price - stop_loss
    else:  # Short position
        risk_per_share = stop_loss - entry_price
    
    if risk_per_share <= 0:
        raise ValueError("Risk per share must be positive")
    
    # Calculate position size
    position_size = risk_amount / risk_per_share
    
    # Apply minimum position size if specified
    if min_position_size is not None and position_size < min_position_size:
        position_size = min_position_size
    
    # Apply maximum position size if specified
    if max_position_size is not None and position_size > max_position_size:
        position_size = max_position_size
    
    # Round to whole shares if fractional shares are not allowed
    if not allow_fractional_shares:
        position_size = np.floor(position_size)
    
    return position_size


def calculate_stop_loss(
    entry_price: float,
    direction: str,
    method: str = "atr",
    atr_value: Optional[float] = None,
    atr_multiplier: float = 2.0,
    percent: float = 0.05,
    support_resistance_level: Optional[float] = None
) -> float:
    """
    Calculate the stop loss price based on the specified method.
    
    Args:
        entry_price: Entry price
        direction: Trade direction ('long' or 'short')
        method: Method to calculate stop loss ('atr', 'percent', or 'support_resistance')
        atr_value: ATR value (required if method is 'atr')
        atr_multiplier: Multiplier for ATR (default: 2.0)
        percent: Percentage for stop loss (default: 0.05 for 5%)
        support_resistance_level: Support/resistance level (required if method is 'support_resistance')
        
    Returns:
        float: Calculated stop loss price
    """
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")
    
    direction = direction.lower()
    if direction not in ["long", "short"]:
        raise ValueError("Direction must be 'long' or 'short'")
    
    method = method.lower()
    if method not in ["atr", "percent", "support_resistance"]:
        raise ValueError("Method must be 'atr', 'percent', or 'support_resistance'")
    
    if method == "atr":
        if atr_value is None or atr_value <= 0:
            raise ValueError("ATR value must be positive when using ATR method")
        
        if direction == "long":
            return entry_price - (atr_value * atr_multiplier)
        else:  # short
            return entry_price + (atr_value * atr_multiplier)
    
    elif method == "percent":
        if percent <= 0 or percent > 0.2:
            raise ValueError("Percent must be between 0 and 0.2 (0% to 20%)")
        
        if direction == "long":
            return entry_price * (1 - percent)
        else:  # short
            return entry_price * (1 + percent)
    
    elif method == "support_resistance":
        if support_resistance_level is None or support_resistance_level <= 0:
            raise ValueError("Support/resistance level must be positive when using support/resistance method")
        
        if direction == "long" and support_resistance_level >= entry_price:
            raise ValueError("Support level must be below entry price for long positions")
        if direction == "short" and support_resistance_level <= entry_price:
            raise ValueError("Resistance level must be above entry price for short positions")
        
        return support_resistance_level
    
    # This should never happen due to the validation above
    raise ValueError(f"Unsupported method: {method}")


def calculate_take_profit(
    entry_price: float,
    stop_loss: float,
    risk_reward_ratio: float = 2.0
) -> float:
    """
    Calculate the take profit price based on the risk-reward ratio.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_reward_ratio: Desired risk-reward ratio (default: 2.0)
        
    Returns:
        float: Calculated take profit price
    """
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")
    if stop_loss <= 0:
        raise ValueError("Stop loss must be positive")
    if risk_reward_ratio <= 0:
        raise ValueError("Risk-reward ratio must be positive")
    
    # Calculate risk (distance from entry to stop loss)
    risk = abs(entry_price - stop_loss)
    
    # Calculate reward (distance from entry to take profit)
    reward = risk * risk_reward_ratio
    
    # Calculate take profit price
    if entry_price > stop_loss:  # Long position
        take_profit = entry_price + reward
    else:  # Short position
        take_profit = entry_price - reward
    
    return take_profit


def calculate_risk_metrics(
    trades: Dict,
    initial_balance: float
) -> Dict:
    """
    Calculate risk metrics based on trade history.
    
    Args:
        trades: Dictionary of trade information
        initial_balance: Initial account balance
        
    Returns:
        Dict: Dictionary of risk metrics
    """
    if not trades:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "sharpe_ratio": 0.0
        }
    
    # Extract profit/loss values
    profits = [trade["profit_loss"] for trade in trades.values() if trade["profit_loss"] > 0]
    losses = [trade["profit_loss"] for trade in trades.values() if trade["profit_loss"] < 0]
    
    # Calculate win rate
    total_trades = len(trades)
    winning_trades = len(profits)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Calculate profit factor
    total_profit = sum(profits) if profits else 0.0
    total_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate average win and loss
    average_win = total_profit / winning_trades if winning_trades > 0 else 0.0
    average_loss = total_loss / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0.0
    
    # Calculate maximum drawdown
    balance_curve = [initial_balance]
    for trade in trades.values():
        balance_curve.append(balance_curve[-1] + trade["profit_loss"])
    
    max_balance = initial_balance
    max_drawdown = 0.0
    
    for balance in balance_curve:
        max_balance = max(max_balance, balance)
        drawdown = max_balance - balance
        max_drawdown = max(max_drawdown, drawdown)
    
    max_drawdown_percent = (max_drawdown / max_balance) * 100 if max_balance > 0 else 0.0
    
    # Calculate Sharpe ratio (simplified)
    returns = [trade["profit_loss"] / initial_balance for trade in trades.values()]
    mean_return = np.mean(returns) if returns else 0.0
    std_return = np.std(returns) if returns else 0.0
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
    
    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_win": average_win,
        "average_loss": average_loss,
        "max_drawdown": max_drawdown,
        "max_drawdown_percent": max_drawdown_percent,
        "sharpe_ratio": sharpe_ratio
    }
