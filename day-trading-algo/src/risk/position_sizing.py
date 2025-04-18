#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Position sizing module for the day trading algorithm.

This module handles position sizing and risk management calculations.
"""

import logging
from typing import Dict, Optional, Union
import math

from src.data.models import TradeDirection

# Set up logging
logger = logging.getLogger(__name__)


def calculate_position_size(account_balance: float, risk_per_trade: float, 
                          entry_price: float, stop_loss: float, 
                          direction: Union[TradeDirection, str]) -> float:
    """
    Calculate position size based on risk parameters.
    
    Args:
        account_balance: Current account balance
        risk_per_trade: Risk per trade as a decimal (e.g., 0.01 for 1%)
        entry_price: Entry price
        stop_loss: Stop loss price
        direction: Trade direction (long or short)
        
    Returns:
        float: Position size in number of shares/contracts
    """
    # Convert direction to TradeDirection enum if it's a string
    if isinstance(direction, str):
        direction = TradeDirection(direction)
    
    # Calculate risk amount in currency
    risk_amount = account_balance * risk_per_trade
    
    # Calculate risk per share
    if direction == TradeDirection.LONG:
        risk_per_share = entry_price - stop_loss
    else:  # SHORT
        risk_per_share = stop_loss - entry_price
    
    # Avoid division by zero
    if risk_per_share <= 0:
        logger.warning(f"Invalid risk per share: {risk_per_share}. Using minimum risk.")
        risk_per_share = 0.01
    
    # Calculate position size
    position_size = risk_amount / risk_per_share
    
    # Round down to avoid exceeding risk
    position_size = math.floor(position_size * 100) / 100
    
    logger.info(f"Calculated position size: {position_size} shares/contracts")
    return position_size


def calculate_stop_loss(entry_price: float, direction: Union[TradeDirection, str], 
                      method: str = "percent", percent: float = 1.0, 
                      atr_value: Optional[float] = None, 
                      atr_multiplier: float = 2.0) -> float:
    """
    Calculate stop loss price based on various methods.
    
    Args:
        entry_price: Entry price
        direction: Trade direction (long or short)
        method: Method to calculate stop loss (percent, atr)
        percent: Percentage for stop loss if method is percent
        atr_value: ATR value if method is atr
        atr_multiplier: Multiplier for ATR if method is atr
        
    Returns:
        float: Stop loss price
    """
    # Convert direction to TradeDirection enum if it's a string
    if isinstance(direction, str):
        direction = TradeDirection(direction)
    
    if method == "percent":
        # Calculate stop loss based on percentage
        if direction == TradeDirection.LONG:
            stop_loss = entry_price * (1 - percent / 100)
        else:  # SHORT
            stop_loss = entry_price * (1 + percent / 100)
    
    elif method == "atr" and atr_value is not None:
        # Calculate stop loss based on ATR
        if direction == TradeDirection.LONG:
            stop_loss = entry_price - (atr_value * atr_multiplier)
        else:  # SHORT
            stop_loss = entry_price + (atr_value * atr_multiplier)
    
    else:
        # Default to 1% stop loss
        if direction == TradeDirection.LONG:
            stop_loss = entry_price * 0.99
        else:  # SHORT
            stop_loss = entry_price * 1.01
    
    # Round to 4 decimal places
    stop_loss = round(stop_loss, 4)
    
    logger.info(f"Calculated stop loss: {stop_loss}")
    return stop_loss


def calculate_take_profit(entry_price: float, stop_loss: float, 
                        risk_reward_ratio: float = 2.0) -> float:
    """
    Calculate take profit price based on risk-reward ratio.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_reward_ratio: Risk-reward ratio
        
    Returns:
        float: Take profit price
    """
    # Calculate risk
    risk = abs(entry_price - stop_loss)
    
    # Calculate reward
    reward = risk * risk_reward_ratio
    
    # Calculate take profit
    if entry_price > stop_loss:  # LONG
        take_profit = entry_price + reward
    else:  # SHORT
        take_profit = entry_price - reward
    
    # Round to 4 decimal places
    take_profit = round(take_profit, 4)
    
    logger.info(f"Calculated take profit: {take_profit}")
    return take_profit


def calculate_risk_metrics(account_balance: float, open_positions: Dict) -> Dict:
    """
    Calculate risk metrics for the current portfolio.
    
    Args:
        account_balance: Current account balance
        open_positions: Dictionary of open positions
        
    Returns:
        Dict: Risk metrics
    """
    total_risk = 0.0
    max_drawdown = 0.0
    
    for symbol, position in open_positions.items():
        # Calculate risk for this position
        if position.direction == TradeDirection.LONG:
            risk = (position.entry_price - position.stop_loss) * position.size
        else:  # SHORT
            risk = (position.stop_loss - position.entry_price) * position.size
        
        # Add to total risk
        total_risk += risk
        
        # Update max drawdown
        max_drawdown = max(max_drawdown, risk)
    
    # Calculate risk metrics
    risk_percent = (total_risk / account_balance) * 100 if account_balance > 0 else 0
    max_drawdown_percent = (max_drawdown / account_balance) * 100 if account_balance > 0 else 0
    
    return {
        "total_risk": total_risk,
        "risk_percent": risk_percent,
        "max_drawdown": max_drawdown,
        "max_drawdown_percent": max_drawdown_percent,
        "num_positions": len(open_positions)
    }
