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
                          direction: Union[TradeDirection, str],
                          max_position_pct: float = 0.1) -> float:
    """
    Calculate position size based on risk parameters.

    Args:
        account_balance: Current account balance
        risk_per_trade: Risk per trade as a decimal (e.g., 0.01 for 1%)
        entry_price: Entry price
        stop_loss: Stop loss price
        direction: Trade direction (long or short)
        max_position_pct: Maximum position size as percentage of account balance (default: 10%)

    Returns:
        float: Position size in number of shares/contracts
    """
    # Convert direction to TradeDirection enum if it's a string
    if isinstance(direction, str):
        direction = TradeDirection(direction)

    # Calculate risk amount in currency (limited to max_position_pct)
    risk_amount = min(account_balance * risk_per_trade, account_balance * max_position_pct)

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

    # Calculate maximum affordable position size based on max_position_pct
    max_affordable = (account_balance * max_position_pct) / entry_price

    # Take the smaller of the two position sizes
    position_size = min(position_size, max_affordable)

    # For expensive stocks (>£50), allow partial shares
    if entry_price > 50.0:
        # Round to 2 decimal places for partial shares
        position_size = round(position_size, 2)
        if position_size < 0.1:  # Minimum position size
            position_size = 0.1
    else:
        # For cheaper stocks, round down to whole shares
        position_size = math.floor(position_size)
        if position_size < 1:  # Ensure at least 1 share
            position_size = 1

    # Calculate actual cost
    cost = position_size * entry_price

    # Ensure we don't exceed max position percentage
    if cost > account_balance * max_position_pct:
        # Recalculate position size to stay within limits
        position_size = (account_balance * max_position_pct) / entry_price
        if entry_price > 50.0:
            position_size = round(position_size, 2)
        else:
            position_size = math.floor(position_size)
            if position_size < 1:
                position_size = 1

    logger.info(f"Calculated position size: {position_size} shares/contracts (max: {max_position_pct*100}% of balance)")
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


def calculate_risk_metrics(account_balance: float, open_positions: Dict, peak_balance: float = None) -> Dict:
    """
    Calculate risk metrics for the current portfolio.

    Args:
        account_balance: Current account balance
        open_positions: Dictionary of open positions
        peak_balance: Peak account balance (for drawdown calculation)

    Returns:
        Dict: Risk metrics
    """
    total_risk = 0.0
    total_exposure = 0.0
    max_position_risk = 0.0

    # Use peak_balance for drawdown calculation, or current balance if not provided
    if peak_balance is None or peak_balance <= 0:
        peak_balance = account_balance

    # Calculate current drawdown
    current_drawdown = peak_balance - account_balance
    drawdown_percent = (current_drawdown / peak_balance) * 100 if peak_balance > 0 else 0

    for position_id, position in open_positions.items():
        # Calculate position value
        position_value = position.get("current_price", position.get("entry_price", 0)) * position.get("size", 0)
        total_exposure += position_value

        # Calculate risk for this position
        if hasattr(position, "direction") and hasattr(position, "stop_loss"):
            # For position objects with direction and stop_loss attributes
            if position.direction == TradeDirection.LONG:
                risk = (position.entry_price - position.stop_loss) * position.size
            else:  # SHORT
                risk = (position.stop_loss - position.entry_price) * position.size
        else:
            # For dictionary positions, estimate risk as 2% of position value
            risk = position_value * 0.02

        # Add to total risk
        total_risk += risk

        # Update max position risk
        max_position_risk = max(max_position_risk, risk)

    # Calculate risk metrics
    risk_percent = (total_risk / account_balance) * 100 if account_balance > 0 else 0
    exposure_percent = (total_exposure / account_balance) * 100 if account_balance > 0 else 0
    max_position_risk_percent = (max_position_risk / account_balance) * 100 if account_balance > 0 else 0

    return {
        "total_risk": total_risk,
        "risk_percent": risk_percent,
        "total_exposure": total_exposure,
        "exposure_percent": exposure_percent,
        "max_position_risk": max_position_risk,
        "max_position_risk_percent": max_position_risk_percent,
        "current_drawdown": current_drawdown,
        "drawdown_percent": drawdown_percent,
        "num_positions": len(open_positions),
        "account_balance": account_balance,
        "peak_balance": peak_balance
    }
