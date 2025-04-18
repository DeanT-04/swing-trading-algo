#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Risk Manager Module

This module handles risk management for the day trading algorithm,
including position sizing, risk limits, and drawdown protection.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk Manager class for managing trading risk."""
    
    def __init__(self, config: Dict):
        """
        Initialize the Risk Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.risk_config = config.get("risk", {})
        
        # Default risk parameters
        self.max_position_pct = self.risk_config.get("max_position_pct", 0.1)  # 10% of balance
        self.max_total_risk_pct = self.risk_config.get("max_total_risk_pct", 0.5)  # 50% of balance
        self.max_drawdown_pct = self.risk_config.get("max_drawdown_pct", 0.1)  # 10% drawdown limit
        self.max_trades = self.risk_config.get("max_trades", 5)  # Maximum 5 concurrent trades
        
        # Track current risk exposure
        self.current_exposure = 0.0
        self.peak_balance = self.risk_config.get("initial_balance", 50.0)
        self.current_balance = self.peak_balance
        self.active_trades = {}
        
        logger.info(f"Risk Manager initialized with max position: {self.max_position_pct*100}% of balance")
    
    def calculate_position_size(self, symbol: str, price: float, balance: float) -> Tuple[float, float]:
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            symbol: Stock symbol
            price: Current price
            balance: Current account balance
            
        Returns:
            Tuple[float, float]: Position size and cost
        """
        # Calculate maximum amount to risk on this trade (10% of balance)
        max_risk_amount = balance * self.max_position_pct
        
        # Calculate position size
        if price > 0:
            # Calculate how many shares/contracts we can buy
            max_shares = max_risk_amount / price
            
            # For expensive stocks (>£50), allow partial shares
            if price > 50.0:
                # Round to 2 decimal places for partial shares
                position_size = round(max_shares, 2)
                if position_size < 0.1:  # Minimum position size
                    position_size = 0.1
            else:
                # For cheaper stocks, round down to whole shares
                position_size = int(max_shares)
                if position_size < 1:  # Ensure at least 1 share
                    position_size = 1
            
            # Calculate actual cost
            cost = position_size * price
            
            # Ensure we don't exceed max risk amount
            if cost > max_risk_amount:
                # Recalculate position size to stay within limits
                position_size = max_risk_amount / price
                if price > 50.0:
                    position_size = round(position_size, 2)
                else:
                    position_size = int(position_size)
                    if position_size < 1:
                        position_size = 1
                
                cost = position_size * price
            
            logger.info(f"Position size for {symbol} at £{price:.2f}: {position_size} shares, cost: £{cost:.2f}")
            return position_size, cost
        else:
            logger.warning(f"Invalid price {price} for {symbol}")
            return 0.0, 0.0
    
    def can_open_trade(self, symbol: str, price: float, direction: str, balance: float) -> Tuple[bool, float, float]:
        """
        Check if a new trade can be opened based on risk parameters.
        
        Args:
            symbol: Stock symbol
            price: Current price
            direction: Trade direction (LONG/SHORT)
            balance: Current account balance
            
        Returns:
            Tuple[bool, float, float]: (Can trade, position size, cost)
        """
        # Check if we've reached the maximum number of trades
        if len(self.active_trades) >= self.max_trades:
            logger.info(f"Maximum number of trades ({self.max_trades}) reached, cannot open new trade")
            return False, 0.0, 0.0
        
        # Check if we're already trading this symbol
        if symbol in [t.get("symbol") for t in self.active_trades.values()]:
            logger.info(f"Already trading {symbol}, cannot open another position")
            return False, 0.0, 0.0
        
        # Check if we have enough balance
        if balance <= 0:
            logger.warning(f"Insufficient balance (£{balance:.2f}) to open trade")
            return False, 0.0, 0.0
        
        # Calculate position size
        position_size, cost = self.calculate_position_size(symbol, price, balance)
        
        # Check if the cost is affordable
        if cost > balance:
            logger.info(f"Cost (£{cost:.2f}) exceeds available balance (£{balance:.2f})")
            return False, 0.0, 0.0
        
        # Check if adding this trade would exceed our total risk limit
        new_exposure = self.current_exposure + cost
        if new_exposure / self.peak_balance > self.max_total_risk_pct:
            logger.info(f"New exposure (£{new_exposure:.2f}) would exceed maximum risk limit")
            return False, 0.0, 0.0
        
        # All checks passed
        return True, position_size, cost
    
    def register_trade(self, trade_id: str, trade_data: Dict) -> None:
        """
        Register a new trade with the risk manager.
        
        Args:
            trade_id: Unique trade identifier
            trade_data: Trade data dictionary
        """
        self.active_trades[trade_id] = trade_data
        cost = trade_data.get("cost", trade_data.get("entry_price", 0.0) * trade_data.get("size", 0.0))
        self.current_exposure += cost
        self.current_balance -= cost
        
        logger.info(f"Registered trade {trade_id}: {trade_data.get('symbol')}, cost: £{cost:.2f}")
        logger.info(f"Current exposure: £{self.current_exposure:.2f} ({self.current_exposure/self.peak_balance*100:.1f}% of peak balance)")
    
    def unregister_trade(self, trade_id: str, profit_loss: float) -> None:
        """
        Unregister a trade with the risk manager.
        
        Args:
            trade_id: Unique trade identifier
            profit_loss: Profit/loss from the trade
        """
        if trade_id in self.active_trades:
            trade = self.active_trades.pop(trade_id)
            cost = trade.get("cost", trade.get("entry_price", 0.0) * trade.get("size", 0.0))
            self.current_exposure -= cost
            self.current_balance += cost + profit_loss
            
            # Update peak balance if we have a new high
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            
            logger.info(f"Unregistered trade {trade_id}: {trade.get('symbol')}, P/L: £{profit_loss:.2f}")
            logger.info(f"Current exposure: £{self.current_exposure:.2f} ({self.current_exposure/self.peak_balance*100:.1f}% of peak balance)")
            logger.info(f"Current balance: £{self.current_balance:.2f}, Peak balance: £{self.peak_balance:.2f}")
        else:
            logger.warning(f"Trade {trade_id} not found in active trades")
    
    def check_drawdown(self) -> bool:
        """
        Check if we've exceeded our maximum drawdown limit.
        
        Returns:
            bool: True if drawdown limit exceeded, False otherwise
        """
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            if drawdown > self.max_drawdown_pct:
                logger.warning(f"Maximum drawdown limit exceeded: {drawdown*100:.1f}% > {self.max_drawdown_pct*100:.1f}%")
                return True
        return False
    
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics.
        
        Returns:
            Dict: Risk metrics dictionary
        """
        drawdown = 0.0
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        return {
            "current_exposure": self.current_exposure,
            "exposure_pct": self.current_exposure / self.peak_balance if self.peak_balance > 0 else 0.0,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "drawdown": drawdown,
            "active_trades": len(self.active_trades),
            "max_trades": self.max_trades
        }
    
    def update_config(self, config: Dict) -> None:
        """
        Update risk configuration.
        
        Args:
            config: New configuration dictionary
        """
        self.config = config
        self.risk_config = config.get("risk", {})
        
        # Update risk parameters
        self.max_position_pct = self.risk_config.get("max_position_pct", self.max_position_pct)
        self.max_total_risk_pct = self.risk_config.get("max_total_risk_pct", self.max_total_risk_pct)
        self.max_drawdown_pct = self.risk_config.get("max_drawdown_pct", self.max_drawdown_pct)
        self.max_trades = self.risk_config.get("max_trades", self.max_trades)
        
        logger.info(f"Risk Manager configuration updated: max position: {self.max_position_pct*100}% of balance")
