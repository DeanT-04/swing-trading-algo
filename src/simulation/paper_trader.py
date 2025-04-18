#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Paper trading module.

This module provides functionality to simulate trades in real-time
without actually executing them in the market.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

from src.data.models import Account, Stock, TimeFrame, Trade, TradeDirection
from src.risk.position_sizing import calculate_position_size

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Simulates trades in real-time without actually executing them.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the paper trader with configuration parameters.
        
        Args:
            config: Paper trader configuration dictionary
        """
        self.config = config
        
        # Extract configuration parameters
        self.initial_balance = config.get("initial_balance", 1000.0)
        self.currency = config.get("currency", "GBP")
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        self.max_open_positions = config.get("max_open_positions", 3)
        self.slippage = config.get("slippage", 0.001)
        self.commission = config.get("commission", 0.002)
        self.enable_fractional_shares = config.get("enable_fractional_shares", True)
        self.state_file = config.get("state_file", "paper_trader_state.json")
        
        # Initialize account
        self.account = Account(
            initial_balance=self.initial_balance,
            currency=self.currency
        )
        
        # Track trades by symbol
        self.open_trades_by_symbol = {}
        
        # Load state if available
        self._load_state()
        
        logger.info(f"Initialized PaperTrader with initial balance: {self.initial_balance} {self.currency}")
    
    def process_signal(self, signal: Dict, stock: Stock, current_price: float, current_time: datetime) -> Optional[Trade]:
        """
        Process a trading signal and execute a paper trade if appropriate.
        
        Args:
            signal: Signal dictionary
            stock: Stock object
            current_price: Current price of the stock
            current_time: Current time
            
        Returns:
            Optional[Trade]: Executed trade or None if no trade was executed
        """
        symbol = stock.symbol
        
        # Check if we already have an open position for this symbol
        if symbol in self.open_trades_by_symbol:
            logger.info(f"Already have an open position for {symbol}, skipping signal")
            return None
        
        # Check if we have reached the maximum number of open positions
        if len(self.open_trades_by_symbol) >= self.max_open_positions:
            logger.info(f"Maximum number of open positions reached ({self.max_open_positions}), skipping signal")
            return None
        
        # Apply slippage to entry price
        direction = signal["direction"]
        entry_price = signal["entry_price"]
        
        if direction == TradeDirection.LONG:
            # For long positions, slippage increases the entry price
            adjusted_entry_price = entry_price * (1 + self.slippage)
        else:  # SHORT
            # For short positions, slippage decreases the entry price
            adjusted_entry_price = entry_price * (1 - self.slippage)
        
        # Calculate position size
        position_size = calculate_position_size(
            account_balance=self.account.current_balance,
            risk_per_trade=self.risk_per_trade,
            entry_price=adjusted_entry_price,
            stop_loss=signal["stop_loss"],
            allow_fractional_shares=self.enable_fractional_shares
        )
        
        # Check if position size is valid
        if position_size <= 0:
            logger.warning(f"Calculated position size is invalid: {position_size}, skipping signal")
            return None
        
        # Calculate commission
        commission_amount = adjusted_entry_price * position_size * self.commission
        
        # Check if we have enough balance for the trade including commission
        trade_cost = adjusted_entry_price * position_size + commission_amount
        if trade_cost > self.account.current_balance:
            logger.warning(f"Insufficient balance for trade: {trade_cost} > {self.account.current_balance}, skipping signal")
            return None
        
        # Create trade object
        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_time=current_time,
            entry_price=adjusted_entry_price,
            position_size=position_size,
            stop_loss=signal["stop_loss"],
            take_profit=signal["take_profit"]
        )
        
        # Open the trade
        try:
            self.account.open_trade(trade)
            self.open_trades_by_symbol[symbol] = trade
            
            # Deduct commission
            self.account.current_balance -= commission_amount
            
            logger.info(f"Opened {direction.value} trade for {symbol} at {adjusted_entry_price} with position size {position_size}")
            
            # Save state
            self._save_state()
            
            return trade
        
        except ValueError as e:
            logger.error(f"Error opening trade: {e}")
            return None
    
    def update_positions(self, stocks: Dict[str, Stock], current_time: datetime) -> List[Trade]:
        """
        Update open positions based on current prices and check for exits.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
            current_time: Current time
            
        Returns:
            List[Trade]: List of closed trades
        """
        closed_trades = []
        
        # Make a copy of the keys to avoid modifying the dictionary during iteration
        symbols = list(self.open_trades_by_symbol.keys())
        
        for symbol in symbols:
            if symbol not in stocks:
                logger.warning(f"Stock {symbol} not found in stocks dictionary, skipping update")
                continue
            
            stock = stocks[symbol]
            trade = self.open_trades_by_symbol[symbol]
            
            # Get the latest price data
            df = stock.get_dataframe(TimeFrame.DAILY)
            if df.empty:
                logger.warning(f"No price data available for {symbol}, skipping update")
                continue
            
            # Get the latest prices
            latest_data = df.iloc[-1]
            current_price = latest_data["close"]
            high_price = latest_data["high"]
            low_price = latest_data["low"]
            
            # Check for stop loss hit
            stop_loss_hit = False
            if trade.direction == TradeDirection.LONG and low_price <= trade.stop_loss:
                stop_loss_hit = True
                exit_price = max(trade.stop_loss, low_price)  # Use stop loss or low, whichever is higher
                exit_reason = "Stop loss hit"
            elif trade.direction == TradeDirection.SHORT and high_price >= trade.stop_loss:
                stop_loss_hit = True
                exit_price = min(trade.stop_loss, high_price)  # Use stop loss or high, whichever is lower
                exit_reason = "Stop loss hit"
            
            # Check for take profit hit
            take_profit_hit = False
            if trade.direction == TradeDirection.LONG and high_price >= trade.take_profit:
                take_profit_hit = True
                exit_price = min(trade.take_profit, high_price)  # Use take profit or high, whichever is lower
                exit_reason = "Take profit hit"
            elif trade.direction == TradeDirection.SHORT and low_price <= trade.take_profit:
                take_profit_hit = True
                exit_price = max(trade.take_profit, low_price)  # Use take profit or low, whichever is higher
                exit_reason = "Take profit hit"
            
            # Close the trade if stop loss or take profit was hit
            if stop_loss_hit or take_profit_hit:
                # Apply slippage to exit price
                if trade.direction == TradeDirection.LONG:
                    # For long positions, slippage decreases the exit price
                    adjusted_exit_price = exit_price * (1 - self.slippage)
                else:  # SHORT
                    # For short positions, slippage increases the exit price
                    adjusted_exit_price = exit_price * (1 + self.slippage)
                
                # Calculate commission
                commission_amount = adjusted_exit_price * trade.position_size * self.commission
                
                # Close the trade
                self.account.close_trade(trade, adjusted_exit_price, current_time, exit_reason)
                
                # Deduct commission
                self.account.current_balance -= commission_amount
                
                # Remove from open trades
                del self.open_trades_by_symbol[symbol]
                
                logger.info(f"Closed {trade.direction.value} trade for {symbol} at {adjusted_exit_price} ({exit_reason})")
                closed_trades.append(trade)
        
        # Save state if trades were closed
        if closed_trades:
            self._save_state()
        
        return closed_trades
    
    def get_account_summary(self) -> Dict:
        """
        Get a summary of the current account status.
        
        Returns:
            Dict: Account summary
        """
        return {
            "initial_balance": self.account.initial_balance,
            "current_balance": self.account.current_balance,
            "profit_loss": self.account.profit_loss,
            "profit_loss_percent": self.account.profit_loss_percent,
            "open_positions": len(self.open_trades_by_symbol),
            "total_trades": len(self.account.trades),
            "currency": self.account.currency
        }
    
    def get_trade_history(self) -> List[Dict]:
        """
        Get the history of all trades.
        
        Returns:
            List[Dict]: List of trade dictionaries
        """
        trade_history = []
        
        for trade in self.account.trades:
            trade_dict = {
                "symbol": trade.symbol,
                "direction": trade.direction.value,
                "entry_time": trade.entry_time,
                "entry_price": trade.entry_price,
                "position_size": trade.position_size,
                "stop_loss": trade.stop_loss,
                "take_profit": trade.take_profit,
                "exit_time": trade.exit_time,
                "exit_price": trade.exit_price,
                "exit_reason": trade.exit_reason,
                "duration": trade.duration,
                "profit_loss": trade.profit_loss,
                "profit_loss_percent": trade.profit_loss_percent,
                "is_open": trade.is_open
            }
            trade_history.append(trade_dict)
        
        return trade_history
    
    def _save_state(self):
        """Save the current state to a file."""
        try:
            state = {
                "account": {
                    "initial_balance": self.account.initial_balance,
                    "current_balance": self.account.current_balance,
                    "currency": self.account.currency
                },
                "trades": self.get_trade_history(),
                "open_trades": list(self.open_trades_by_symbol.keys())
            }
            
            with open(self.state_file, 'w') as file:
                json.dump(state, file, indent=4, default=str)
            
            logger.debug(f"State saved to {self.state_file}")
        
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _load_state(self):
        """Load the state from a file."""
        try:
            state_path = Path(self.state_file)
            if not state_path.exists():
                logger.info(f"State file {self.state_file} not found, using default state")
                return
            
            with open(state_path, 'r') as file:
                state = json.load(file)
            
            # Restore account balance
            self.account.current_balance = state["account"]["current_balance"]
            
            logger.info(f"State loaded from {self.state_file}")
            logger.info(f"Current balance: {self.account.current_balance} {self.account.currency}")
            logger.info(f"Total trades: {len(state['trades'])}")
            logger.info(f"Open trades: {len(state['open_trades'])}")
        
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            logger.info("Using default state")
