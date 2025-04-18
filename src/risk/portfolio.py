#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Portfolio risk management module.

This module provides functionality for portfolio-level risk management,
including sector diversification and correlation analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from src.data.models import Stock, TimeFrame, Trade, TradeDirection

logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """
    Manages portfolio-level risk.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the portfolio risk manager with configuration parameters.
        
        Args:
            config: Portfolio risk manager configuration dictionary
        """
        self.config = config
        
        # Extract configuration parameters
        self.max_portfolio_risk = config.get("max_portfolio_risk", 0.05)  # 5% of portfolio
        self.max_sector_allocation = config.get("max_sector_allocation", 0.30)  # 30% per sector
        self.max_position_allocation = config.get("max_position_allocation", 0.20)  # 20% per position
        self.max_correlated_positions = config.get("max_correlated_positions", 2)
        self.correlation_threshold = config.get("correlation_threshold", 0.7)
        self.lookback_days = config.get("lookback_days", 90)
        
        logger.info(f"Initialized PortfolioRiskManager with max portfolio risk: {self.max_portfolio_risk}")
    
    def calculate_portfolio_risk(self, open_positions: List[Trade], stocks: Dict[str, Stock], timeframe: TimeFrame) -> float:
        """
        Calculate the current portfolio risk.
        
        Args:
            open_positions: List of open trade positions
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            
        Returns:
            float: Portfolio risk as a percentage of portfolio value
        """
        if not open_positions:
            return 0.0
        
        # Calculate portfolio value
        portfolio_value = sum(trade.position_size * trade.entry_price for trade in open_positions)
        
        if portfolio_value == 0:
            return 0.0
        
        # Calculate position weights
        position_weights = {}
        for trade in open_positions:
            position_value = trade.position_size * trade.entry_price
            position_weights[trade.symbol] = position_value / portfolio_value
        
        # Calculate returns for each position
        returns = {}
        for symbol, stock in stocks.items():
            if symbol not in position_weights:
                continue
            
            df = stock.get_dataframe(timeframe)
            if df.empty:
                continue
            
            # Calculate daily returns
            returns[symbol] = df['close'].pct_change().dropna()
        
        # Align returns by date
        returns_df = pd.DataFrame(returns)
        returns_df = returns_df.dropna(how='all')
        
        if returns_df.empty:
            return 0.0
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov()
        
        # Calculate portfolio variance
        portfolio_variance = 0.0
        for symbol1 in position_weights:
            if symbol1 not in cov_matrix:
                continue
            
            for symbol2 in position_weights:
                if symbol2 not in cov_matrix:
                    continue
                
                portfolio_variance += position_weights[symbol1] * position_weights[symbol2] * cov_matrix.loc[symbol1, symbol2]
        
        # Calculate portfolio risk (standard deviation)
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # Annualize risk (assuming 252 trading days per year)
        annualized_risk = portfolio_risk * np.sqrt(252)
        
        return annualized_risk
    
    def calculate_sector_allocations(self, open_positions: List[Trade], stocks: Dict[str, Stock]) -> Dict[str, float]:
        """
        Calculate the current sector allocations.
        
        Args:
            open_positions: List of open trade positions
            stocks: Dictionary of Stock objects by symbol
            
        Returns:
            Dict[str, float]: Sector allocations as percentages of portfolio value
        """
        if not open_positions:
            return {}
        
        # Calculate portfolio value
        portfolio_value = sum(trade.position_size * trade.entry_price for trade in open_positions)
        
        if portfolio_value == 0:
            return {}
        
        # Calculate sector allocations
        sector_allocations = {}
        for trade in open_positions:
            symbol = trade.symbol
            if symbol not in stocks or not stocks[symbol].sector:
                sector = "Unknown"
            else:
                sector = stocks[symbol].sector
            
            position_value = trade.position_size * trade.entry_price
            
            if sector not in sector_allocations:
                sector_allocations[sector] = 0.0
            
            sector_allocations[sector] += position_value / portfolio_value
        
        return sector_allocations
    
    def calculate_correlation_matrix(self, stocks: Dict[str, Stock], timeframe: TimeFrame) -> pd.DataFrame:
        """
        Calculate the correlation matrix for the stocks.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Calculate returns for each stock
        returns = {}
        for symbol, stock in stocks.items():
            df = stock.get_dataframe(timeframe)
            if df.empty:
                continue
            
            # Calculate daily returns
            returns[symbol] = df['close'].pct_change().dropna()
        
        # Align returns by date
        returns_df = pd.DataFrame(returns)
        returns_df = returns_df.dropna(how='all')
        
        if returns_df.empty:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def find_correlated_pairs(self, stocks: Dict[str, Stock], timeframe: TimeFrame) -> List[Tuple[str, str, float]]:
        """
        Find pairs of stocks with correlation above the threshold.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            
        Returns:
            List[Tuple[str, str, float]]: List of correlated pairs with their correlation coefficient
        """
        correlation_matrix = self.calculate_correlation_matrix(stocks, timeframe)
        
        if correlation_matrix.empty:
            return []
        
        # Find pairs with correlation above threshold
        correlated_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                symbol1 = correlation_matrix.columns[i]
                symbol2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]
                
                if abs(correlation) >= self.correlation_threshold:
                    correlated_pairs.append((symbol1, symbol2, correlation))
        
        # Sort by correlation (descending)
        correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return correlated_pairs
    
    def check_position_risk(self, trade: Trade, open_positions: List[Trade], stocks: Dict[str, Stock], timeframe: TimeFrame) -> Dict:
        """
        Check if a new trade would violate portfolio risk constraints.
        
        Args:
            trade: Proposed new trade
            open_positions: List of current open positions
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            
        Returns:
            Dict: Risk check results
        """
        # Calculate current portfolio value
        current_portfolio_value = sum(pos.position_size * pos.entry_price for pos in open_positions)
        
        # Calculate new position value
        new_position_value = trade.position_size * trade.entry_price
        
        # Calculate new portfolio value
        new_portfolio_value = current_portfolio_value + new_position_value
        
        if new_portfolio_value == 0:
            return {"allowed": False, "reason": "Invalid portfolio value"}
        
        # Check position allocation
        position_allocation = new_position_value / new_portfolio_value
        if position_allocation > self.max_position_allocation:
            return {
                "allowed": False,
                "reason": f"Position allocation ({position_allocation:.2%}) exceeds maximum ({self.max_position_allocation:.2%})"
            }
        
        # Check sector allocation
        symbol = trade.symbol
        if symbol in stocks and stocks[symbol].sector:
            sector = stocks[symbol].sector
        else:
            sector = "Unknown"
        
        # Calculate current sector allocation
        current_sector_allocation = 0.0
        for pos in open_positions:
            pos_symbol = pos.symbol
            if pos_symbol in stocks and stocks[pos_symbol].sector == sector:
                current_sector_allocation += pos.position_size * pos.entry_price
        
        # Calculate new sector allocation
        new_sector_allocation = (current_sector_allocation + new_position_value) / new_portfolio_value
        if new_sector_allocation > self.max_sector_allocation:
            return {
                "allowed": False,
                "reason": f"Sector allocation for {sector} ({new_sector_allocation:.2%}) exceeds maximum ({self.max_sector_allocation:.2%})"
            }
        
        # Check correlation with existing positions
        if len(open_positions) > 0:
            correlation_matrix = self.calculate_correlation_matrix(stocks, timeframe)
            
            if not correlation_matrix.empty and symbol in correlation_matrix.columns:
                correlated_positions = 0
                
                for pos in open_positions:
                    pos_symbol = pos.symbol
                    if pos_symbol in correlation_matrix.columns:
                        correlation = correlation_matrix.loc[symbol, pos_symbol]
                        
                        # Check if correlation is above threshold
                        if abs(correlation) >= self.correlation_threshold:
                            correlated_positions += 1
                
                if correlated_positions >= self.max_correlated_positions:
                    return {
                        "allowed": False,
                        "reason": f"Too many correlated positions ({correlated_positions} >= {self.max_correlated_positions})"
                    }
        
        # Calculate new portfolio risk
        new_positions = open_positions + [trade]
        new_portfolio_risk = self.calculate_portfolio_risk(new_positions, stocks, timeframe)
        
        if new_portfolio_risk > self.max_portfolio_risk:
            return {
                "allowed": False,
                "reason": f"Portfolio risk ({new_portfolio_risk:.2%}) exceeds maximum ({self.max_portfolio_risk:.2%})"
            }
        
        return {"allowed": True}
    
    def optimize_position_size(self, trade: Trade, open_positions: List[Trade], stocks: Dict[str, Stock], timeframe: TimeFrame) -> float:
        """
        Optimize the position size to meet portfolio risk constraints.
        
        Args:
            trade: Proposed trade
            open_positions: List of current open positions
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            
        Returns:
            float: Optimized position size
        """
        # Start with the proposed position size
        position_size = trade.position_size
        
        # Calculate current portfolio value
        current_portfolio_value = sum(pos.position_size * pos.entry_price for pos in open_positions)
        
        # Calculate maximum position value based on position allocation constraint
        max_position_value = (current_portfolio_value / (1 - self.max_position_allocation)) * self.max_position_allocation
        
        # Calculate maximum position size
        max_position_size = max_position_value / trade.entry_price
        
        # Adjust position size if needed
        if position_size > max_position_size:
            position_size = max_position_size
        
        # Check sector allocation
        symbol = trade.symbol
        if symbol in stocks and stocks[symbol].sector:
            sector = stocks[symbol].sector
        else:
            sector = "Unknown"
        
        # Calculate current sector allocation
        current_sector_allocation = 0.0
        for pos in open_positions:
            pos_symbol = pos.symbol
            if pos_symbol in stocks and stocks[pos_symbol].sector == sector:
                current_sector_allocation += pos.position_size * pos.entry_price
        
        # Calculate maximum sector allocation
        max_sector_value = (current_portfolio_value / (1 - self.max_sector_allocation)) * self.max_sector_allocation
        
        # Calculate maximum position value based on sector allocation
        max_sector_position_value = max_sector_value - current_sector_allocation
        
        # Calculate maximum position size based on sector allocation
        max_sector_position_size = max_sector_position_value / trade.entry_price
        
        # Adjust position size if needed
        if position_size > max_sector_position_size:
            position_size = max_sector_position_size
        
        # Binary search to find the maximum position size that meets portfolio risk constraint
        if len(open_positions) > 0:
            # Create a copy of the trade with the current position size
            test_trade = Trade(
                symbol=trade.symbol,
                direction=trade.direction,
                entry_time=trade.entry_time,
                entry_price=trade.entry_price,
                position_size=position_size,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit
            )
            
            # Check if the current position size meets the portfolio risk constraint
            new_positions = open_positions + [test_trade]
            new_portfolio_risk = self.calculate_portfolio_risk(new_positions, stocks, timeframe)
            
            if new_portfolio_risk > self.max_portfolio_risk:
                # Binary search to find the maximum position size
                min_size = 0.0
                max_size = position_size
                
                while max_size - min_size > 0.01:
                    mid_size = (min_size + max_size) / 2
                    
                    # Create a test trade with the mid size
                    test_trade.position_size = mid_size
                    
                    # Check portfolio risk
                    new_positions = open_positions + [test_trade]
                    new_portfolio_risk = self.calculate_portfolio_risk(new_positions, stocks, timeframe)
                    
                    if new_portfolio_risk <= self.max_portfolio_risk:
                        min_size = mid_size
                    else:
                        max_size = mid_size
                
                position_size = min_size
        
        return position_size
