#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time data module.

This module provides functionality to retrieve real-time market data
and update the trading system accordingly.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable

import yfinance as yf
import pandas as pd

from src.data.models import OHLCV, Stock, TimeFrame

logger = logging.getLogger(__name__)


class RealTimeDataManager:
    """
    Manages real-time data retrieval and processing.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the real-time data manager with configuration parameters.
        
        Args:
            config: Real-time data configuration dictionary
        """
        self.config = config
        
        # Extract configuration parameters
        self.symbols = config.get("symbols", [])
        self.timeframes = [TimeFrame.from_string(tf) for tf in config.get("timeframes", ["daily"])]
        self.update_interval = config.get("update_interval", 60)  # seconds
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5)  # seconds
        
        # Initialize data storage
        self.stocks = {}
        self.last_update = {}
        self.callbacks = []
        
        # Initialize threading
        self.running = False
        self.thread = None
        
        logger.info(f"Initialized RealTimeDataManager with {len(self.symbols)} symbols")
    
    def start(self):
        """Start the real-time data manager."""
        if self.running:
            logger.warning("Real-time data manager is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Real-time data manager started")
    
    def stop(self):
        """Stop the real-time data manager."""
        if not self.running:
            logger.warning("Real-time data manager is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info("Real-time data manager stopped")
    
    def add_callback(self, callback: Callable[[Dict[str, Stock]], None]):
        """
        Add a callback function to be called when data is updated.
        
        Args:
            callback: Callback function that takes a dictionary of Stock objects
        """
        self.callbacks.append(callback)
        logger.info(f"Added callback: {callback.__name__}")
    
    def get_stocks(self) -> Dict[str, Stock]:
        """
        Get the current stocks data.
        
        Returns:
            Dict[str, Stock]: Dictionary of Stock objects by symbol
        """
        return self.stocks.copy()
    
    def _update_loop(self):
        """Update loop for real-time data."""
        while self.running:
            try:
                self._update_data()
                
                # Call callbacks
                for callback in self.callbacks:
                    try:
                        callback(self.stocks)
                    except Exception as e:
                        logger.error(f"Error in callback {callback.__name__}: {e}")
                
                # Sleep until next update
                time.sleep(self.update_interval)
            
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(self.retry_delay)
    
    def _update_data(self):
        """Update data for all symbols and timeframes."""
        current_time = datetime.now()
        
        for symbol in self.symbols:
            # Skip if updated recently
            if symbol in self.last_update and (current_time - self.last_update[symbol]).total_seconds() < self.update_interval:
                continue
            
            # Initialize stock if not exists
            if symbol not in self.stocks:
                self.stocks[symbol] = Stock(symbol=symbol)
            
            # Update data for each timeframe
            for timeframe in self.timeframes:
                self._update_symbol_data(symbol, timeframe)
            
            # Update last update time
            self.last_update[symbol] = current_time
    
    def _update_symbol_data(self, symbol: str, timeframe: TimeFrame):
        """
        Update data for a specific symbol and timeframe.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe for data
        """
        # Map timeframe to yfinance interval
        interval_map = {
            TimeFrame.DAILY: "1d",
            TimeFrame.FOUR_HOUR: "1h",  # yfinance doesn't have 4h, use 1h and aggregate later
            TimeFrame.ONE_HOUR: "1h"
        }
        
        interval = interval_map.get(timeframe, "1d")
        
        # Calculate start date based on timeframe
        end_date = datetime.now()
        if timeframe == TimeFrame.DAILY:
            start_date = end_date - timedelta(days=5)
        elif timeframe == TimeFrame.FOUR_HOUR:
            start_date = end_date - timedelta(days=2)
        elif timeframe == TimeFrame.ONE_HOUR:
            start_date = end_date - timedelta(days=1)
        else:
            start_date = end_date - timedelta(days=1)
        
        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                # Download data
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    progress=False
                )
                
                if data.empty:
                    logger.warning(f"No data found for {symbol} with interval {interval}")
                    return
                
                # Aggregate to 4h if needed
                if timeframe == TimeFrame.FOUR_HOUR and interval == "1h":
                    data = data.resample('4H').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
                
                # Process the data
                for index, row in data.iterrows():
                    # Convert index to datetime if it's not already
                    if not isinstance(index, datetime):
                        timestamp = pd.to_datetime(index)
                    else:
                        timestamp = index
                    
                    # Create OHLCV object
                    try:
                        ohlcv = OHLCV(
                            timestamp=timestamp,
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=int(row["Volume"])
                        )
                        
                        # Add to stock data
                        self.stocks[symbol].add_data_point(timeframe, ohlcv)
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error processing data point for {symbol}: {e}")
                        continue
                
                logger.debug(f"Updated data for {symbol} with timeframe {timeframe.value}")
                return
            
            except Exception as e:
                logger.error(f"Error updating data for {symbol} with timeframe {timeframe.value} (attempt {attempt+1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay)
        
        logger.error(f"Failed to update data for {symbol} with timeframe {timeframe.value} after {self.max_retries} attempts")


class RealTimeTrader:
    """
    Real-time trader that processes data updates and executes trades.
    """
    
    def __init__(self, config: Dict, strategy, risk_manager, simulator):
        """
        Initialize the real-time trader with configuration parameters.
        
        Args:
            config: Real-time trader configuration dictionary
            strategy: Trading strategy object
            risk_manager: Risk management object
            simulator: Trade simulator object
        """
        self.config = config
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.simulator = simulator
        
        # Extract configuration parameters
        self.timeframe = TimeFrame.from_string(config.get("timeframe", "daily"))
        self.enable_trading = config.get("enable_trading", False)
        
        logger.info(f"Initialized RealTimeTrader with timeframe {self.timeframe.value}")
    
    def process_data_update(self, stocks: Dict[str, Stock]):
        """
        Process data update and execute trades if necessary.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
        """
        if not self.enable_trading:
            logger.info("Trading is disabled")
            return
        
        current_time = datetime.now()
        
        # Analyze each stock and generate signals
        for symbol, stock in stocks.items():
            logger.info(f"Analyzing {symbol}")
            
            # Analyze the stock
            results = self.strategy.analyze(stock, self.timeframe)
            
            if not results or "signals" not in results or not results["signals"]:
                logger.info(f"No signals generated for {symbol}")
                continue
            
            logger.info(f"Generated {len(results['signals'])} signals for {symbol}")
            
            # Process the most recent signal
            signal = results["signals"][-1]
            
            # Add symbol to signal
            signal["symbol"] = symbol
            
            # Get the timestamp and corresponding price data
            timestamp = signal["timestamp"]
            df = stock.get_dataframe(self.timeframe)
            if timestamp not in df.index:
                logger.warning(f"Timestamp {timestamp} not found in data for {symbol}")
                continue
            
            # Check if the signal is recent
            signal_time = timestamp
            if isinstance(signal_time, pd.Timestamp):
                signal_time = signal_time.to_pydatetime()
            
            # Only process signals from the last update interval
            if (current_time - signal_time).total_seconds() > 3600:  # 1 hour
                logger.info(f"Signal for {symbol} is too old: {signal_time}")
                continue
            
            current_price = df.loc[timestamp, "close"]
            
            # Process the signal
            trade = self.simulator.process_signal(signal, stock, current_price, timestamp)
            
            if trade:
                logger.info(f"Opened {trade.direction.value} trade for {symbol} at {trade.entry_price}")
        
        # Update positions for all stocks
        closed_trades = self.simulator.update_positions(stocks, current_time)
        
        if closed_trades:
            logger.info(f"Closed {len(closed_trades)} trades")
            
            # Print trade details
            for trade in closed_trades:
                logger.info(f"Closed {trade.direction.value} trade for {trade.symbol} at {trade.exit_price} ({trade.exit_reason})")
                logger.info(f"Profit/Loss: {trade.profit_loss:.2f} ({trade.profit_loss_percent:.2f}%)")
        
        # Print account summary
        account_summary = self.simulator.get_account_summary()
        logger.info(f"Account balance: {account_summary['current_balance']:.2f} {account_summary['currency']}")
        logger.info(f"Profit/Loss: {account_summary['profit_loss']:.2f} {account_summary['currency']} ({account_summary['profit_loss_percent']:.2f}%)")
        logger.info(f"Open positions: {account_summary['open_positions']}")
        logger.info(f"Total trades: {account_summary['total_trades']}")
