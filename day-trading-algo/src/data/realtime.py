#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time data provider for the day trading algorithm.

This module handles fetching real-time market data for the day trading algorithm.
"""

import logging
import time
import threading
import queue
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

from src.data.models import Stock, TimeFrame, OHLCV

# Set up logging
logger = logging.getLogger(__name__)


class RealTimeDataProvider:
    """
    Real-time data provider for the day trading algorithm.
    
    This class handles fetching real-time market data from various sources.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the real-time data provider.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get("data", {})
        
        # Get symbols from config
        self.symbols = self.data_config.get("symbols", [])
        if not self.symbols:
            logger.error("No symbols configured for trading")
            raise ValueError("No symbols configured for trading")
        
        # Get timeframes from config
        timeframes_str = self.data_config.get("timeframes", ["5m"])
        self.timeframes = [TimeFrame.from_string(tf) for tf in timeframes_str]
        
        # Initialize data cache
        self.data_cache = {}
        for symbol in self.symbols:
            self.data_cache[symbol] = {}
            for timeframe in self.timeframes:
                self.data_cache[symbol][timeframe] = []
        
        # Initialize data queue
        self.data_queue = queue.Queue()
        
        # Initialize running flag
        self.running = False
        
        # Initialize last update time
        self.last_update = {}
        for symbol in self.symbols:
            self.last_update[symbol] = {}
            for timeframe in self.timeframes:
                self.last_update[symbol][timeframe] = datetime.now() - timedelta(days=1)
        
        # Initialize update interval
        self.update_interval = self.config.get("paper_trading", {}).get("data_streaming", {}).get("polling_interval", 10)
        
        logger.info(f"Initialized real-time data provider with {len(self.symbols)} symbols and {len(self.timeframes)} timeframes")
    
    def start(self):
        """Start the real-time data provider."""
        if self.running:
            logger.warning("Real-time data provider already running")
            return
        
        logger.info("Starting real-time data provider")
        
        # Set running flag
        self.running = True
        
        # Start data fetcher thread
        self.fetcher_thread = threading.Thread(
            target=self._data_fetcher,
            daemon=True
        )
        self.fetcher_thread.start()
    
    def stop(self):
        """Stop the real-time data provider."""
        if not self.running:
            logger.warning("Real-time data provider not running")
            return
        
        logger.info("Stopping real-time data provider")
        
        # Clear running flag
        self.running = False
        
        # Wait for thread to finish
        self.fetcher_thread.join(timeout=5)
        
        logger.info("Real-time data provider stopped")
    
    def get_latest_data(self, symbol: str, timeframe: TimeFrame) -> Optional[OHLCV]:
        """
        Get the latest data for a symbol and timeframe.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            
        Returns:
            Optional[OHLCV]: Latest data or None if no data available
        """
        if symbol not in self.data_cache:
            logger.warning(f"Symbol {symbol} not in data cache")
            return None
        
        if timeframe not in self.data_cache[symbol]:
            logger.warning(f"Timeframe {timeframe.value} not in data cache for {symbol}")
            return None
        
        if not self.data_cache[symbol][timeframe]:
            logger.warning(f"No data available for {symbol} {timeframe.value}")
            return None
        
        return self.data_cache[symbol][timeframe][-1]
    
    def get_stock(self, symbol: str) -> Optional[Stock]:
        """
        Get a Stock object for a symbol with all available timeframes.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Optional[Stock]: Stock object or None if no data available
        """
        if symbol not in self.data_cache:
            logger.warning(f"Symbol {symbol} not in data cache")
            return None
        
        stock = Stock(symbol=symbol)
        
        for timeframe in self.timeframes:
            if timeframe in self.data_cache[symbol] and self.data_cache[symbol][timeframe]:
                stock.add_data(timeframe, self.data_cache[symbol][timeframe])
        
        if not stock.data:
            logger.warning(f"No data available for {symbol}")
            return None
        
        return stock
    
    def get_data_queue(self) -> queue.Queue:
        """
        Get the data queue.
        
        Returns:
            queue.Queue: Data queue
        """
        return self.data_queue
    
    def _data_fetcher(self):
        """Fetch real-time data in a separate thread."""
        logger.info(f"Starting data fetcher thread for {len(self.symbols)} symbols")
        
        while self.running:
            try:
                # Fetch data for each symbol and timeframe
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        # Check if it's time to update
                        now = datetime.now()
                        last_update = self.last_update[symbol][timeframe]
                        
                        # Calculate update interval based on timeframe
                        interval = self._get_update_interval(timeframe)
                        
                        if now - last_update >= timedelta(seconds=interval):
                            # Fetch data
                            data = self._fetch_latest_data(symbol, timeframe)
                            
                            if data:
                                # Update data cache
                                self.data_cache[symbol][timeframe] = data
                                
                                # Update last update time
                                self.last_update[symbol][timeframe] = now
                                
                                # Put stock in queue
                                stock = Stock(symbol=symbol)
                                stock.add_data(timeframe, data)
                                self.data_queue.put(stock)
                                
                                logger.debug(f"Fetched {len(data)} data points for {symbol} {timeframe.value}")
                
                # Sleep for a while
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in data fetcher: {e}")
                time.sleep(self.update_interval)
    
    def _fetch_latest_data(self, symbol: str, timeframe: TimeFrame) -> List[OHLCV]:
        """
        Fetch the latest data for a symbol and timeframe.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            
        Returns:
            List[OHLCV]: List of OHLCV objects
        """
        try:
            # Convert timeframe to Yahoo Finance interval
            interval = self._convert_timeframe_to_interval(timeframe)
            
            # Calculate period based on timeframe
            period = self._get_period(timeframe)
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol} {timeframe.value}")
                return []
            
            # Convert to OHLCV objects
            data = []
            for timestamp, row in df.iterrows():
                data.append(OHLCV(
                    timestamp=timestamp.to_pydatetime(),
                    open=row["Open"],
                    high=row["High"],
                    low=row["Low"],
                    close=row["Close"],
                    volume=row["Volume"]
                ))
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe.value}: {e}")
            return []
    
    def _convert_timeframe_to_interval(self, timeframe: TimeFrame) -> str:
        """
        Convert TimeFrame enum to Yahoo Finance interval string.
        
        Args:
            timeframe: TimeFrame enum
            
        Returns:
            str: Yahoo Finance interval string
        """
        mapping = {
            TimeFrame.MINUTE_1: "1m",
            TimeFrame.MINUTE_5: "5m",
            TimeFrame.MINUTE_15: "15m",
            TimeFrame.HOUR_1: "1h",
            TimeFrame.DAY_1: "1d"
        }
        
        return mapping.get(timeframe, "1d")
    
    def _get_period(self, timeframe: TimeFrame) -> str:
        """
        Get the period string for Yahoo Finance based on timeframe.
        
        Args:
            timeframe: TimeFrame enum
            
        Returns:
            str: Yahoo Finance period string
        """
        mapping = {
            TimeFrame.MINUTE_1: "1d",
            TimeFrame.MINUTE_5: "5d",
            TimeFrame.MINUTE_15: "5d",
            TimeFrame.HOUR_1: "1mo",
            TimeFrame.DAY_1: "3mo"
        }
        
        return mapping.get(timeframe, "1d")
    
    def _get_update_interval(self, timeframe: TimeFrame) -> int:
        """
        Get the update interval in seconds based on timeframe.
        
        Args:
            timeframe: TimeFrame enum
            
        Returns:
            int: Update interval in seconds
        """
        mapping = {
            TimeFrame.MINUTE_1: 60,
            TimeFrame.MINUTE_5: 300,
            TimeFrame.MINUTE_15: 900,
            TimeFrame.HOUR_1: 3600,
            TimeFrame.DAY_1: 86400
        }
        
        return mapping.get(timeframe, 300)
