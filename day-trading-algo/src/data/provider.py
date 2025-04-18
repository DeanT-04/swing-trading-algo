#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data provider for the day trading algorithm.

This module handles fetching and caching stock data from various sources.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import yfinance as yf
import time
import json

from src.data.models import Stock, TimeFrame, OHLCV

# Set up logging
logger = logging.getLogger(__name__)


class DataProvider:
    """
    Base class for data providers.
    
    This class defines the interface for data providers and implements
    common functionality like caching.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data provider.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config.get("data", {})
        self.cache_dir = self.data_config.get("cache_dir", "data/cache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized data provider with cache directory: {self.cache_dir}")
    
    def get_stock_data(self, symbol: str, timeframes: List[TimeFrame], 
                      start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[Stock]:
        """
        Get stock data for a symbol and timeframes.
        
        Args:
            symbol: Stock symbol
            timeframes: List of timeframes to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Stock object with data or None if data couldn't be fetched
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_stock_data")
    
    def get_multiple_stocks_data(self, symbols: List[str], timeframes: List[TimeFrame],
                               start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Stock]:
        """
        Get data for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            timeframes: List of timeframes to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping symbols to Stock objects
        """
        result = {}
        
        for symbol in symbols:
            try:
                stock = self.get_stock_data(symbol, timeframes, start_date, end_date)
                if stock:
                    result[symbol] = stock
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return result
    
    def _get_cache_path(self, symbol: str, timeframe: TimeFrame, start_date: str, end_date: str) -> str:
        """
        Get the path to the cache file for a symbol and timeframe.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Path to cache file
        """
        return os.path.join(self.cache_dir, f"{symbol}_{timeframe.value}_{start_date}_{end_date}.csv")
    
    def _load_from_cache(self, symbol: str, timeframe: TimeFrame, 
                        start_date: str, end_date: str) -> Optional[List[OHLCV]]:
        """
        Load data from cache.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of OHLCV objects or None if cache doesn't exist
        """
        cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            df = pd.read_csv(cache_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            data = []
            for _, row in df.iterrows():
                data.append(OHLCV(
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"]
                ))
            
            logger.info(f"Loaded {len(data)} data points for {symbol} {timeframe.value} from cache")
            return data
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None
    
    def _save_to_cache(self, symbol: str, timeframe: TimeFrame, 
                      start_date: str, end_date: str, data: List[OHLCV]):
        """
        Save data to cache.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data: List of OHLCV objects
        """
        cache_path = self._get_cache_path(symbol, timeframe, start_date, end_date)
        
        try:
            df = pd.DataFrame([{
                "timestamp": point.timestamp,
                "open": point.open,
                "high": point.high,
                "low": point.low,
                "close": point.close,
                "volume": point.volume
            } for point in data])
            
            df.to_csv(cache_path, index=False)
            logger.info(f"Saved {len(data)} data points for {symbol} {timeframe.value} to cache")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")


class YahooFinanceProvider(DataProvider):
    """
    Data provider that fetches data from Yahoo Finance.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Yahoo Finance data provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.rate_limit_delay = 0.5  # Delay between API calls to avoid rate limiting
        logger.info("Initialized Yahoo Finance data provider")
    
    def get_stock_data(self, symbol: str, timeframes: List[TimeFrame], 
                      start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[Stock]:
        """
        Get stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            timeframes: List of timeframes to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Stock object with data or None if data couldn't be fetched
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            # Default to 60 days of history
            history_days = self.data_config.get("history_days", 60)
            start_date = (datetime.now() - timedelta(days=history_days)).strftime("%Y-%m-%d")
        
        # Create stock object
        stock = Stock(symbol=symbol)
        
        # Fetch data for each timeframe
        for timeframe in timeframes:
            try:
                # Try to load from cache first
                data = self._load_from_cache(symbol, timeframe, start_date, end_date)
                
                if data is None:
                    # If not in cache, fetch from Yahoo Finance
                    data = self._fetch_from_yahoo(symbol, timeframe, start_date, end_date)
                    
                    if data:
                        # Save to cache
                        self._save_to_cache(symbol, timeframe, start_date, end_date, data)
                
                if data:
                    stock.add_data(timeframe, data)
            except Exception as e:
                logger.error(f"Error fetching {timeframe.value} data for {symbol}: {e}")
        
        # Return None if no data was fetched
        if not stock.data:
            logger.warning(f"No data fetched for {symbol}")
            return None
        
        return stock
    
    def _fetch_from_yahoo(self, symbol: str, timeframe: TimeFrame, 
                         start_date: str, end_date: str) -> List[OHLCV]:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of OHLCV objects
        """
        logger.info(f"Fetching {timeframe.value} data for {symbol} from {start_date} to {end_date}")
        
        # Convert timeframe to Yahoo Finance interval
        interval = self._convert_timeframe_to_interval(timeframe)
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        # Add a delay to avoid rate limiting
        time.sleep(self.rate_limit_delay)
        
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
        
        logger.info(f"Fetched {len(data)} data points for {symbol} {timeframe.value}")
        return data
    
    def _convert_timeframe_to_interval(self, timeframe: TimeFrame) -> str:
        """
        Convert TimeFrame enum to Yahoo Finance interval string.
        
        Args:
            timeframe: TimeFrame enum
            
        Returns:
            Yahoo Finance interval string
        """
        mapping = {
            TimeFrame.MINUTE_1: "1m",
            TimeFrame.MINUTE_5: "5m",
            TimeFrame.MINUTE_15: "15m",
            TimeFrame.HOUR_1: "1h",
            TimeFrame.DAY_1: "1d"
        }
        
        return mapping.get(timeframe, "1d")
