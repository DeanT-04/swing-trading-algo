#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data provider for retrieving market data.

This module provides functionality to retrieve historical and real-time
market data from various sources such as Alpha Vantage, Yahoo Finance, or CSV files.
"""

import logging
import os
import pandas as pd
import requests
import yfinance as yf
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.data.models import OHLCV, Stock, TimeFrame

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """
    Abstract base class for data providers.
    """

    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: TimeFrame, start_date: datetime, end_date: Optional[datetime] = None) -> Stock:
        """
        Retrieve historical data for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for the data
            start_date: Start date for the data
            end_date: End date for the data (optional, defaults to current date)

        Returns:
            Stock: Stock object with historical data
        """
        pass

    @abstractmethod
    def get_latest_data(self, symbol: str, timeframe: TimeFrame) -> Stock:
        """
        Retrieve the latest data for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for the data

        Returns:
            Stock: Stock object with the latest data
        """
        pass


class AlphaVantageProvider(DataProvider):
    """
    Data provider using Alpha Vantage API.
    """

    def __init__(self, api_key: str, cache_dir: Optional[str] = None):
        """
        Initialize the Alpha Vantage data provider.

        Args:
            api_key: Alpha Vantage API key
            cache_dir: Directory to cache data (optional)
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.cache_dir = None

        logger.info("Initialized Alpha Vantage data provider")

    def get_historical_data(self, symbol: str, timeframe: TimeFrame, start_date: datetime, end_date: Optional[datetime] = None) -> Stock:
        """
        Retrieve historical data from Alpha Vantage.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for the data
            start_date: Start date for the data
            end_date: End date for the data (optional, defaults to current date)

        Returns:
            Stock: Stock object with historical data
        """
        if end_date is None:
            end_date = datetime.now()

        # Check cache first
        stock = self._check_cache(symbol, timeframe, start_date, end_date)
        if stock:
            return stock

        # Map timeframe to Alpha Vantage function and interval
        function, interval = self._map_timeframe(timeframe)

        # Prepare request parameters
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full"
        }

        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = interval

        # Make the request
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Create stock object
            stock = Stock(symbol=symbol)

            # Parse the response
            time_series_key = self._get_time_series_key(function)
            if time_series_key in data:
                time_series = data[time_series_key]

                for date_str, values in time_series.items():
                    # Parse date string
                    if function == "TIME_SERIES_INTRADAY":
                        timestamp = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    else:
                        timestamp = datetime.strptime(date_str, "%Y-%m-%d")

                    # Check if the date is within the requested range
                    if start_date <= timestamp <= end_date:
                        # Create OHLCV object
                        ohlcv = OHLCV(
                            timestamp=timestamp,
                            open=float(values["1. open"]),
                            high=float(values["2. high"]),
                            low=float(values["3. low"]),
                            close=float(values["4. close"]),
                            volume=int(values["5. volume"])
                        )

                        # Add to stock data
                        stock.add_data_point(timeframe, ohlcv)

            # Cache the data
            self._cache_data(stock, timeframe)

            return stock

        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving data from Alpha Vantage: {e}")
            return Stock(symbol=symbol)

    def get_latest_data(self, symbol: str, timeframe: TimeFrame) -> Stock:
        """
        Retrieve the latest data from Alpha Vantage.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for the data

        Returns:
            Stock: Stock object with the latest data
        """
        # For simplicity, we'll just get the last 5 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        return self.get_historical_data(symbol, timeframe, start_date, end_date)

    def _map_timeframe(self, timeframe: TimeFrame) -> tuple:
        """
        Map timeframe to Alpha Vantage function and interval.

        Args:
            timeframe: Timeframe enum

        Returns:
            tuple: (function, interval)
        """
        if timeframe == TimeFrame.DAILY:
            return "TIME_SERIES_DAILY", None
        elif timeframe == TimeFrame.FOUR_HOUR:
            return "TIME_SERIES_INTRADAY", "60min"  # Alpha Vantage doesn't have 4h, use 1h and aggregate later
        elif timeframe == TimeFrame.ONE_HOUR:
            return "TIME_SERIES_INTRADAY", "60min"
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

    def _get_time_series_key(self, function: str) -> str:
        """
        Get the time series key for the Alpha Vantage response.

        Args:
            function: Alpha Vantage function

        Returns:
            str: Time series key
        """
        if function == "TIME_SERIES_DAILY":
            return "Time Series (Daily)"
        elif function == "TIME_SERIES_INTRADAY":
            return "Time Series (60min)"
        else:
            raise ValueError(f"Unsupported function: {function}")

    def _check_cache(self, symbol: str, timeframe: TimeFrame, start_date: datetime, end_date: datetime) -> Optional[Stock]:
        """
        Check if data is available in cache.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for the data
            start_date: Start date for the data
            end_date: End date for the data

        Returns:
            Optional[Stock]: Stock object from cache or None if not available
        """
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{symbol}_{timeframe.value}.csv"

        if not cache_file.exists():
            return None

        try:
            # Read cache file
            df = pd.read_csv(cache_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter by date range
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

            if df.empty:
                return None

            # Create stock object
            stock = Stock(symbol=symbol)

            # Add data points
            for _, row in df.iterrows():
                ohlcv = OHLCV(
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"]
                )
                stock.add_data_point(timeframe, ohlcv)

            return stock

        except Exception as e:
            logger.error(f"Error reading cache file: {e}")
            return None

    def _cache_data(self, stock: Stock, timeframe: TimeFrame) -> None:
        """
        Cache data to file.

        Args:
            stock: Stock object with data
            timeframe: Timeframe for the data
        """
        if not self.cache_dir or timeframe not in stock.data:
            return

        try:
            # Convert to DataFrame
            df = stock.get_dataframe(timeframe)

            if df.empty:
                return

            # Reset index to make timestamp a column
            df = df.reset_index()

            # Save to cache file
            cache_file = self.cache_dir / f"{stock.symbol}_{timeframe.value}.csv"
            df.to_csv(cache_file, index=False)

            logger.info(f"Cached data for {stock.symbol} ({timeframe.value}) to {cache_file}")

        except Exception as e:
            logger.error(f"Error caching data: {e}")


class YahooFinanceProvider(DataProvider):
    """
    Data provider using Yahoo Finance API.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the Yahoo Finance data provider.

        Args:
            cache_dir: Directory to cache data (optional)
        """
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.cache_dir = None

        logger.info("Initialized Yahoo Finance data provider")

    def get_historical_data(self, symbol: str, timeframe: TimeFrame, start_date: datetime, end_date: Optional[datetime] = None) -> Stock:
        """
        Retrieve historical data from Yahoo Finance.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for the data
            start_date: Start date for the data
            end_date: End date for the data (optional, defaults to current date)

        Returns:
            Stock: Stock object with historical data
        """
        if end_date is None:
            end_date = datetime.now()

        # Check cache first
        stock = self._check_cache(symbol, timeframe, start_date, end_date)
        if stock:
            return stock

        # Map timeframe to yfinance interval
        interval = self._map_timeframe(timeframe)

        logger.info(f"Downloading {symbol} data from {start_date} to {end_date} with interval {interval}")

        try:
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return Stock(symbol=symbol)

            # Create stock object
            stock = Stock(symbol=symbol)

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
                    stock.add_data_point(timeframe, ohlcv)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error processing data point for {symbol}: {e}")
                    continue

            # Cache the data
            self._cache_data(stock, timeframe)

            return stock

        except Exception as e:
            logger.error(f"Error retrieving data from Yahoo Finance: {e}")
            return Stock(symbol=symbol)

    def get_latest_data(self, symbol: str, timeframe: TimeFrame) -> Stock:
        """
        Retrieve the latest data from Yahoo Finance.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for the data

        Returns:
            Stock: Stock object with the latest data
        """
        # For simplicity, we'll just get the last 5 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        return self.get_historical_data(symbol, timeframe, start_date, end_date)

    def _map_timeframe(self, timeframe: TimeFrame) -> str:
        """
        Map timeframe to yfinance interval.

        Args:
            timeframe: Timeframe enum

        Returns:
            str: yfinance interval string
        """
        if timeframe == TimeFrame.DAILY:
            return "1d"
        elif timeframe == TimeFrame.FOUR_HOUR:
            return "1h"  # yfinance doesn't have 4h, use 1h and aggregate later
        elif timeframe == TimeFrame.ONE_HOUR:
            return "1h"
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

    def _check_cache(self, symbol: str, timeframe: TimeFrame, start_date: datetime, end_date: datetime) -> Optional[Stock]:
        """
        Check if data is available in cache.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for the data
            start_date: Start date for the data
            end_date: End date for the data

        Returns:
            Optional[Stock]: Stock object from cache or None if not available
        """
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{symbol}_{timeframe.value}.csv"

        if not cache_file.exists():
            return None

        try:
            # Read cache file
            df = pd.read_csv(cache_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter by date range
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

            if df.empty:
                return None

            # Create stock object
            stock = Stock(symbol=symbol)

            # Add data points
            for _, row in df.iterrows():
                ohlcv = OHLCV(
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"]
                )
                stock.add_data_point(timeframe, ohlcv)

            return stock

        except Exception as e:
            logger.error(f"Error reading cache file: {e}")
            return None

    def _cache_data(self, stock: Stock, timeframe: TimeFrame) -> None:
        """
        Cache data to file.

        Args:
            stock: Stock object with data
            timeframe: Timeframe for the data
        """
        if not self.cache_dir or timeframe not in stock.data:
            return

        try:
            # Convert to DataFrame
            df = stock.get_dataframe(timeframe)

            if df.empty:
                return

            # Reset index to make timestamp a column
            df = df.reset_index()

            # Save to cache file
            cache_file = self.cache_dir / f"{stock.symbol}_{timeframe.value}.csv"
            df.to_csv(cache_file, index=False)

            logger.info(f"Cached data for {stock.symbol} ({timeframe.value}) to {cache_file}")

        except Exception as e:
            logger.error(f"Error caching data: {e}")


class CSVProvider(DataProvider):
    """
    Data provider using CSV files.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the CSV data provider.

        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        logger.info(f"Initialized CSV data provider with directory: {data_dir}")

    def get_historical_data(self, symbol: str, timeframe: TimeFrame, start_date: datetime, end_date: Optional[datetime] = None) -> Stock:
        """
        Retrieve historical data from CSV files.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for the data
            start_date: Start date for the data
            end_date: End date for the data (optional, defaults to current date)

        Returns:
            Stock: Stock object with historical data
        """
        if end_date is None:
            end_date = datetime.now()

        # Look for CSV file
        csv_file = self.data_dir / f"{symbol}_{timeframe.value}.csv"

        if not csv_file.exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return Stock(symbol=symbol)

        try:
            # Read CSV file
            df = pd.read_csv(csv_file)

            # Ensure timestamp column exists
            if "timestamp" not in df.columns:
                logger.error(f"CSV file does not have a timestamp column: {csv_file}")
                return Stock(symbol=symbol)

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter by date range
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

            # Create stock object
            stock = Stock(symbol=symbol)

            # Add data points
            for _, row in df.iterrows():
                ohlcv = OHLCV(
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"]
                )
                stock.add_data_point(timeframe, ohlcv)

            return stock

        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return Stock(symbol=symbol)

    def get_latest_data(self, symbol: str, timeframe: TimeFrame) -> Stock:
        """
        Retrieve the latest data from CSV files.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for the data

        Returns:
            Stock: Stock object with the latest data
        """
        # For CSV files, we'll just get all data and filter to the last few days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        return self.get_historical_data(symbol, timeframe, start_date, end_date)


def create_data_provider(config: Dict) -> DataProvider:
    """
    Create a data provider based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        DataProvider: Data provider instance
    """
    provider_type = config.get("provider", "alpha_vantage").lower()

    if provider_type == "alpha_vantage":
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("API key is required for Alpha Vantage provider")

        cache_dir = config.get("cache_dir")
        return AlphaVantageProvider(api_key=api_key, cache_dir=cache_dir)

    elif provider_type == "yahoo_finance":
        cache_dir = config.get("cache_dir")
        return YahooFinanceProvider(cache_dir=cache_dir)

    elif provider_type == "csv":
        data_dir = config.get("data_dir")
        if not data_dir:
            raise ValueError("Data directory is required for CSV provider")

        return CSVProvider(data_dir=data_dir)

    else:
        raise ValueError(f"Unsupported data provider: {provider_type}")
