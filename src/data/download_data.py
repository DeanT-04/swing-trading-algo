#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download historical stock data using yfinance.

This script downloads historical data for the specified symbols and
saves it to CSV files in the specified directory.
"""

import os
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml

from src.data.models import TimeFrame

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_stock_data(symbol, start_date, end_date, timeframe, output_dir):
    """
    Download historical stock data using yfinance.

    Args:
        symbol: Stock symbol
        start_date: Start date for data
        end_date: End date for data
        timeframe: Timeframe for data
        output_dir: Directory to save data

    Returns:
        pandas.DataFrame: Downloaded data
    """
    # Map TimeFrame to yfinance interval
    interval_map = {
        TimeFrame.DAILY: "1d",
        TimeFrame.FOUR_HOUR: "1h",  # yfinance doesn't have 4h, use 1h and aggregate later
        TimeFrame.ONE_HOUR: "1h"
    }

    if isinstance(timeframe, str):
        timeframe = TimeFrame.from_string(timeframe)

    interval = interval_map.get(timeframe, "1d")

    logger.info(f"Downloading {symbol} data from {start_date} to {end_date} with interval {interval}")

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
            logger.warning(f"No data found for {symbol}")
            return None

        # Aggregate to 4h if needed
        if timeframe == TimeFrame.FOUR_HOUR and interval == "1h":
            logger.info(f"Aggregating 1h data to 4h for {symbol}")
            data = data.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save to CSV
        output_file = os.path.join(output_dir, f"{symbol}_{timeframe.value}.csv")

        # Reset index to make timestamp a column
        data = data.reset_index()

        # Rename columns to match our OHLCV model
        data.columns = [str(c).lower() for c in data.columns]
        if 'date' in data.columns:
            data = data.rename(columns={'date': 'timestamp'})
        elif 'datetime' in data.columns:
            data = data.rename(columns={'datetime': 'timestamp'})

        data = data.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })

        # Save to CSV
        data.to_csv(output_file, index=False)
        logger.info(f"Saved {len(data)} rows to {output_file}")

        return data

    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return None


def load_config(config_path="config/config.yaml"):
    """Load configuration from config file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None


def main():
    """Main function to download stock data."""
    parser = argparse.ArgumentParser(description="Download historical stock data")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output-dir", default="data/csv",
                        help="Directory to save data")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if not config:
        return

    # Get symbols and timeframes from config
    symbols = config["data"]["symbols"]
    timeframes = [TimeFrame.from_string(tf) for tf in config["data"]["timeframes"]]
    history_days = config["data"]["history_days"]

    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=history_days)

    # Download data for each symbol and timeframe
    for symbol in symbols:
        for timeframe in timeframes:
            download_stock_data(symbol, start_date, end_date, timeframe, output_dir)


if __name__ == "__main__":
    main()
