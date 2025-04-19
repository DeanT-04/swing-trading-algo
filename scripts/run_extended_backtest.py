#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run extended backtests on historical data.

This script runs backtests on historical data over a long period (e.g., 10 years)
to evaluate the performance of trading strategies.
"""

import os
import sys
import yaml
import logging
import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('.')

from src.data.models import TimeFrame
from src.data.provider import create_data_provider
from src.strategy.basic_swing import BasicSwingStrategy
from src.simulation.backtester import Backtester
from src.utils.logger import setup_logging


def load_config(config_path="config/config.yaml"):
    """Load configuration from config file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def create_output_dirs(config):
    """Create output directories for logs, data, and reports."""
    # Create logs directory
    logs_dir = Path("logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create data cache directory
    if "cache_dir" in config["data"]:
        cache_dir = Path(config["data"]["cache_dir"])
        os.makedirs(cache_dir, exist_ok=True)

    # Create reports directory
    reports_dir = Path("reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Create backtest directories
    backtest_dir = Path("backtest")
    backtest_extended_dir = Path("backtest/extended")
    os.makedirs(backtest_dir, exist_ok=True)
    os.makedirs(backtest_extended_dir, exist_ok=True)

    return {
        "logs_dir": logs_dir,
        "reports_dir": reports_dir,
        "backtest_dir": backtest_dir,
        "backtest_extended_dir": backtest_extended_dir
    }


def plot_equity_curve(equity_curve, initial_balance, output_path):
    """Plot the equity curve and save to file."""
    plt.figure(figsize=(12, 6))

    # Convert to DataFrame if it's a list
    if isinstance(equity_curve, list):
        equity_curve = pd.DataFrame(equity_curve)

    # Plot equity curve
    plt.plot(equity_curve["date"], equity_curve["equity"], label="Equity")

    # Plot initial balance
    plt.axhline(y=initial_balance, color='r', linestyle='--', label="Initial Balance")

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Extended Backtest Equity Curve")
    plt.legend()
    plt.grid(True)

    # Format x-axis dates
    plt.gcf().autofmt_xdate()

    # Save to file
    plt.savefig(output_path)
    plt.close()


def plot_annual_returns(annual_returns, output_path):
    """Plot annual returns and save to file."""
    plt.figure(figsize=(12, 6))

    # Plot annual returns
    years = list(annual_returns.keys())
    returns = list(annual_returns.values())

    plt.bar(years, returns)

    # Add labels and title
    plt.xlabel("Year")
    plt.ylabel("Return (%)")
    plt.title("Annual Returns")
    plt.grid(True, axis='y')

    # Add return values on top of bars
    for i, v in enumerate(returns):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')

    # Save to file
    plt.savefig(output_path)
    plt.close()


def plot_drawdowns(drawdowns, output_path):
    """Plot drawdowns and save to file."""
    plt.figure(figsize=(12, 6))

    # Plot drawdowns
    plt.plot(drawdowns.index, drawdowns.values)

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.title("Drawdowns")
    plt.grid(True)

    # Format x-axis dates
    plt.gcf().autofmt_xdate()

    # Save to file
    plt.savefig(output_path)
    plt.close()


def plot_monthly_returns_heatmap(monthly_returns, output_path):
    """Plot monthly returns heatmap and save to file."""
    plt.figure(figsize=(12, 8))

    # Create a pivot table of monthly returns
    pivot_table = pd.pivot_table(
        monthly_returns,
        values='return',
        index='year',
        columns='month',
        aggfunc='sum'
    )

    # Create heatmap
    plt.imshow(pivot_table, cmap='RdYlGn', aspect='auto')

    # Add colorbar
    plt.colorbar(label='Return (%)')

    # Add labels and title
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.title("Monthly Returns Heatmap")

    # Set x-axis labels (months)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(12), month_labels)

    # Set y-axis labels (years)
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)

    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            value = pivot_table.iloc[i, j]
            if not np.isnan(value):
                text_color = 'black' if abs(value) < 10 else 'white'
                plt.text(j, i, f"{value:.2f}%", ha='center', va='center', color=text_color)

    # Save to file
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def calculate_extended_metrics(results, initial_balance):
    """
    Calculate extended performance metrics from backtest results.

    Args:
        results: List of backtest result dictionaries
        initial_balance: Initial account balance

    Returns:
        Dict: Dictionary of extended performance metrics
    """
    # Combine equity curves
    equity_curves = []
    for result in results:
        equity_curve = result["equity_curve"]
        if isinstance(equity_curve, list):
            equity_curve = pd.DataFrame(equity_curve)
        equity_curves.append(equity_curve)

    combined_equity_curve = pd.concat(equity_curves).sort_index()

    # Calculate drawdowns
    peak = combined_equity_curve['equity'].cummax()
    drawdown = (combined_equity_curve['equity'] - peak) / peak * 100

    # Calculate annual returns
    annual_returns = {}
    # Check if index is datetime
    if pd.api.types.is_datetime64_any_dtype(combined_equity_curve.index):
        for year in range(combined_equity_curve.index[0].year, combined_equity_curve.index[-1].year + 1):
            year_data = combined_equity_curve[combined_equity_curve.index.year == year]
            if not year_data.empty:
                start_equity = year_data.iloc[0]['equity']
                end_equity = year_data.iloc[-1]['equity']
                annual_return = (end_equity / start_equity - 1) * 100
                annual_returns[year] = annual_return
    else:
        # If index is not datetime, use a simpler approach
        # Group by year from the 'date' column
        # Handle timezone-aware datetimes
        combined_equity_curve['year'] = pd.to_datetime(combined_equity_curve['date'], utc=True).dt.year
        years = combined_equity_curve['year'].unique()

        for year in years:
            year_data = combined_equity_curve[combined_equity_curve['year'] == year]
            if not year_data.empty:
                start_equity = year_data.iloc[0]['equity']
                end_equity = year_data.iloc[-1]['equity']
                annual_return = (end_equity / start_equity - 1) * 100
                annual_returns[year] = annual_return

    # Calculate monthly returns
    monthly_returns = []

    # Check if index is datetime
    if pd.api.types.is_datetime64_any_dtype(combined_equity_curve.index):
        for year in range(combined_equity_curve.index[0].year, combined_equity_curve.index[-1].year + 1):
            for month in range(1, 13):
                month_data = combined_equity_curve[
                    (combined_equity_curve.index.year == year) &
                    (combined_equity_curve.index.month == month)
                ]
                if not month_data.empty:
                    start_equity = month_data.iloc[0]['equity']
                    end_equity = month_data.iloc[-1]['equity']
                    monthly_return = (end_equity / start_equity - 1) * 100
                    monthly_returns.append({
                        'year': year,
                        'month': month,
                        'return': monthly_return
                    })
    else:
        # If index is not datetime, use a simpler approach
        # Convert date column to datetime if it's not already
        # Handle timezone-aware datetimes
        combined_equity_curve['date_dt'] = pd.to_datetime(combined_equity_curve['date'], utc=True)
        combined_equity_curve['year'] = combined_equity_curve['date_dt'].dt.year
        combined_equity_curve['month'] = combined_equity_curve['date_dt'].dt.month

        # Get unique year-month combinations
        year_month_combos = combined_equity_curve[['year', 'month']].drop_duplicates().values

        for year, month in year_month_combos:
            month_data = combined_equity_curve[
                (combined_equity_curve['year'] == year) &
                (combined_equity_curve['month'] == month)
            ]
            if not month_data.empty:
                start_equity = month_data.iloc[0]['equity']
                end_equity = month_data.iloc[-1]['equity']
                monthly_return = (end_equity / start_equity - 1) * 100
                monthly_returns.append({
                    'year': year,
                    'month': month,
                    'return': monthly_return
                })

    monthly_returns_df = pd.DataFrame(monthly_returns)

    # Calculate total trades
    total_trades = sum(result["total_trades"] for result in results)
    winning_trades = sum(result["winning_trades"] for result in results)
    losing_trades = sum(result["losing_trades"] for result in results)

    # Calculate win rate
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    # Calculate final balance
    final_balance = combined_equity_curve.iloc[-1]['equity']

    # Calculate total return
    total_return = final_balance - initial_balance
    total_return_percent = (final_balance / initial_balance - 1) * 100

    # Calculate CAGR (Compound Annual Growth Rate)
    # Check if index is datetime
    if pd.api.types.is_datetime64_any_dtype(combined_equity_curve.index):
        years = (combined_equity_curve.index[-1] - combined_equity_curve.index[0]).days / 365.25
    else:
        # If index is not datetime, use the date column
        start_date = pd.to_datetime(combined_equity_curve['date'].iloc[0], utc=True)
        end_date = pd.to_datetime(combined_equity_curve['date'].iloc[-1], utc=True)
        years = (end_date - start_date).days / 365.25

    # Ensure years is at least 0.1 to avoid division by zero or very small numbers
    years = max(years, 0.1)
    cagr = (final_balance / initial_balance) ** (1 / years) - 1

    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    daily_returns = combined_equity_curve['equity'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0

    # Calculate Sortino ratio (assuming risk-free rate of 0)
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() != 0 else 0

    # Calculate maximum drawdown
    max_drawdown = drawdown.min()

    # Calculate average trade metrics
    avg_profit = sum(result.get("avg_profit", 0) * result["winning_trades"] for result in results) / winning_trades if winning_trades > 0 else 0
    avg_loss = sum(result.get("avg_loss", 0) * result["losing_trades"] for result in results) / losing_trades if losing_trades > 0 else 0

    return {
        "combined_equity_curve": combined_equity_curve,
        "drawdowns": drawdown,
        "annual_returns": annual_returns,
        "monthly_returns": monthly_returns_df,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "final_balance": final_balance,
        "total_return": total_return,
        "total_return_percent": total_return_percent,
        "cagr": cagr * 100,  # Convert to percentage
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss
    }


def main():
    """Main function to run the extended backtest."""
    parser = argparse.ArgumentParser(description="Run extended backtests on historical data")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--start-year", type=int, default=2013,
                        help="Start year for the backtest")
    parser.add_argument("--end-year", type=int, default=2023,
                        help="End year for the backtest")
    parser.add_argument("--symbols", help="Comma-separated list of symbols to backtest")
    parser.add_argument("--timeframe", choices=["daily", "4h", "1h"],
                        help="Timeframe for the backtest")
    parser.add_argument("--period", type=int, default=1,
                        help="Period length in years for each backtest")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create output directories
    output_dirs = create_output_dirs(config)

    # Set up logging
    logger = setup_logging(config["general"], output_dirs["logs_dir"])
    logger.info("Starting Extended Backtest")

    # Initialize backtest configuration
    if "extended_backtest" not in config:
        config["extended_backtest"] = {
            "start_year": args.start_year,
            "end_year": args.end_year,
            "period": args.period,
            "timeframe": config["data"]["timeframes"][0]
        }

    # Override configuration with command-line arguments
    if args.start_year:
        config["extended_backtest"]["start_year"] = args.start_year

    if args.end_year:
        config["extended_backtest"]["end_year"] = args.end_year

    if args.period:
        config["extended_backtest"]["period"] = args.period

    if args.timeframe:
        config["extended_backtest"]["timeframe"] = args.timeframe

    if args.symbols:
        config["data"]["symbols"] = args.symbols.split(",")

    # Initialize data provider
    try:
        data_provider = create_data_provider(config["data"])
        logger.info(f"Data provider initialized: {data_provider.__class__.__name__}")
    except Exception as e:
        logger.error(f"Error initializing data provider: {e}")
        return

    # Initialize timeframe
    timeframe = TimeFrame.from_string(config["extended_backtest"]["timeframe"])

    # Initialize strategy
    strategy = BasicSwingStrategy(config["strategy"])
    logger.info(f"Strategy initialized: {strategy.name}")

    # Initialize backtester
    backtester = Backtester(config["risk"])

    # Generate date ranges for backtests
    start_year = config["extended_backtest"]["start_year"]
    end_year = config["extended_backtest"]["end_year"]
    period = config["extended_backtest"]["period"]

    date_ranges = []
    for year in range(start_year, end_year, period):
        start_date = datetime(year, 1, 1)
        end_date = datetime(min(year + period, end_year), 1, 1)
        date_ranges.append((start_date, end_date))

    # Run backtests for each date range
    results = []

    for start_date, end_date in tqdm(date_ranges, desc="Running backtests"):
        logger.info(f"Running backtest from {start_date} to {end_date}")

        # Retrieve historical data for symbols
        symbols = config["data"]["symbols"]

        stocks = {}
        for symbol in symbols:
            try:
                logger.info(f"Retrieving data for {symbol} from {start_date} to {end_date}")
                stock = data_provider.get_historical_data(symbol, timeframe, start_date, end_date)
                stocks[symbol] = stock
                logger.info(f"Retrieved {len(stock.data.get(timeframe, []))} data points for {symbol}")
            except Exception as e:
                logger.error(f"Error retrieving data for {symbol}: {e}")

        if not stocks:
            logger.error(f"No data retrieved for any symbols for period {start_date} to {end_date}. Skipping.")
            continue

        # Run backtest
        result = backtester.run_backtest(stocks, strategy, timeframe, start_date, end_date)

        if not result["success"]:
            logger.error(f"Backtest failed for period {start_date} to {end_date}: {result.get('error', 'Unknown error')}")
            continue

        # Print backtest results
        logger.info(f"Backtest completed for period {start_date} to {end_date}")
        logger.info(f"Total trades: {result['total_trades']}")
        logger.info(f"Total return: {result['total_return_percent']:.2f}%")

        # Add to results
        results.append(result)

    if not results:
        logger.error("No successful backtests. Exiting.")
        return

    # Calculate extended metrics
    logger.info("Calculating extended metrics")
    extended_metrics = calculate_extended_metrics(results, config["general"]["initial_capital"])

    # Print extended metrics
    logger.info("Extended backtest completed")
    logger.info(f"Total trades: {extended_metrics['total_trades']}")
    logger.info(f"Win rate: {extended_metrics['win_rate']:.2f}%")
    logger.info(f"Total return: {extended_metrics['total_return_percent']:.2f}%")
    logger.info(f"CAGR: {extended_metrics['cagr']:.2f}%")
    logger.info(f"Sharpe ratio: {extended_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Sortino ratio: {extended_metrics['sortino_ratio']:.2f}")
    logger.info(f"Max drawdown: {extended_metrics['max_drawdown']:.2f}%")

    # Save extended metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert DataFrames to lists for JSON serialization
    extended_metrics_json = extended_metrics.copy()
    extended_metrics_json["combined_equity_curve"] = extended_metrics["combined_equity_curve"].to_dict(orient="records")

    # Convert drawdowns to a regular Python dict with string keys
    drawdowns_dict = {}
    for k, v in extended_metrics["drawdowns"].to_dict().items():
        drawdowns_dict[str(k)] = v
    extended_metrics_json["drawdowns"] = drawdowns_dict

    extended_metrics_json["monthly_returns"] = extended_metrics["monthly_returns"].to_dict(orient="records")

    # Convert annual_returns to a regular Python dict with string keys
    annual_returns_dict = {}
    for k, v in extended_metrics["annual_returns"].items():
        annual_returns_dict[str(k)] = v
    extended_metrics_json["annual_returns"] = annual_returns_dict

    metrics_file = output_dirs["backtest_extended_dir"] / f"extended_metrics_{timestamp}.json"

    with open(metrics_file, 'w') as file:
        json.dump(extended_metrics_json, file, indent=4, default=str)

    logger.info(f"Extended metrics saved to {metrics_file}")

    # Plot equity curve
    equity_curve_file = output_dirs["backtest_extended_dir"] / f"equity_curve_{timestamp}.png"
    plot_equity_curve(extended_metrics["combined_equity_curve"], config["general"]["initial_capital"], equity_curve_file)
    logger.info(f"Equity curve saved to {equity_curve_file}")

    # Plot annual returns
    annual_returns_file = output_dirs["backtest_extended_dir"] / f"annual_returns_{timestamp}.png"
    plot_annual_returns(extended_metrics["annual_returns"], annual_returns_file)
    logger.info(f"Annual returns plot saved to {annual_returns_file}")

    # Plot drawdowns
    drawdowns_file = output_dirs["backtest_extended_dir"] / f"drawdowns_{timestamp}.png"
    plot_drawdowns(extended_metrics["drawdowns"], drawdowns_file)
    logger.info(f"Drawdowns plot saved to {drawdowns_file}")

    # Plot monthly returns heatmap
    if not extended_metrics["monthly_returns"].empty:
        monthly_returns_file = output_dirs["backtest_extended_dir"] / f"monthly_returns_{timestamp}.png"
        plot_monthly_returns_heatmap(extended_metrics["monthly_returns"], monthly_returns_file)
        logger.info(f"Monthly returns heatmap saved to {monthly_returns_file}")


if __name__ == "__main__":
    main()
