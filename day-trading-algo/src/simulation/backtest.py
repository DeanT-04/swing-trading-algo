#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtesting module for the day trading algorithm.

This module implements a backtesting system to evaluate the performance
of the day trading algorithm on historical data.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.models import Stock, TimeFrame, TradeDirection, Trade, Position
from src.data.provider import YahooFinanceProvider
from src.strategy.intraday_strategy import IntradayStrategy
from src.risk.position_sizing import calculate_position_size, calculate_risk_metrics

# Set up logging
logger = logging.getLogger(__name__)


def run_backtest(config: Dict, start_date: str, end_date: str, timeframe: TimeFrame) -> Dict:
    """
    Run a backtest of the day trading algorithm.
    
    Args:
        config: Configuration dictionary
        start_date: Start date for the backtest (YYYY-MM-DD)
        end_date: End date for the backtest (YYYY-MM-DD)
        timeframe: Timeframe to use for the backtest
        
    Returns:
        Dict: Backtest results
    """
    logger.info(f"Starting backtest from {start_date} to {end_date} with timeframe {timeframe.value}")
    
    # Initialize data provider
    data_provider = YahooFinanceProvider(config)
    
    # Initialize strategy
    strategy = IntradayStrategy(config)
    
    # Get symbols from config
    symbols = config.get("data", {}).get("symbols", [])
    if not symbols:
        logger.error("No symbols configured for trading")
        return {"success": False, "error": "No symbols configured"}
    
    # Limit to 10 symbols for testing
    symbols = symbols[:10]
    logger.info(f"Using {len(symbols)} symbols for backtest")
    
    # Fetch data for all symbols
    logger.info(f"Fetching data for {len(symbols)} symbols")
    stocks = data_provider.get_multiple_stocks_data(symbols, [timeframe], start_date, end_date)
    
    if not stocks:
        logger.error("No data fetched for any symbol")
        return {"success": False, "error": "No data fetched"}
    
    logger.info(f"Data fetched for {len(stocks)} symbols")
    
    # Initialize backtest state
    initial_balance = config.get("risk", {}).get("initial_balance", 50.0)
    balance = initial_balance
    positions = []
    trades = []
    daily_balances = []
    
    # Run backtest for each symbol
    for symbol, stock in stocks.items():
        logger.info(f"Analyzing {symbol}...")
        
        # Analyze the stock
        analysis = strategy.analyze(stock, timeframe)
        
        if not analysis:
            logger.warning(f"No analysis results for {symbol}")
            continue
        
        # Get signals
        signals = analysis.get("signals", [])
        
        if not signals:
            logger.info(f"No signals found for {symbol}")
            continue
        
        logger.info(f"Found {len(signals)} signals for {symbol}")
        
        # Process signals
        for signal in signals:
            # Skip if we don't have enough balance
            if balance < 10.0:  # Minimum balance to trade
                logger.warning(f"Insufficient balance ({balance}) to process signal")
                continue
            
            # Calculate position size
            position_size = calculate_position_size(
                account_balance=balance,
                risk_per_trade=config.get("risk", {}).get("max_risk_per_trade", 0.01),
                entry_price=signal["entry_price"],
                stop_loss=signal["stop_loss"],
                direction=signal["direction"]
            )
            
            # Skip if position size is too small
            if position_size < 0.01:
                logger.info(f"Position size too small ({position_size}), skipping signal")
                continue
            
            # Create position
            position = Position(
                symbol=symbol,
                direction=signal["direction"],
                entry_price=signal["entry_price"],
                stop_loss=signal["stop_loss"],
                take_profit=signal["take_profit"],
                size=position_size,
                entry_time=signal["timestamp"],
                status="open"
            )
            
            # Add position to list
            positions.append(position)
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                direction=signal["direction"],
                entry_price=signal["entry_price"],
                entry_time=signal["timestamp"],
                size=position_size,
                stop_loss=signal["stop_loss"],
                take_profit=signal["take_profit"],
                status="open",
                reason=signal.get("reason", "")
            )
            
            # Add trade to list
            trades.append(trade)
            
            logger.info(f"Opened {signal['direction'].name} position for {symbol} at {signal['entry_price']}")
            logger.info(f"Position size: {position_size}, Stop loss: {signal['stop_loss']}, Take profit: {signal['take_profit']}")
            
            # Simulate position until exit
            exit_price, exit_time, exit_reason = simulate_position_exit(
                position, stock, timeframe, signal["timestamp"]
            )
            
            if exit_price and exit_time:
                # Calculate profit/loss
                if position.direction == TradeDirection.LONG:
                    profit_loss = (exit_price - position.entry_price) * position.size
                else:  # SHORT
                    profit_loss = (position.entry_price - exit_price) * position.size
                
                # Update balance
                balance += profit_loss
                
                # Update position
                position.exit_price = exit_price
                position.exit_time = exit_time
                position.profit_loss = profit_loss
                position.status = "closed"
                
                # Update trade
                trade.exit_price = exit_price
                trade.exit_time = exit_time
                trade.profit_loss = profit_loss
                trade.status = "closed"
                trade.exit_reason = exit_reason
                
                logger.info(f"Closed {position.direction.name} position for {symbol} at {exit_price}")
                logger.info(f"Profit/Loss: {profit_loss:.2f} ({exit_reason})")
                logger.info(f"New balance: {balance:.2f}")
                
                # Record daily balance
                daily_balances.append({
                    "date": exit_time.strftime("%Y-%m-%d"),
                    "balance": balance
                })
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(trades, initial_balance, balance)
    
    # Generate equity curve
    equity_curve_path = generate_equity_curve(daily_balances, initial_balance)
    
    # Save results
    results = {
        "success": True,
        "initial_balance": initial_balance,
        "final_balance": balance,
        "total_return": balance - initial_balance,
        "total_return_percent": (balance - initial_balance) / initial_balance * 100,
        "trades_count": len(trades),
        "metrics": metrics,
        "equity_curve_path": equity_curve_path
    }
    
    # Save detailed results to file
    save_backtest_results(results, trades)
    
    logger.info(f"Backtest completed with final balance: {balance:.2f}")
    logger.info(f"Total return: {balance - initial_balance:.2f} ({(balance - initial_balance) / initial_balance * 100:.2f}%)")
    
    return results


def simulate_position_exit(position: Position, stock: Stock, timeframe: TimeFrame, 
                         entry_time: datetime) -> Tuple[Optional[float], Optional[datetime], str]:
    """
    Simulate the exit of a position based on stop loss, take profit, or end of data.
    
    Args:
        position: Position to simulate
        stock: Stock data
        timeframe: Timeframe to use
        entry_time: Entry time
        
    Returns:
        Tuple: (exit_price, exit_time, exit_reason)
    """
    # Get data after entry time
    data = stock.data.get(timeframe, [])
    
    if not data:
        logger.warning(f"No data available for {stock.symbol}")
        return None, None, "no_data"
    
    # Find entry index
    entry_index = -1
    for i, point in enumerate(data):
        if point.timestamp >= entry_time:
            entry_index = i
            break
    
    if entry_index == -1 or entry_index >= len(data) - 1:
        logger.warning(f"Entry time {entry_time} not found in data or at the end of data")
        return None, None, "entry_not_found"
    
    # Simulate position until exit
    for i in range(entry_index + 1, len(data)):
        point = data[i]
        
        # Check stop loss
        if position.direction == TradeDirection.LONG and point.low <= position.stop_loss:
            return position.stop_loss, point.timestamp, "stop_loss"
        
        if position.direction == TradeDirection.SHORT and point.high >= position.stop_loss:
            return position.stop_loss, point.timestamp, "stop_loss"
        
        # Check take profit
        if position.direction == TradeDirection.LONG and point.high >= position.take_profit:
            return position.take_profit, point.timestamp, "take_profit"
        
        if position.direction == TradeDirection.SHORT and point.low <= position.take_profit:
            return position.take_profit, point.timestamp, "take_profit"
    
    # If we reach the end of data, close at the last price
    last_point = data[-1]
    return last_point.close, last_point.timestamp, "end_of_data"


def calculate_performance_metrics(trades: List[Trade], initial_balance: float, 
                                final_balance: float) -> Dict:
    """
    Calculate performance metrics for the backtest.
    
    Args:
        trades: List of trades
        initial_balance: Initial account balance
        final_balance: Final account balance
        
    Returns:
        Dict: Performance metrics
    """
    # Filter closed trades
    closed_trades = [t for t in trades if t.status == "closed"]
    
    if not closed_trades:
        return {
            "win_rate": 0,
            "profit_factor": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "avg_trade": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }
    
    # Calculate metrics
    winning_trades = [t for t in closed_trades if t.profit_loss and t.profit_loss > 0]
    losing_trades = [t for t in closed_trades if t.profit_loss and t.profit_loss <= 0]
    
    win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
    
    total_profit = sum(t.profit_loss for t in winning_trades) if winning_trades else 0
    total_loss = sum(t.profit_loss for t in losing_trades) if losing_trades else 0
    
    profit_factor = abs(total_profit / total_loss) if total_loss else float('inf')
    
    avg_win = total_profit / len(winning_trades) if winning_trades else 0
    avg_loss = total_loss / len(losing_trades) if losing_trades else 0
    
    largest_win = max([t.profit_loss for t in winning_trades]) if winning_trades else 0
    largest_loss = min([t.profit_loss for t in losing_trades]) if losing_trades else 0
    
    avg_trade = (total_profit + total_loss) / len(closed_trades) if closed_trades else 0
    
    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "avg_trade": avg_trade,
        "total_trades": len(closed_trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades)
    }


def generate_equity_curve(daily_balances: List[Dict], initial_balance: float) -> str:
    """
    Generate an equity curve chart.
    
    Args:
        daily_balances: List of daily balance records
        initial_balance: Initial account balance
        
    Returns:
        str: Path to the generated chart
    """
    if not daily_balances:
        logger.warning("No daily balances to generate equity curve")
        return ""
    
    # Create directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(daily_balances)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # Add initial balance
    initial_row = pd.DataFrame([{
        "date": df["date"].min() - timedelta(days=1),
        "balance": initial_balance
    }])
    df = pd.concat([initial_row, df]).reset_index(drop=True)
    
    # Set up plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot equity curve
    plt.plot(df["date"], df["balance"], marker="o", linestyle="-", color="blue")
    
    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Account Balance")
    plt.title("Equity Curve")
    
    # Add grid
    plt.grid(True)
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    filename = f"reports/equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.close()
    
    logger.info(f"Equity curve saved to {filename}")
    
    return filename


def save_backtest_results(results: Dict, trades: List[Trade]) -> str:
    """
    Save backtest results to a file.
    
    Args:
        results: Backtest results
        trades: List of trades
        
    Returns:
        str: Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Create results dictionary
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "trades": [t.to_dict() for t in trades]
    }
    
    # Save to file
    filename = f"reports/backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=4)
    
    logger.info(f"Backtest results saved to {filename}")
    
    return filename
