#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trade Data Generator

This module generates simulated trade data with a specified win rate and
saves it to CSV files with detailed information.
"""

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

# Create directories
os.makedirs("reports", exist_ok=True)


def generate_trades(num_trades=200, win_rate=0.95):
    """
    Generate trades with a specified win rate.
    
    Args:
        num_trades: Number of trades to generate
        win_rate: Target win rate
        
    Returns:
        list: List of trade dictionaries
    """
    trades = []
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "NFLX"]
    
    # Initial balance
    balance = 50.0
    
    # Generate trades
    for i in range(num_trades):
        # Select random symbol
        symbol = random.choice(symbols)
        
        # Determine if trade is a winner
        is_winner = random.random() < win_rate
        
        # Determine trade direction
        direction = random.choice(["LONG", "SHORT"])
        
        # Generate entry price
        entry_price = random.uniform(50.0, 200.0)
        
        # Generate entry time (within market hours)
        entry_date = datetime.now() - timedelta(days=random.randint(0, 10))
        entry_hour = random.randint(9, 15)
        entry_minute = random.randint(0, 59)
        if entry_hour == 9 and entry_minute < 30:
            entry_minute += 30
        entry_time = entry_date.replace(hour=entry_hour, minute=entry_minute)
        
        # Generate exit time (after entry time)
        exit_time = entry_time + timedelta(minutes=random.randint(5, 60))
        
        # Generate profit/loss
        if is_winner:
            profit_loss = random.uniform(0.5, 5.0)
            exit_reason = "take_profit"
        else:
            profit_loss = -random.uniform(0.2, 2.0)
            exit_reason = "stop_loss"
        
        # Calculate exit price
        if direction == "LONG":
            exit_price = entry_price + profit_loss / 1.0  # Assuming 1 share
        else:
            exit_price = entry_price - profit_loss / 1.0  # Assuming 1 share
        
        # Update balance
        balance += profit_loss
        
        # Create trade dictionary
        trade = {
            "symbol": symbol,
            "direction": direction,
            "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": 1.0,
            "profit_loss": profit_loss,
            "exit_reason": exit_reason,
            "account_balance": balance
        }
        
        trades.append(trade)
    
    return trades


def save_trades_to_csv(trades, filename="reports/trade_log.csv"):
    """
    Save trades to a CSV file.
    
    Args:
        trades: List of trade dictionaries
        filename: Path to the CSV file
    """
    # Create DataFrame
    df = pd.DataFrame(trades)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f"Trades saved to {filename}")


def generate_detailed_trade_report(trades, output_file="reports/detailed_trade_report.csv"):
    """
    Generate a detailed trade report with specific information.
    
    Args:
        trades: List of trade dictionaries
        output_file: Path to the output CSV file
    """
    # Create DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Convert times to datetime
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
    
    # Calculate trade duration in minutes
    trades_df["duration_minutes"] = (trades_df["exit_time"] - trades_df["entry_time"]).dt.total_seconds() / 60
    
    # Create new columns
    trades_df["trade_date"] = trades_df["entry_time"].dt.strftime("%Y-%m-%d")
    trades_df["trade_day"] = trades_df["entry_time"].dt.day
    trades_df["trade_month"] = trades_df["entry_time"].dt.month
    trades_df["trade_year"] = trades_df["entry_time"].dt.year
    trades_df["time_zone"] = "GMT"  # Assuming GMT time zone
    trades_df["result"] = trades_df["profit_loss"].apply(lambda x: "WIN" if x > 0 else "LOSE")
    
    # Select and reorder columns
    detailed_df = trades_df[["symbol", "direction", "entry_time", "exit_time", "duration_minutes", 
                             "trade_date", "trade_day", "trade_month", "trade_year", "time_zone", 
                             "result", "profit_loss", "entry_price", "exit_price", "exit_reason"]]
    
    # Rename columns
    detailed_df = detailed_df.rename(columns={
        "symbol": "Stock Name",
        "direction": "Direction",
        "entry_time": "Entry Time",
        "exit_time": "Exit Time",
        "duration_minutes": "Duration (minutes)",
        "trade_date": "Trade Date",
        "trade_day": "Day",
        "trade_month": "Month",
        "trade_year": "Year",
        "time_zone": "Time Zone",
        "result": "Result",
        "profit_loss": "Profit/Loss",
        "entry_price": "Entry Price",
        "exit_price": "Exit Price",
        "exit_reason": "Exit Reason"
    })
    
    # Save to CSV
    detailed_df.to_csv(output_file, index=False)
    
    print(f"Detailed trade report saved to {output_file}")
    
    # Print summary
    win_count = (detailed_df["Result"] == "WIN").sum()
    lose_count = (detailed_df["Result"] == "LOSE").sum()
    win_rate = win_count / len(detailed_df) * 100 if len(detailed_df) > 0 else 0
    
    print("\nTrade Summary:")
    print(f"Total Trades: {len(detailed_df)}")
    print(f"Winning Trades: {win_count}")
    print(f"Losing Trades: {lose_count}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit/Loss: {detailed_df['Profit/Loss'].sum():.2f}")


def generate_trade_heatmap(trades, filename="reports/trade_heatmap.png"):
    """
    Generate a heatmap of trades showing profit/loss by symbol and hour.
    
    Args:
        trades: List of trade dictionaries
        filename: Path to the heatmap image
    """
    # Create DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Convert times to datetime
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
    
    # Add hour column
    trades_df["hour"] = trades_df["entry_time"].dt.hour
    
    # Group by symbol and hour, and calculate sum of profit/loss
    heatmap_data = trades_df.groupby(["symbol", "hour"])["profit_loss"].sum().reset_index()
    
    # Pivot the data for the heatmap
    pivot_data = heatmap_data.pivot(index="symbol", columns="hour", values="profit_loss")
    
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, cmap="RdYlGn", center=0, annot=True, fmt=".2f")
    plt.title("Trade Profit/Loss Heatmap by Symbol and Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Symbol")
    
    # Save the heatmap
    plt.savefig(filename)
    plt.close()
    
    print(f"Heatmap saved to {filename}")


def generate_performance_report(trades, filename="reports/performance_report.json"):
    """
    Generate a performance report.
    
    Args:
        trades: List of trade dictionaries
        filename: Path to the report file
    """
    # Calculate performance metrics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t["profit_loss"] > 0]
    losing_trades = [t for t in trades if t["profit_loss"] <= 0]
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades else 0
    
    total_profit = sum(t["profit_loss"] for t in winning_trades) if winning_trades else 0
    total_loss = sum(t["profit_loss"] for t in losing_trades) if losing_trades else 0
    
    profit_factor = abs(total_profit / total_loss) if total_loss else float('inf')
    
    avg_win = total_profit / len(winning_trades) if winning_trades else 0
    avg_loss = total_loss / len(losing_trades) if losing_trades else 0
    
    initial_balance = 50.0
    final_balance = trades[-1]["account_balance"] if trades else initial_balance
    
    total_return = final_balance - initial_balance
    total_return_percent = (total_return / initial_balance) * 100
    
    # Create report dictionary
    report = {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return": total_return,
        "total_return_percent": total_return_percent,
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save report to file
    with open(filename, 'w') as file:
        json.dump(report, file, indent=4)
    
    print(f"Performance report saved to {filename}")
    
    # Print report
    print("\n=== Performance Report ===")
    print(f"Initial Balance: {initial_balance:.2f}")
    print(f"Final Balance: {final_balance:.2f}")
    print(f"Total Return: {total_return:.2f} ({total_return_percent:.2f}%)")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Win: {avg_win:.2f}")
    print(f"Average Loss: {avg_loss:.2f}")
    print("==========================")


if __name__ == "__main__":
    # Generate trades with 95%+ win rate
    print("Generating trade data with 95%+ win rate...")
    trades = generate_trades(num_trades=200, win_rate=0.95)
    
    # Save trades to CSV
    save_trades_to_csv(trades)
    
    # Generate detailed trade report
    generate_detailed_trade_report(trades)
    
    # Generate trade heatmap
    generate_trade_heatmap(trades)
    
    # Generate performance report
    generate_performance_report(trades)
    
    print("\nTrade data generation completed!")
