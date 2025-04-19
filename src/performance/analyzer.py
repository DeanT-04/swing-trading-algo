#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance analysis for trading strategies.

This module provides functionality to analyze trading performance,
calculate performance metrics, and generate reports.
"""

import logging
import numpy as np
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes trading performance and generates reports.
    """

    def __init__(self, config: Dict):
        """
        Initialize the performance analyzer with configuration parameters.

        Args:
            config: Performance analyzer configuration dictionary
        """
        self.config = config

        # Extract configuration parameters
        self.metrics = config.get("metrics", ["win_rate", "profit_factor", "max_drawdown", "sharpe_ratio"])
        self.reporting_frequency = config.get("reporting_frequency", "daily")
        self.save_to_file = config.get("save_to_file", True)
        self.plot_equity_curve = config.get("plot_equity_curve", True)

        logger.info(f"Initialized PerformanceAnalyzer with metrics: {self.metrics}")

    def analyze_trades(self, trades: List[Dict], initial_balance: float) -> Dict:
        """
        Analyze a list of trades and calculate performance metrics.

        Args:
            trades: List of trade dictionaries
            initial_balance: Initial account balance

        Returns:
            Dict: Dictionary of performance metrics
        """
        if not trades:
            logger.warning("No trades to analyze")
            return self._empty_metrics()

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)

        # Filter out open trades
        closed_trades = df[~df["is_open"]]

        if closed_trades.empty:
            logger.warning("No closed trades to analyze")
            return self._empty_metrics()

        # Calculate basic metrics
        metrics = {}

        # Win rate
        if "win_rate" in self.metrics:
            winning_trades = closed_trades[closed_trades["profit_loss"] > 0]
            metrics["win_rate"] = len(winning_trades) / len(closed_trades)

        # Profit factor
        if "profit_factor" in self.metrics:
            total_profit = closed_trades[closed_trades["profit_loss"] > 0]["profit_loss"].sum()
            total_loss = abs(closed_trades[closed_trades["profit_loss"] < 0]["profit_loss"].sum())
            metrics["profit_factor"] = total_profit / total_loss if total_loss > 0 else float('inf')

        # Average win and loss
        if "average_win" in self.metrics:
            winning_trades = closed_trades[closed_trades["profit_loss"] > 0]
            metrics["average_win"] = winning_trades["profit_loss"].mean() if not winning_trades.empty else 0.0

        if "average_loss" in self.metrics:
            losing_trades = closed_trades[closed_trades["profit_loss"] < 0]
            metrics["average_loss"] = losing_trades["profit_loss"].mean() if not losing_trades.empty else 0.0

        # Maximum drawdown
        if "max_drawdown" in self.metrics or "max_drawdown_percent" in self.metrics:
            # Sort trades by exit time
            sorted_trades = closed_trades.sort_values("exit_time")

            # Calculate cumulative balance
            balance = initial_balance
            balances = [initial_balance]

            for _, trade in sorted_trades.iterrows():
                balance += trade["profit_loss"]
                balances.append(balance)

            # Calculate drawdown
            max_balance = initial_balance
            max_drawdown = 0.0

            for b in balances:
                max_balance = max(max_balance, b)
                drawdown = max_balance - b
                max_drawdown = max(max_drawdown, drawdown)

            metrics["max_drawdown"] = max_drawdown
            metrics["max_drawdown_percent"] = (max_drawdown / max_balance) * 100 if max_balance > 0 else 0.0

        # Sharpe ratio
        if "sharpe_ratio" in self.metrics:
            # Sort trades by exit time
            sorted_trades = closed_trades.sort_values("exit_time")

            # Calculate daily returns
            returns = []

            if not sorted_trades.empty:
                # Group trades by day
                sorted_trades["exit_date"] = sorted_trades["exit_time"].dt.date
                daily_pnl = sorted_trades.groupby("exit_date")["profit_loss"].sum()

                # Calculate daily returns
                balance = initial_balance
                for date, pnl in daily_pnl.items():
                    returns.append(pnl / balance)
                    balance += pnl

            # Calculate Sharpe ratio
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                metrics["sharpe_ratio"] = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
            else:
                metrics["sharpe_ratio"] = 0.0

        # Total return
        metrics["total_return"] = closed_trades["profit_loss"].sum()
        metrics["total_return_percent"] = (metrics["total_return"] / initial_balance) * 100

        # Number of trades
        metrics["total_trades"] = len(closed_trades)
        metrics["winning_trades"] = len(closed_trades[closed_trades["profit_loss"] > 0])
        metrics["losing_trades"] = len(closed_trades[closed_trades["profit_loss"] < 0])

        # Average trade duration
        if "duration" in closed_trades.columns:
            metrics["average_duration"] = closed_trades["duration"].mean()

        return metrics

    def generate_report(self, trades: List[Dict], initial_balance: float, output_dir: Optional[str] = None) -> str:
        """
        Generate a performance report.

        Args:
            trades: List of trade dictionaries
            initial_balance: Initial account balance
            output_dir: Directory to save the report (optional)

        Returns:
            str: Report content
        """
        # Analyze trades
        metrics = self.analyze_trades(trades, initial_balance)

        # Generate report content
        report = "# Trading Performance Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report += "## Performance Metrics\n\n"
        report += f"- Initial Balance: £{initial_balance:.2f}\n"
        report += f"- Final Balance: £{initial_balance + metrics['total_return']:.2f}\n"
        report += f"- Total Return: £{metrics['total_return']:.2f} ({metrics['total_return_percent']:.2f}%)\n"
        report += f"- Total Trades: {metrics['total_trades']}\n"
        report += f"- Winning Trades: {metrics['winning_trades']} ({metrics['win_rate']*100:.2f}%)\n"
        report += f"- Losing Trades: {metrics['losing_trades']}\n"

        if "profit_factor" in metrics:
            report += f"- Profit Factor: {metrics['profit_factor']:.2f}\n"

        if "average_win" in metrics:
            report += f"- Average Win: £{metrics['average_win']:.2f}\n"

        if "average_loss" in metrics:
            report += f"- Average Loss: £{metrics['average_loss']:.2f}\n"

        if "max_drawdown" in metrics:
            report += f"- Maximum Drawdown: £{metrics['max_drawdown']:.2f} ({metrics['max_drawdown_percent']:.2f}%)\n"

        if "sharpe_ratio" in metrics:
            report += f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"

        if "average_duration" in metrics:
            report += f"- Average Trade Duration: {metrics['average_duration']:.2f} days\n"

        report += "\n## Trade List\n\n"
        report += "| Symbol | Direction | Entry Time | Exit Time | Entry Price | Exit Price | P/L | P/L % | Reason |\n"
        report += "|--------|-----------|------------|-----------|-------------|------------|-----|-------|--------|\n"

        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda x: x["exit_time"] if x["exit_time"] else datetime.max)

        for trade in sorted_trades:
            if not trade["is_open"]:
                report += f"| {trade['symbol']} | {trade['direction']} | "
                report += f"{trade['entry_time'].strftime('%Y-%m-%d')} | {trade['exit_time'].strftime('%Y-%m-%d')} | "
                report += f"£{trade['entry_price']:.4f} | £{trade['exit_price']:.4f} | "
                report += f"£{trade['profit_loss']:.2f} | {trade['profit_loss_percent']:.2f}% | {trade['exit_reason']} |\n"

        # Save report to file
        if self.save_to_file and output_dir:
            output_path = Path(output_dir)
            os.makedirs(output_path, exist_ok=True)

            report_file = output_path / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, "w") as f:
                f.write(report)

            logger.info(f"Performance report saved to {report_file}")

            # Generate equity curve plot
            if self.plot_equity_curve:
                self._plot_equity_curve(trades, initial_balance, output_path)

        return report

    def _empty_metrics(self) -> Dict:
        """
        Return empty metrics dictionary.

        Returns:
            Dict: Empty metrics dictionary
        """
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "total_return_percent": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }

    def _plot_equity_curve(self, trades: List[Dict], initial_balance: float, output_dir: Path) -> None:
        """
        Plot equity curve and save to file.

        Args:
            trades: List of trade dictionaries
            initial_balance: Initial account balance
            output_dir: Directory to save the plot
        """
        # Filter out open trades
        closed_trades = [t for t in trades if not t["is_open"]]

        if not closed_trades:
            logger.warning("No closed trades to plot equity curve")
            return

        # Sort trades by exit time
        sorted_trades = sorted(closed_trades, key=lambda x: x["exit_time"])

        # Calculate equity curve
        dates = [datetime.now()]  # Start with current date
        equity = [initial_balance]  # Start with initial balance

        for trade in sorted_trades:
            dates.append(trade["exit_time"])
            equity.append(equity[-1] + trade["profit_loss"])

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, label="Equity Curve")
        plt.axhline(y=initial_balance, color="r", linestyle="--", label="Initial Balance")

        # Add labels and title
        plt.xlabel("Date")
        plt.ylabel("Equity (£)")
        plt.title("Trading Equity Curve")
        plt.legend()
        plt.grid(True)

        # Format x-axis dates
        plt.gcf().autofmt_xdate()

        # Save plot
        plot_file = output_dir / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file)
        plt.close()

        logger.info(f"Equity curve plot saved to {plot_file}")
