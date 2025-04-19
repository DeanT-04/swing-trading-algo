#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced UI for Day Trading Algorithm
This module provides a beautiful console UI for the day trading algorithm
using rich library for advanced terminal formatting.
"""

import os
import time
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.box import ROUNDED, HEAVY, DOUBLE, SIMPLE
    from rich.live import Live
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not available. Install with: pip install rich")
    print("Falling back to basic console UI")


class EnhancedConsoleUI:
    """Enhanced Console UI for Day Trading Algorithm."""

    def __init__(self, initial_balance: float = 50.0, update_interval: float = 1.0):
        """
        Initialize the Enhanced Console UI.

        Args:
            initial_balance: Initial account balance
            update_interval: UI update interval in seconds
        """
        self.initial_balance = initial_balance
        self.account_balance = initial_balance
        self.update_interval = update_interval

        # Algorithm status
        self.algorithm_status = "WAITING"
        self.market_status = "UNKNOWN"

        # Market timing information
        self.market_open_time = "9:30 AM ET"
        self.market_close_time = "4:00 PM ET"
        self.pre_market_time = "4:00 AM - 9:30 AM ET"
        self.after_hours_time = "4:00 PM - 8:00 PM ET"
        self.time_to_market = "Unknown"

        # Trade data
        self.active_trades = []
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Allocated funds
        self.allocated_funds = 0.0

        # UI thread
        self.running = False
        self.ui_thread = None

        # Rich console
        if RICH_AVAILABLE:
            self.console = Console()
            self.layout = self._create_layout()
            self.live = None

    def start(self):
        """Start the UI."""
        if self.running:
            return

        self.running = True

        if RICH_AVAILABLE:
            # Start the live display
            self.live = Live(self.layout, refresh_per_second=4, screen=True)
            self.live.start()

            # Start the update thread
            self.ui_thread = threading.Thread(target=self._update_loop)
            self.ui_thread.daemon = True
            self.ui_thread.start()
        else:
            # Start the update thread for basic console
            self.ui_thread = threading.Thread(target=self._update_loop)
            self.ui_thread.daemon = True
            self.ui_thread.start()

    def stop(self):
        """Stop the UI."""
        self.running = False

        if self.ui_thread:
            self.ui_thread.join(timeout=1.0)

        if RICH_AVAILABLE and self.live:
            self.live.stop()

    def update_data(self, data_type: str, data: Dict):
        """
        Update the UI data.

        Args:
            data_type: Type of data to update (status, trade, statistics, etc.)
            data: Data dictionary
        """
        if data_type == "status":
            self.algorithm_status = data.get("status", "OFFLINE")

        elif data_type == "statistics":
            stats = data.get("statistics", {})
            self.account_balance = stats.get("balance", self.account_balance)
            self.total_trades = stats.get("total_trades", self.total_trades)
            self.winning_trades = stats.get("winning_trades", self.winning_trades)
            self.losing_trades = stats.get("losing_trades", self.losing_trades)

        elif data_type == "trade_opened":
            trade = data.get("trade", {})
            self.active_trades.append(trade)

            # Update allocated funds
            self.allocated_funds += trade.get("allocation", 0.0)

        elif data_type == "trade_closed":
            trade = data.get("trade", {})
            trade_id = trade.get("id", "")

            # Remove from active trades
            for i, active_trade in enumerate(self.active_trades):
                if active_trade.get("id", "") == trade_id:
                    # Update allocated funds
                    self.allocated_funds -= active_trade.get("allocation", 0.0)

                    # Remove from active trades
                    self.active_trades.pop(i)
                    break

            # Add to trade history (at the beginning)
            self.trade_history.insert(0, trade)

            # Limit trade history to 10 entries
            if len(self.trade_history) > 10:
                self.trade_history.pop()

    def _update_loop(self):
        """Update loop for the console UI."""
        while self.running:
            # Update market status
            self._update_market_status()

            if RICH_AVAILABLE:
                # Update the layout
                self._update_layout()
            else:
                # Basic console output
                self._print_basic_info()

            # Sleep for the update interval
            time.sleep(self.update_interval)

    def _update_market_status(self):
        """Update the market status."""
        # Get current time in US Eastern Time
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        # Define market hours
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        pre_market_open = now.replace(hour=4, minute=0, second=0, microsecond=0)
        after_hours_close = now.replace(hour=20, minute=0, second=0, microsecond=0)

        # Store market times for display
        self.market_open_time = "9:30 AM ET"
        self.market_close_time = "4:00 PM ET"
        self.pre_market_time = "4:00 AM - 9:30 AM ET"
        self.after_hours_time = "4:00 PM - 8:00 PM ET"

        # Calculate time until market open/close
        if now < market_open:
            time_diff = market_open - now
            hours, remainder = divmod(time_diff.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_to_market = f"{hours}h {minutes}m until open"
        elif now < market_close:
            time_diff = market_close - now
            hours, remainder = divmod(time_diff.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_to_market = f"{hours}h {minutes}m until close"
        else:
            next_open = market_open + timedelta(days=1)
            if now.weekday() == 4:  # Friday
                next_open = market_open + timedelta(days=3)  # Next Monday
            elif now.weekday() == 5:  # Saturday
                next_open = market_open + timedelta(days=2)  # Next Monday
            elif now.weekday() == 6:  # Sunday
                next_open = market_open + timedelta(days=1)  # Next Monday

            time_diff = next_open - now
            hours, remainder = divmod(time_diff.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_to_market = f"{time_diff.days}d {hours}h {minutes}m until next open"

        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            self.market_status = "CLOSED (Weekend)"
            return

        # Check market hours
        if now < pre_market_open:
            self.market_status = "CLOSED (Overnight)"
        elif now < market_open:
            self.market_status = "PRE-MARKET"
        elif now <= market_close:
            self.market_status = "OPEN"
        elif now <= after_hours_close:
            self.market_status = "AFTER-HOURS"
        else:
            self.market_status = "CLOSED (Overnight)"

    def _create_layout(self):
        """Create the layout for the UI."""
        if not RICH_AVAILABLE:
            return None

        # Create the layout
        layout = Layout(name="root")

        # Split the layout into sections
        layout.split(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        # Split the body into left and right columns
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Split the left column into account info and active trades
        layout["left"].split(
            Layout(name="account_info", size=8),
            Layout(name="active_trades")
        )

        # Split the right column into trade history and performance
        layout["right"].split(
            Layout(name="trade_history", size=7),
            Layout(name="performance", size=8)
        )

        # Update all panels
        layout["header"].update(self._create_header_panel())
        layout["account_info"].update(self._create_account_panel())
        layout["active_trades"].update(self._create_active_trades_panel())
        layout["trade_history"].update(self._create_trade_history_panel())
        layout["performance"].update(self._create_performance_panel())
        layout["footer"].update(self._create_footer_panel())

        return layout

    def _update_layout(self):
        """Update the layout with current data."""
        if not RICH_AVAILABLE or not self.layout or not self.live:
            return

        # Update header
        self.layout["header"].update(self._create_header_panel())

        # Update account info
        self.layout["account_info"].update(self._create_account_panel())

        # Update active trades
        self.layout["active_trades"].update(self._create_active_trades_panel())

        # Update trade history
        self.layout["trade_history"].update(self._create_trade_history_panel())

        # Update performance
        self.layout["performance"].update(self._create_performance_panel())

        # Update footer
        self.layout["footer"].update(self._create_footer_panel())

    def _create_header_panel(self):
        """Create the header panel."""
        # Get current time in local and Eastern time
        local_now = datetime.now()
        eastern = pytz.timezone('US/Eastern')
        eastern_now = datetime.now(eastern)

        # Create header text
        header = Table.grid(padding=0)
        header.add_column(style="bold cyan", justify="center")

        # Create title with fancy formatting
        title = Text()
        title.append("🚀 ", style="bold yellow")
        title.append("MULTI-TIMEFRAME DAY TRADING ALGORITHM", style="bold cyan")
        title.append(" 📈", style="bold green")

        # Add title to header
        header.add_row(title)

        # Create status line
        status_line = Text()
        status_line.append("Status: ", style="dim")

        if self.algorithm_status == "ONLINE":
            status_line.append(self.algorithm_status, style="bold green")
        elif self.algorithm_status == "WAITING":
            status_line.append(self.algorithm_status, style="bold yellow")
        else:
            status_line.append(self.algorithm_status, style="bold red")

        status_line.append(" | ")
        status_line.append("Market: ", style="dim")

        if self.market_status == "OPEN":
            status_line.append(self.market_status, style="bold green")
        elif self.market_status == "PRE-MARKET" or self.market_status == "AFTER-HOURS":
            status_line.append(self.market_status, style="bold yellow")
        else:
            status_line.append(self.market_status, style="bold red")

        status_line.append(" | ")
        status_line.append(f"Local: {local_now.strftime('%H:%M:%S')}", style="dim")
        status_line.append(" | ")
        status_line.append(f"Eastern: {eastern_now.strftime('%H:%M:%S ET')}", style="dim")

        # Add status line to header
        header.add_row(status_line)

        # Create market info line
        market_info = Text()
        market_info.append("Market Hours: ", style="dim")
        market_info.append(f"{self.market_open_time} - {self.market_close_time}", style="bold white")
        market_info.append(" | ")
        market_info.append(f"{self.time_to_market}", style="bold yellow")

        # Add market info line to header
        header.add_row(market_info)

        return Panel(header, box=HEAVY, border_style="cyan")

    def _create_account_panel(self):
        """Create the account info panel."""
        # Calculate profit/loss
        profit_loss = self.account_balance - self.initial_balance
        profit_loss_pct = (profit_loss / self.initial_balance) * 100 if self.initial_balance > 0 else 0

        # Calculate available funds
        available_funds = self.account_balance - self.allocated_funds
        allocated_pct = (self.allocated_funds / self.account_balance) * 100 if self.account_balance > 0 else 0

        # Create account table
        account_table = Table(box=SIMPLE, show_header=False, padding=(1, 2))
        account_table.add_column("Metric", style="dim", width=12)
        account_table.add_column("Value", style="bold")

        # Add balance row with currency symbol
        account_table.add_row("Balance", f"${self.account_balance:.2f}")

        # Add profit/loss row with color
        if profit_loss >= 0:
            account_table.add_row("Profit/Loss", f"[green]+${profit_loss:.2f} (+{profit_loss_pct:.2f}%)[/green]")
        else:
            account_table.add_row("Profit/Loss", f"[red]-${abs(profit_loss):.2f} ({profit_loss_pct:.2f}%)[/red]")

        # Add allocated funds row
        account_table.add_row("Allocated", f"${self.allocated_funds:.2f} ({allocated_pct:.2f}%)")

        # Add available funds row
        account_table.add_row("Available", f"${available_funds:.2f}")

        # Create progress bar for allocated funds
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="green"),
            TextColumn("[bold]{task.percentage:.0f}%"),
            expand=False
        )

        # Add task to progress bar
        task_id = progress.add_task("Allocation", total=100, completed=allocated_pct)

        # Create grid for account info
        account_grid = Table.grid()
        account_grid.add_column()
        account_grid.add_row(account_table)
        account_grid.add_row("")
        account_grid.add_row(progress)

        return Panel(account_grid, title="💰 Account Information", box=ROUNDED, border_style="blue")

    def _create_active_trades_panel(self):
        """Create the active trades panel."""
        if not self.active_trades:
            return Panel("No active trades", title="📊 Active Trades (0)", box=ROUNDED, border_style="green")

        # Create active trades table
        active_table = Table(box=SIMPLE)
        active_table.add_column("Symbol", style="bold")
        active_table.add_column("Direction")
        active_table.add_column("Entry Price")
        active_table.add_column("Current")
        active_table.add_column("P/L")
        active_table.add_column("Allocation")

        for trade in self.active_trades:
            symbol = trade.get("symbol", "")
            direction = trade.get("direction", "")
            entry_price = trade.get("entry_price", 0.0)
            current_price = trade.get("current_price", entry_price)
            allocation = trade.get("allocation", 0.0)

            # Calculate profit/loss
            if direction.upper() == "LONG":
                profit_loss = (current_price - entry_price) * (allocation / entry_price)
            else:
                profit_loss = (entry_price - current_price) * (allocation / entry_price)

            # Format values
            entry_price_str = f"${entry_price:.2f}"
            current_price_str = f"${current_price:.2f}"
            allocation_str = f"${allocation:.2f}"

            # Format direction with arrow
            if direction.upper() == "LONG":
                direction = "[green]▲ LONG[/green]"
            else:
                direction = "[red]▼ SHORT[/red]"

            # Format profit/loss with color
            if profit_loss > 0:
                profit_loss_str = f"[green]+${profit_loss:.2f}[/green]"
            elif profit_loss < 0:
                profit_loss_str = f"[red]-${abs(profit_loss):.2f}[/red]"
            else:
                profit_loss_str = f"${profit_loss:.2f}"

            active_table.add_row(
                symbol, direction, entry_price_str, current_price_str,
                profit_loss_str, allocation_str
            )

        return Panel(active_table, title=f"📊 Active Trades ({len(self.active_trades)})", box=ROUNDED, border_style="green")

    def _create_trade_history_panel(self):
        """Create the trade history panel."""
        if not self.trade_history:
            return Panel("No trade history", title="📜 Recent Trades (0)", box=ROUNDED, border_style="yellow")

        # Create trade history table
        history_table = Table(box=SIMPLE)
        history_table.add_column("Symbol", style="bold")
        history_table.add_column("Direction")
        history_table.add_column("Entry")
        history_table.add_column("Exit")
        history_table.add_column("P/L")
        history_table.add_column("Result")

        for trade in self.trade_history:
            symbol = trade.get("symbol", "")
            direction = trade.get("direction", "")
            entry_price = trade.get("entry_price", 0.0)
            exit_price = trade.get("exit_price", 0.0)
            profit_loss = trade.get("profit_loss", 0.0)

            # Format values
            entry_price_str = f"${entry_price:.2f}"
            exit_price_str = f"${exit_price:.2f}"

            # Format direction with arrow
            if direction.upper() == "LONG":
                direction = "[green]▲ LONG[/green]"
            else:
                direction = "[red]▼ SHORT[/red]"

            # Format profit/loss with color
            if profit_loss > 0:
                profit_loss_str = f"[green]+${profit_loss:.2f}[/green]"
                result = "[green]WIN[/green]"
            elif profit_loss < 0:
                profit_loss_str = f"[red]-${abs(profit_loss):.2f}[/red]"
                result = "[red]LOSE[/red]"
            else:
                profit_loss_str = f"${profit_loss:.2f}"
                result = "EVEN"

            history_table.add_row(
                symbol, direction, entry_price_str, exit_price_str,
                profit_loss_str, result
            )

        return Panel(history_table, title=f"📜 Recent Trades ({len(self.trade_history)})", box=ROUNDED, border_style="yellow")

    def _create_performance_panel(self):
        """Create the performance panel."""
        # Create performance table
        performance_table = Table(box=SIMPLE, show_header=False, padding=(1, 2))
        performance_table.add_column("Metric", style="dim", width=15)
        performance_table.add_column("Value", style="bold")

        performance_table.add_row("Total Trades", str(self.total_trades))

        # Calculate win rate
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0

        # Color the win rate based on value
        if win_rate >= 95:
            win_rate_str = f"[green]{win_rate:.2f}%[/green]"
        elif win_rate >= 70:
            win_rate_str = f"[yellow]{win_rate:.2f}%[/yellow]"
        else:
            win_rate_str = f"[red]{win_rate:.2f}%[/red]"

        performance_table.add_row("Win Rate", win_rate_str)

        # Add winning and losing trades
        performance_table.add_row("Winning Trades", f"[green]{self.winning_trades}[/green]")
        performance_table.add_row("Losing Trades", f"[red]{self.losing_trades}[/red]")

        # Create win rate progress bar
        progress = Progress(
            TextColumn("[bold purple]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="green"),
            TextColumn("[bold]{task.percentage:.0f}%"),
            expand=False
        )

        # Add task to progress bar
        task_id = progress.add_task("Win Rate", total=100, completed=win_rate)

        # Create grid for performance metrics
        performance_grid = Table.grid()
        performance_grid.add_column()
        performance_grid.add_row(performance_table)
        performance_grid.add_row("")
        performance_grid.add_row(progress)

        return Panel(performance_grid, title="📈 Performance Metrics", box=ROUNDED, border_style="magenta")

    def _create_footer_panel(self):
        """Create the footer panel."""
        footer = Table.grid(padding=0)
        footer.add_column(style="dim", justify="center")

        # Add controls
        controls = Text()
        controls.append("Press Ctrl+C to exit", style="dim")

        footer.add_row(controls)

        # Add target win rate
        target = Text()
        target.append("Target win rate: ", style="dim")
        target.append("95%+", style="bold green")
        target.append(" | ")
        target.append("Pre-Market: ", style="dim")
        target.append(f"{self.pre_market_time}", style="bold yellow")
        target.append(" | ")
        target.append("After-Hours: ", style="dim")
        target.append(f"{self.after_hours_time}", style="bold yellow")

        footer.add_row(target)

        return Panel(footer, box=HEAVY, border_style="cyan")

    def _print_basic_info(self):
        """Print basic information to the console (when rich is not available)."""
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')

        # Get current time in local and Eastern time
        local_now = datetime.now()
        eastern = pytz.timezone('US/Eastern')
        eastern_now = datetime.now(eastern)

        # Print header
        print("=" * 80)
        print(f"Day Trading Algorithm | Status: {self.algorithm_status} | Market: {self.market_status}")
        print(f"Local: {local_now.strftime('%H:%M:%S')} | Eastern: {eastern_now.strftime('%H:%M:%S ET')}")
        print("=" * 80)

        # Print account info
        profit_loss = self.account_balance - self.initial_balance
        profit_loss_pct = (profit_loss / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        available_funds = self.account_balance - self.allocated_funds
        allocated_pct = (self.allocated_funds / self.account_balance) * 100 if self.account_balance > 0 else 0

        print("\nAccount Information:")
        print(f"Balance: ${self.account_balance:.2f}")

        if profit_loss >= 0:
            print(f"Profit/Loss: +${profit_loss:.2f} (+{profit_loss_pct:.2f}%)")
        else:
            print(f"Profit/Loss: -${abs(profit_loss):.2f} ({profit_loss_pct:.2f}%)")

        print(f"Allocated: ${self.allocated_funds:.2f} ({allocated_pct:.2f}%)")
        print(f"Available: ${available_funds:.2f}")

        # Print active trades
        print(f"\nActive Trades ({len(self.active_trades)}):")

        if self.active_trades:
            print(f"{'Symbol':<8} {'Direction':<8} {'Entry':<10} {'Current':<10} {'P/L':<10} {'Allocation':<10}")
            print("-" * 60)

            for trade in self.active_trades:
                symbol = trade.get("symbol", "")
                direction = trade.get("direction", "")
                entry_price = trade.get("entry_price", 0.0)
                current_price = trade.get("current_price", entry_price)
                allocation = trade.get("allocation", 0.0)

                # Calculate profit/loss
                if direction.upper() == "LONG":
                    profit_loss = (current_price - entry_price) * (allocation / entry_price)
                else:
                    profit_loss = (entry_price - current_price) * (allocation / entry_price)

                # Format values
                entry_price_str = f"${entry_price:.2f}"
                current_price_str = f"${current_price:.2f}"
                allocation_str = f"${allocation:.2f}"

                # Format profit/loss
                if profit_loss > 0:
                    profit_loss_str = f"+${profit_loss:.2f}"
                elif profit_loss < 0:
                    profit_loss_str = f"-${abs(profit_loss):.2f}"
                else:
                    profit_loss_str = f"${profit_loss:.2f}"

                print(f"{symbol:<8} {direction:<8} {entry_price_str:<10} {current_price_str:<10} {profit_loss_str:<10} {allocation_str:<10}")
        else:
            print("No active trades")

        # Print trade history
        print(f"\nRecent Trades ({len(self.trade_history)}):")

        if self.trade_history:
            print(f"{'Symbol':<8} {'Direction':<8} {'Entry':<10} {'Exit':<10} {'P/L':<10} {'Result':<6}")
            print("-" * 60)

            for trade in self.trade_history:
                symbol = trade.get("symbol", "")
                direction = trade.get("direction", "")
                entry_price = trade.get("entry_price", 0.0)
                exit_price = trade.get("exit_price", 0.0)
                profit_loss = trade.get("profit_loss", 0.0)

                # Format values
                entry_price_str = f"${entry_price:.2f}"
                exit_price_str = f"${exit_price:.2f}"

                # Format profit/loss
                if profit_loss > 0:
                    profit_loss_str = f"+${profit_loss:.2f}"
                    result = "WIN"
                elif profit_loss < 0:
                    profit_loss_str = f"-${abs(profit_loss):.2f}"
                    result = "LOSE"
                else:
                    profit_loss_str = f"${profit_loss:.2f}"
                    result = "EVEN"

                print(f"{symbol:<8} {direction:<8} {entry_price_str:<10} {exit_price_str:<10} {profit_loss_str:<10} {result:<6}")
        else:
            print("No trade history")

        # Print performance metrics
        print("\nPerformance Metrics:")
        print(f"Total Trades: {self.total_trades}")

        # Calculate win rate
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        print(f"Win Rate: {win_rate:.2f}%")

        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.losing_trades}")

        # Print footer
        print("\nPress Ctrl+C to exit")


def main():
    """Main function for the Enhanced Console UI."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Day Trading Algorithm UI')
    parser.add_argument('-test', action='store_true', help='Run in test mode with simulated trades')
    parser.add_argument('-timeframe', type=str, default='1m', help='Trading timeframe (1m, 5m)')
    args = parser.parse_args()

    # Create UI
    ui = EnhancedConsoleUI()

    # Start the UI
    ui.start()

    try:
        # Update status
        ui.update_data("status", {"status": "ONLINE"})

        if args.test:
            # Simulate some trading activity in test mode
            time.sleep(1)
            print(f"Running in TEST mode with {args.timeframe} timeframe")

            # Add some active trades
            ui.update_data("trade_opened", {
                "trade": {
                    "id": "1",
                    "symbol": "AAPL",
                    "direction": "LONG",
                    "entry_price": 175.50,
                    "current_price": 176.25,
                    "allocation": 5.0
                }
            })

            time.sleep(2)

            ui.update_data("trade_opened", {
                "trade": {
                    "id": "2",
                    "symbol": "MSFT",
                    "direction": "LONG",
                    "entry_price": 325.75,
                    "current_price": 327.50,
                    "allocation": 5.0
                }
            })

            time.sleep(2)

            # Close a trade
            ui.update_data("trade_closed", {
                "trade": {
                    "id": "1",
                    "symbol": "AAPL",
                    "direction": "LONG",
                    "entry_price": 175.50,
                    "exit_price": 177.25,
                    "profit_loss": 0.50,
                    "allocation": 5.0
                }
            })

            # Update statistics
            ui.update_data("statistics", {
                "statistics": {
                    "balance": 50.50,
                    "total_trades": 1,
                    "winning_trades": 1,
                    "losing_trades": 0
                }
            })
        else:
            # In real mode, connect to the actual trading algorithm
            print(f"Running with {args.timeframe} timeframe")
            # Here we would connect to the actual trading algorithm
            # For now, we'll just display a waiting message
            ui.update_data("status", {"status": "WAITING"})

        # Keep the UI running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        # Stop the UI
        ui.stop()
        print("\nUI stopped by user")


if __name__ == "__main__":
    main()
