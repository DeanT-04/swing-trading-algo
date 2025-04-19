#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Console UI for the day trading algorithm.

This module provides a rich console UI for displaying trading information
in a more visually appealing and informative way.
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Any

# Try to import rich for enhanced console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.live import Live
    from rich.box import Box, ROUNDED
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not found. Install with: pip install rich")
    print("Falling back to basic console output.")


class ConsoleUI:
    """Enhanced console UI for the day trading algorithm."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the console UI.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.running = False
        self.update_interval = self.config.get("update_interval", 1)  # seconds
        
        # Trading data
        self.account_balance = 50.0  # Default starting balance
        self.initial_balance = 50.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.active_trades = {}
        self.trade_history = []
        self.market_status = "CLOSED"
        self.algorithm_status = "OFFLINE"
        
        # Market hours (US Eastern Time)
        self.market_open_time = (9, 30)  # 9:30 AM ET
        self.market_close_time = (16, 0)  # 4:00 PM ET
        
        # Initialize rich console if available
        if RICH_AVAILABLE:
            self.console = Console()
            self.layout = self._create_layout()
            self.live = None
        
        # Thread for updating the UI
        self.update_thread = None
    
    def start(self):
        """Start the console UI."""
        if self.running:
            return
        
        self.running = True
        
        if RICH_AVAILABLE:
            # Start the live display
            self.live = Live(self.layout, refresh_per_second=1, screen=True)
            self.live.start()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop(self):
        """Stop the console UI."""
        self.running = False
        
        if RICH_AVAILABLE and self.live:
            self.live.stop()
        
        if self.update_thread:
            self.update_thread.join(timeout=2)
    
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
            self.winning_trades = int((stats.get("win_rate", 0) * self.total_trades) / 100) if self.total_trades > 0 else 0
            self.losing_trades = self.total_trades - self.winning_trades
        
        elif data_type == "trade_opened":
            trade = data.get("trade", {})
            trade_id = trade.get("id")
            if trade_id:
                self.active_trades[trade_id] = trade
        
        elif data_type == "trade_closed":
            trade = data.get("trade", {})
            trade_id = trade.get("id")
            if trade_id and trade_id in self.active_trades:
                del self.active_trades[trade_id]
                # Add to trade history (limited to last 10)
                self.trade_history.insert(0, trade)
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
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            self.market_status = "CLOSED (Weekend)"
            return
        
        # Create datetime objects for today's trading hours
        market_open = now.replace(hour=self.market_open_time[0], minute=self.market_open_time[1], second=0, microsecond=0)
        market_close = now.replace(hour=self.market_close_time[0], minute=self.market_close_time[1], second=0, microsecond=0)
        
        # Check if current time is within trading hours
        if market_open <= now <= market_close:
            self.market_status = "OPEN"
            # Calculate time until market close
            time_to_close = market_close - now
            hours, remainder = divmod(time_to_close.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.market_status = f"OPEN (Closes in {hours}h {minutes}m)"
        else:
            self.market_status = "CLOSED"
            
            # Calculate time until market open
            if now < market_open:
                time_to_open = market_open - now
            else:
                # Market already closed today, calculate time until tomorrow's open
                next_day = now + timedelta(days=1)
                next_day = next_day.replace(hour=self.market_open_time[0], minute=self.market_open_time[1], second=0, microsecond=0)
                time_to_open = next_day - now
            
            hours, remainder = divmod(time_to_open.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.market_status = f"CLOSED (Opens in {hours}h {minutes}m)"
    
    def _create_layout(self):
        """Create the layout for the rich console."""
        if not RICH_AVAILABLE:
            return None
        
        layout = Layout()
        
        # Split into top and bottom sections
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Split main section into left and right
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        # Split left section into account info and active trades
        layout["left"].split(
            Layout(name="account_info", size=8),
            Layout(name="active_trades")
        )
        
        # Split right section into trade history and performance
        layout["right"].split(
            Layout(name="trade_history"),
            Layout(name="performance", size=8)
        )
        
        return layout
    
    def _update_layout(self):
        """Update the layout with current data."""
        if not RICH_AVAILABLE or not self.layout:
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
        header_text = Text()
        header_text.append("Day Trading Algorithm ", style="bold cyan")
        header_text.append(f"Status: ", style="dim")
        
        if self.algorithm_status == "ONLINE":
            header_text.append(self.algorithm_status, style="bold green")
        else:
            header_text.append(self.algorithm_status, style="bold red")
        
        header_text.append(" | ")
        header_text.append(f"Market: ", style="dim")
        
        if "OPEN" in self.market_status:
            header_text.append(self.market_status, style="bold green")
        else:
            header_text.append(self.market_status, style="bold red")
        
        header_text.append(f" | Local: {local_now.strftime('%H:%M:%S')} | Eastern: {eastern_now.strftime('%H:%M:%S ET')}")
        
        return Panel(header_text, box=box.ROUNDED)
    
    def _create_account_panel(self):
        """Create the account info panel."""
        # Calculate profit/loss
        profit_loss = self.account_balance - self.initial_balance
        profit_loss_pct = (profit_loss / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Create account table
        account_table = Table(show_header=False, box=box.SIMPLE)
        account_table.add_column("Metric", style="dim")
        account_table.add_column("Value", style="bold")
        
        account_table.add_row("Balance", f"${self.account_balance:.2f}")
        
        if profit_loss >= 0:
            account_table.add_row("Profit/Loss", f"+${profit_loss:.2f} (+{profit_loss_pct:.2f}%)")
        else:
            account_table.add_row("Profit/Loss", f"-${abs(profit_loss):.2f} ({profit_loss_pct:.2f}%)")
        
        # Calculate allocation
        allocated = sum(trade.get("cost", 0) for trade in self.active_trades.values())
        available = self.account_balance - allocated
        allocation_pct = (allocated / self.account_balance) * 100 if self.account_balance > 0 else 0
        
        account_table.add_row("Allocated", f"${allocated:.2f} ({allocation_pct:.2f}%)")
        account_table.add_row("Available", f"${available:.2f}")
        
        return Panel(account_table, title="Account Information", box=box.ROUNDED)
    
    def _create_active_trades_panel(self):
        """Create the active trades panel."""
        if not self.active_trades:
            return Panel("No active trades", title="Active Trades (0)", box=box.ROUNDED)
        
        # Create active trades table
        active_table = Table(box=box.SIMPLE)
        active_table.add_column("Symbol", style="cyan")
        active_table.add_column("Direction", style="magenta")
        active_table.add_column("Entry Price", justify="right")
        active_table.add_column("Current", justify="right")
        active_table.add_column("P/L", justify="right")
        active_table.add_column("Allocation", justify="right")
        
        for trade_id, trade in self.active_trades.items():
            symbol = trade.get("symbol", "")
            direction = trade.get("direction", "")
            entry_price = trade.get("entry_price", 0.0)
            current_price = trade.get("current_price", 0.0)
            profit_loss = trade.get("profit_loss", 0.0)
            size = trade.get("size", 0.0)
            cost = entry_price * size
            allocation_pct = (cost / self.account_balance) * 100 if self.account_balance > 0 else 0
            
            # Format values
            entry_price_str = f"${entry_price:.2f}"
            current_price_str = f"${current_price:.2f}"
            allocation_str = f"{allocation_pct:.1f}%"
            
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
        
        return Panel(active_table, title=f"Active Trades ({len(self.active_trades)})", box=box.ROUNDED)
    
    def _create_trade_history_panel(self):
        """Create the trade history panel."""
        if not self.trade_history:
            return Panel("No trade history", title="Recent Trades (0)", box=box.ROUNDED)
        
        # Create trade history table
        history_table = Table(box=box.SIMPLE)
        history_table.add_column("Symbol", style="cyan")
        history_table.add_column("Direction", style="magenta")
        history_table.add_column("Entry", justify="right")
        history_table.add_column("Exit", justify="right")
        history_table.add_column("P/L", justify="right")
        history_table.add_column("Result", justify="center")
        
        for trade in self.trade_history:
            symbol = trade.get("symbol", "")
            direction = trade.get("direction", "")
            entry_price = trade.get("entry_price", 0.0)
            exit_price = trade.get("exit_price", 0.0)
            profit_loss = trade.get("profit_loss", 0.0)
            
            # Format values
            entry_price_str = f"${entry_price:.2f}"
            exit_price_str = f"${exit_price:.2f}"
            
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
        
        return Panel(history_table, title=f"Recent Trades ({len(self.trade_history)})", box=box.ROUNDED)
    
    def _create_performance_panel(self):
        """Create the performance panel."""
        # Create performance table
        performance_table = Table(show_header=False, box=box.SIMPLE)
        performance_table.add_column("Metric", style="dim")
        performance_table.add_column("Value", style="bold")
        
        performance_table.add_row("Total Trades", str(self.total_trades))
        
        # Calculate win rate
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        performance_table.add_row("Win Rate", f"{win_rate:.2f}%")
        
        # Add winning and losing trades
        performance_table.add_row("Winning Trades", str(self.winning_trades))
        performance_table.add_row("Losing Trades", str(self.losing_trades))
        
        return Panel(performance_table, title="Performance Metrics", box=box.ROUNDED)
    
    def _create_footer_panel(self):
        """Create the footer panel."""
        footer_text = Text("Press Ctrl+C to exit", style="dim")
        return Panel(footer_text, box=box.ROUNDED)
    
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
        
        print("\nAccount Information:")
        print(f"Balance: ${self.account_balance:.2f}")
        
        if profit_loss >= 0:
            print(f"Profit/Loss: +${profit_loss:.2f} (+{profit_loss_pct:.2f}%)")
        else:
            print(f"Profit/Loss: -${abs(profit_loss):.2f} ({profit_loss_pct:.2f}%)")
        
        # Calculate allocation
        allocated = sum(trade.get("cost", 0) for trade in self.active_trades.values())
        available = self.account_balance - allocated
        allocation_pct = (allocated / self.account_balance) * 100 if self.account_balance > 0 else 0
        
        print(f"Allocated: ${allocated:.2f} ({allocation_pct:.2f}%)")
        print(f"Available: ${available:.2f}")
        
        # Print active trades
        print(f"\nActive Trades ({len(self.active_trades)}):")
        if self.active_trades:
            print(f"{'Symbol':<8} {'Direction':<8} {'Entry':<10} {'Current':<10} {'P/L':<10} {'Allocation':<10}")
            print("-" * 60)
            
            for trade_id, trade in self.active_trades.items():
                symbol = trade.get("symbol", "")
                direction = trade.get("direction", "")
                entry_price = trade.get("entry_price", 0.0)
                current_price = trade.get("current_price", 0.0)
                profit_loss = trade.get("profit_loss", 0.0)
                size = trade.get("size", 0.0)
                cost = entry_price * size
                allocation_pct = (cost / self.account_balance) * 100 if self.account_balance > 0 else 0
                
                # Format values
                entry_price_str = f"${entry_price:.2f}"
                current_price_str = f"${current_price:.2f}"
                allocation_str = f"{allocation_pct:.1f}%"
                
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
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.losing_trades}")
        
        print("\nPress Ctrl+C to exit")


class ConsoleFormatter(logging.Formatter):
    """Custom formatter for console output."""
    
    def __init__(self, fmt=None, datefmt=None, style='%'):
        """Initialize the formatter."""
        super().__init__(fmt, datefmt, style)
        
        # ANSI color codes
        self.colors = {
            'RESET': '\033[0m',
            'BLACK': '\033[30m',
            'RED': '\033[31m',
            'GREEN': '\033[32m',
            'YELLOW': '\033[33m',
            'BLUE': '\033[34m',
            'MAGENTA': '\033[35m',
            'CYAN': '\033[36m',
            'WHITE': '\033[37m',
            'BOLD': '\033[1m',
            'UNDERLINE': '\033[4m'
        }
        
        # Map log levels to colors
        self.level_colors = {
            logging.DEBUG: self.colors['BLUE'],
            logging.INFO: self.colors['GREEN'],
            logging.WARNING: self.colors['YELLOW'],
            logging.ERROR: self.colors['RED'],
            logging.CRITICAL: self.colors['RED'] + self.colors['BOLD']
        }
    
    def format(self, record):
        """Format the log record."""
        # Get the original formatted message
        message = super().format(record)
        
        # Add color based on log level
        if record.levelno in self.level_colors:
            message = self.level_colors[record.levelno] + message + self.colors['RESET']
        
        return message


def setup_console_formatter():
    """Set up the console formatter for the root logger."""
    # Get the root logger
    logger = logging.getLogger()
    
    # Find the console handler
    console_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            console_handler = handler
            break
    
    # If no console handler found, create one
    if not console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(console_handler)
    
    # Create and set the formatter
    formatter = ConsoleFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)


def create_console_ui(config: Optional[Dict] = None) -> ConsoleUI:
    """
    Create a console UI instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ConsoleUI: Console UI instance
    """
    # Set up the console formatter
    setup_console_formatter()
    
    # Create and return the console UI
    return ConsoleUI(config)


if __name__ == "__main__":
    # Test the console UI
    console_ui = create_console_ui()
    console_ui.start()
    
    try:
        # Simulate some data updates
        console_ui.update_data("status", {"status": "ONLINE"})
        console_ui.update_data("statistics", {"statistics": {"balance": 75.50, "total_trades": 5, "win_rate": 60.0}})
        
        # Add some active trades
        console_ui.update_data("trade_opened", {"trade": {
            "id": "trade1",
            "symbol": "AAPL",
            "direction": "LONG",
            "entry_price": 150.25,
            "current_price": 151.50,
            "profit_loss": 1.25,
            "size": 0.1,
            "cost": 15.025
        }})
        
        console_ui.update_data("trade_opened", {"trade": {
            "id": "trade2",
            "symbol": "MSFT",
            "direction": "SHORT",
            "entry_price": 300.75,
            "current_price": 299.50,
            "profit_loss": 1.25,
            "size": 0.05,
            "cost": 15.0375
        }})
        
        # Wait for a while
        time.sleep(10)
        
        # Close a trade
        console_ui.update_data("trade_closed", {"trade": {
            "id": "trade1",
            "symbol": "AAPL",
            "direction": "LONG",
            "entry_price": 150.25,
            "exit_price": 152.50,
            "profit_loss": 2.25,
            "size": 0.1
        }})
        
        # Wait for user to exit
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        console_ui.stop()
