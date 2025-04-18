#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading UI Module

This module provides a graphical user interface for the day trading algorithm,
showing when trades are opened and closed, and the algorithm's status.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import time
from datetime import datetime
import pytz
from typing import Dict, List, Optional, Tuple, Union

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


class TradingUI:
    """Trading UI class for displaying algorithm status and trade notifications."""

    def __init__(self, title: str = "Day Trading Algorithm"):
        """
        Initialize the Trading UI.

        Args:
            title: Window title
        """
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("800x600")
        self.root.minsize(800, 600)

        # Set icon if available
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass

        # Create a queue for thread-safe communication
        self.queue = queue.Queue()

        # Create UI elements
        self._create_ui()

        # Start the update loop
        self._update_ui()

        # Flag to indicate if the UI is running
        self.running = True

    def _create_ui(self):
        """Create the UI elements."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create status frame
        status_frame = ttk.LabelFrame(main_frame, text="Algorithm Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        # Create status indicators
        self.status_var = tk.StringVar(value="OFFLINE")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 14, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Create status indicator light
        self.status_canvas = tk.Canvas(status_frame, width=20, height=20)
        self.status_canvas.pack(side=tk.LEFT)
        self.status_light = self.status_canvas.create_oval(2, 2, 18, 18, fill="red")

        # Create time display
        self.time_var = tk.StringVar(value="--:--:--")
        time_label = ttk.Label(status_frame, textvariable=self.time_var, font=("Arial", 12))
        time_label.pack(side=tk.RIGHT, padx=10)

        # Create market status display
        self.market_var = tk.StringVar(value="Market: Closed")
        market_label = ttk.Label(status_frame, textvariable=self.market_var, font=("Arial", 12))
        market_label.pack(side=tk.RIGHT, padx=10)

        # Create statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Trading Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))

        # Create statistics grid
        self.stats_grid = ttk.Frame(stats_frame)
        self.stats_grid.pack(fill=tk.X)

        # Create statistics labels
        ttk.Label(self.stats_grid, text="Total Trades:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, text="Win Rate:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, text="Profit/Loss:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, text="Balance:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, text="Exposure:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, text="Max Position:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, text="Drawdown:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)

        # Create statistics variables
        self.total_trades_var = tk.StringVar(value="0")
        self.win_rate_var = tk.StringVar(value="0.0%")
        self.profit_loss_var = tk.StringVar(value="$0.00")
        self.balance_var = tk.StringVar(value="$50.00")
        self.exposure_var = tk.StringVar(value="0.0%")
        self.max_position_var = tk.StringVar(value="10.0%")
        self.drawdown_var = tk.StringVar(value="0.0%")

        # Create statistics value labels
        ttk.Label(self.stats_grid, textvariable=self.total_trades_var, font=("Arial", 10, "bold")).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, textvariable=self.win_rate_var, font=("Arial", 10, "bold")).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, textvariable=self.profit_loss_var, font=("Arial", 10, "bold")).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, textvariable=self.balance_var, font=("Arial", 10, "bold")).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, textvariable=self.exposure_var, font=("Arial", 10, "bold")).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, textvariable=self.max_position_var, font=("Arial", 10, "bold")).grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.stats_grid, textvariable=self.drawdown_var, font=("Arial", 10, "bold")).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # Create notebook for trades and logs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Create active trades frame
        active_trades_frame = ttk.Frame(notebook, padding=10)
        notebook.add(active_trades_frame, text="Active Trades")

        # Create active trades treeview
        columns = ("symbol", "direction", "entry_time", "entry_price", "current_price", "profit_loss")
        self.active_trades_tree = ttk.Treeview(active_trades_frame, columns=columns, show="headings")

        # Set column headings
        self.active_trades_tree.heading("symbol", text="Symbol")
        self.active_trades_tree.heading("direction", text="Direction")
        self.active_trades_tree.heading("entry_time", text="Entry Time")
        self.active_trades_tree.heading("entry_price", text="Entry Price")
        self.active_trades_tree.heading("current_price", text="Current Price")
        self.active_trades_tree.heading("profit_loss", text="Profit/Loss")

        # Set column widths
        self.active_trades_tree.column("symbol", width=80)
        self.active_trades_tree.column("direction", width=80)
        self.active_trades_tree.column("entry_time", width=150)
        self.active_trades_tree.column("entry_price", width=100)
        self.active_trades_tree.column("current_price", width=100)
        self.active_trades_tree.column("profit_loss", width=100)

        # Add scrollbar
        active_trades_scrollbar = ttk.Scrollbar(active_trades_frame, orient=tk.VERTICAL, command=self.active_trades_tree.yview)
        self.active_trades_tree.configure(yscrollcommand=active_trades_scrollbar.set)

        # Pack treeview and scrollbar
        self.active_trades_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        active_trades_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create trade history frame
        trade_history_frame = ttk.Frame(notebook, padding=10)
        notebook.add(trade_history_frame, text="Trade History")

        # Create trade history treeview
        columns = ("symbol", "direction", "entry_time", "exit_time", "entry_price", "exit_price", "profit_loss", "result")
        self.trade_history_tree = ttk.Treeview(trade_history_frame, columns=columns, show="headings")

        # Set column headings
        self.trade_history_tree.heading("symbol", text="Symbol")
        self.trade_history_tree.heading("direction", text="Direction")
        self.trade_history_tree.heading("entry_time", text="Entry Time")
        self.trade_history_tree.heading("exit_time", text="Exit Time")
        self.trade_history_tree.heading("entry_price", text="Entry Price")
        self.trade_history_tree.heading("exit_price", text="Exit Price")
        self.trade_history_tree.heading("profit_loss", text="Profit/Loss")
        self.trade_history_tree.heading("result", text="Result")

        # Set column widths
        self.trade_history_tree.column("symbol", width=80)
        self.trade_history_tree.column("direction", width=80)
        self.trade_history_tree.column("entry_time", width=150)
        self.trade_history_tree.column("exit_time", width=150)
        self.trade_history_tree.column("entry_price", width=100)
        self.trade_history_tree.column("exit_price", width=100)
        self.trade_history_tree.column("profit_loss", width=100)
        self.trade_history_tree.column("result", width=80)

        # Add scrollbar
        trade_history_scrollbar = ttk.Scrollbar(trade_history_frame, orient=tk.VERTICAL, command=self.trade_history_tree.yview)
        self.trade_history_tree.configure(yscrollcommand=trade_history_scrollbar.set)

        # Pack treeview and scrollbar
        self.trade_history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trade_history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create log frame
        log_frame = ttk.Frame(notebook, padding=10)
        notebook.add(log_frame, text="Log")

        # Create log text widget
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

        # Create control frame
        control_frame = ttk.Frame(main_frame, padding=10)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        # Create control buttons
        self.start_button = ttk.Button(control_frame, text="Start Trading", command=self._on_start)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop Trading", command=self._on_stop)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)

        self.exit_button = ttk.Button(control_frame, text="Exit", command=self._on_exit)
        self.exit_button.pack(side=tk.RIGHT, padx=5)

    def _update_ui(self):
        """Update the UI with new data from the queue."""
        # Process all messages in the queue
        while not self.queue.empty():
            try:
                message = self.queue.get_nowait()
                message_type = message.get("type")

                if message_type == "status":
                    self._update_status(message)
                elif message_type == "trade_opened":
                    self._add_active_trade(message)
                elif message_type == "trade_closed":
                    self._add_trade_history(message)
                    self._remove_active_trade(message)
                elif message_type == "statistics":
                    self._update_statistics(message)
                elif message_type == "log":
                    self._add_log(message)

                self.queue.task_done()
            except queue.Empty:
                break

        # Update time
        self._update_time()

        # Schedule the next update
        self.root.after(100, self._update_ui)

    def _update_time(self):
        """Update the time display."""
        # Get current time in Eastern Time (US market time)
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)

        # Update time display
        self.time_var.set(now.strftime("%H:%M:%S"))

        # Check if market is open
        is_open = self._is_market_open(now)

        # Update market status
        if is_open:
            self.market_var.set("Market: Open")
        else:
            self.market_var.set("Market: Closed")

    def _is_market_open(self, now: datetime) -> bool:
        """
        Check if the US stock market is currently open.

        Args:
            now: Current datetime in Eastern Time

        Returns:
            bool: True if market is open, False otherwise
        """
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False

        # Check if it's between 9:30 AM and 4:00 PM Eastern
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    def _update_status(self, message: dict):
        """
        Update the status display.

        Args:
            message: Status message dictionary
        """
        status = message.get("status", "OFFLINE")
        self.status_var.set(status)

        if status == "ONLINE":
            self.status_canvas.itemconfig(self.status_light, fill="green")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.status_canvas.itemconfig(self.status_light, fill="red")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def _add_active_trade(self, message: dict):
        """
        Add a trade to the active trades treeview.

        Args:
            message: Trade opened message dictionary
        """
        trade = message.get("trade", {})

        # Format values
        symbol = trade.get("symbol", "")
        direction = trade.get("direction", "")
        entry_time = trade.get("entry_time", "")
        entry_price = f"${trade.get('entry_price', 0.0):.2f}"
        current_price = f"${trade.get('current_price', 0.0):.2f}"
        profit_loss = trade.get("profit_loss", 0.0)

        # Format profit/loss with color
        if profit_loss > 0:
            profit_loss_str = f"+${profit_loss:.2f}"
        else:
            profit_loss_str = f"-${abs(profit_loss):.2f}"

        # Add to treeview
        trade_id = trade.get("id", "")
        self.active_trades_tree.insert("", tk.END, iid=trade_id, values=(
            symbol, direction, entry_time, entry_price, current_price, profit_loss_str
        ))

        # Add log entry
        self._add_log({
            "type": "log",
            "level": "INFO",
            "message": f"Trade opened: {direction} {symbol} at {entry_price}"
        })

    def _remove_active_trade(self, message: dict):
        """
        Remove a trade from the active trades treeview.

        Args:
            message: Trade closed message dictionary
        """
        trade = message.get("trade", {})
        trade_id = trade.get("id", "")

        # Remove from treeview
        try:
            self.active_trades_tree.delete(trade_id)
        except:
            pass

    def _add_trade_history(self, message: dict):
        """
        Add a trade to the trade history treeview.

        Args:
            message: Trade closed message dictionary
        """
        trade = message.get("trade", {})

        # Format values
        symbol = trade.get("symbol", "")
        direction = trade.get("direction", "")
        entry_time = trade.get("entry_time", "")
        exit_time = trade.get("exit_time", "")
        entry_price = f"${trade.get('entry_price', 0.0):.2f}"
        exit_price = f"${trade.get('exit_price', 0.0):.2f}"
        profit_loss = trade.get("profit_loss", 0.0)

        # Determine result
        if profit_loss > 0:
            result = "WIN"
            profit_loss_str = f"+${profit_loss:.2f}"
        else:
            result = "LOSE"
            profit_loss_str = f"-${abs(profit_loss):.2f}"

        # Add to treeview
        self.trade_history_tree.insert("", 0, values=(
            symbol, direction, entry_time, exit_time, entry_price, exit_price, profit_loss_str, result
        ))

        # Add log entry
        self._add_log({
            "type": "log",
            "level": "INFO",
            "message": f"Trade closed: {direction} {symbol} at {exit_price}, {result}, P/L: {profit_loss_str}"
        })

    def _update_statistics(self, message: dict):
        """
        Update the statistics display.

        Args:
            message: Statistics message dictionary
        """
        stats = message.get("statistics", {})

        # Update statistics variables
        self.total_trades_var.set(str(stats.get("total_trades", 0)))
        self.win_rate_var.set(f"{stats.get('win_rate', 0.0):.1f}%")

        # Calculate total profit/loss from completed trades
        profit_loss = stats.get("total_profit_loss", 0.0)

        # Format profit/loss display
        if profit_loss >= 0:
            self.profit_loss_var.set(f"+${profit_loss:.2f}")
        else:
            self.profit_loss_var.set(f"-${abs(profit_loss):.2f}")

        # Update balance
        self.balance_var.set(f"${stats.get('balance', 50.0):.2f}")

        # Update risk metrics
        self.exposure_var.set(f"{stats.get('exposure_percent', 0.0):.1f}%")
        self.max_position_var.set(f"{stats.get('max_position_pct', 10.0):.1f}%")
        self.drawdown_var.set(f"{stats.get('drawdown_percent', 0.0):.1f}%")

        # Get labels for risk metrics
        for widget in self.stats_grid.winfo_children():
            if hasattr(widget, 'cget') and widget.cget('textvariable') == str(self.exposure_var):
                exposure_label = widget
            if hasattr(widget, 'cget') and widget.cget('textvariable') == str(self.drawdown_var):
                drawdown_label = widget

        # Color-code risk metrics based on values
        exposure_pct = stats.get('exposure_percent', 0.0)
        if 'exposure_label' in locals():
            if exposure_pct > 80.0:
                exposure_label.configure(foreground="red")
            elif exposure_pct > 50.0:
                exposure_label.configure(foreground="orange")
            else:
                exposure_label.configure(foreground="green")

        drawdown_pct = stats.get('drawdown_percent', 0.0)
        if 'drawdown_label' in locals():
            if drawdown_pct > 10.0:
                drawdown_label.configure(foreground="red")
            elif drawdown_pct > 5.0:
                drawdown_label.configure(foreground="orange")
            else:
                drawdown_label.configure(foreground="green")

    def _add_log(self, message: dict):
        """
        Add a message to the log.

        Args:
            message: Log message dictionary
        """
        log_message = message.get("message", "")
        level = message.get("level", "INFO")

        # Get current time
        now = datetime.now().strftime("%H:%M:%S")

        # Format log message
        formatted_message = f"[{now}] [{level}] {log_message}\n"

        # Add to log text widget
        self.log_text.config(state=tk.NORMAL)

        # Set tag for color based on level
        if level == "ERROR":
            self.log_text.insert(tk.END, formatted_message, "error")
            self.log_text.tag_configure("error", foreground="red")
        elif level == "WARNING":
            self.log_text.insert(tk.END, formatted_message, "warning")
            self.log_text.tag_configure("warning", foreground="orange")
        else:
            self.log_text.insert(tk.END, formatted_message)

        # Scroll to end
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _on_start(self):
        """Handle start button click."""
        # Add to queue
        self.queue.put({
            "type": "command",
            "command": "start"
        })

        # Update status
        self.queue.put({
            "type": "status",
            "status": "ONLINE"
        })

        # Add log entry
        self._add_log({
            "type": "log",
            "level": "INFO",
            "message": "Trading started"
        })

    def _on_stop(self):
        """Handle stop button click."""
        # Add to queue
        self.queue.put({
            "type": "command",
            "command": "stop"
        })

        # Update status
        self.queue.put({
            "type": "status",
            "status": "OFFLINE"
        })

        # Add log entry
        self._add_log({
            "type": "log",
            "level": "INFO",
            "message": "Trading stopped"
        })

    def _on_exit(self):
        """Handle exit button click."""
        # Set running flag to False
        self.running = False

        # Add to queue
        self.queue.put({
            "type": "command",
            "command": "exit"
        })

        # Destroy root window
        self.root.destroy()

    def update_active_trade(self, trade_id: str, current_price: float, profit_loss: float):
        """
        Update an active trade with current price and profit/loss.

        Args:
            trade_id: Trade ID
            current_price: Current price
            profit_loss: Current profit/loss
        """
        # Format values
        current_price_str = f"${current_price:.2f}"

        if profit_loss > 0:
            profit_loss_str = f"+${profit_loss:.2f}"
        else:
            profit_loss_str = f"-${abs(profit_loss):.2f}"

        # Update treeview
        try:
            item = self.active_trades_tree.item(trade_id)
            values = list(item["values"])
            values[4] = current_price_str
            values[5] = profit_loss_str
            self.active_trades_tree.item(trade_id, values=values)
        except:
            pass

    def add_message(self, message: dict):
        """
        Add a message to the queue.

        Args:
            message: Message dictionary
        """
        self.queue.put(message)

    def run(self):
        """Run the UI main loop."""
        self.root.mainloop()


def create_ui_thread() -> Tuple[TradingUI, threading.Thread]:
    """
    Create a UI thread.

    Returns:
        Tuple[TradingUI, threading.Thread]: UI instance and thread
    """
    # Create UI queue
    ui_queue = queue.Queue()

    # Create UI in a separate thread
    def ui_thread_func():
        ui = TradingUI()
        ui.run()

    ui_thread = threading.Thread(target=ui_thread_func)
    ui_thread.daemon = True
    ui_thread.start()

    # Create UI instance
    ui = TradingUI()

    return ui, ui_thread


if __name__ == "__main__":
    # Create UI
    ui = TradingUI()

    # Add some test data
    ui.add_message({
        "type": "status",
        "status": "ONLINE"
    })

    ui.add_message({
        "type": "statistics",
        "statistics": {
            "total_trades": 10,
            "win_rate": 80.0,
            "profit_loss": 25.75,
            "balance": 75.75
        }
    })

    ui.add_message({
        "type": "trade_opened",
        "trade": {
            "id": "trade1",
            "symbol": "AAPL",
            "direction": "LONG",
            "entry_time": "2023-05-01 10:30:00",
            "entry_price": 150.25,
            "current_price": 151.50,
            "profit_loss": 1.25
        }
    })

    ui.add_message({
        "type": "trade_opened",
        "trade": {
            "id": "trade2",
            "symbol": "MSFT",
            "direction": "SHORT",
            "entry_time": "2023-05-01 11:15:00",
            "entry_price": 300.75,
            "current_price": 299.50,
            "profit_loss": 1.25
        }
    })

    ui.add_message({
        "type": "trade_closed",
        "trade": {
            "id": "trade1",
            "symbol": "AAPL",
            "direction": "LONG",
            "entry_time": "2023-05-01 10:30:00",
            "exit_time": "2023-05-01 14:45:00",
            "entry_price": 150.25,
            "exit_price": 152.75,
            "profit_loss": 2.50
        }
    })

    ui.add_message({
        "type": "log",
        "level": "INFO",
        "message": "Algorithm started"
    })

    ui.add_message({
        "type": "log",
        "level": "WARNING",
        "message": "API rate limit approaching"
    })

    ui.add_message({
        "type": "log",
        "level": "ERROR",
        "message": "Failed to fetch data for TSLA"
    })

    # Run UI
    ui.run()
