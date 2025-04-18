#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive learning module for the day trading algorithm.

This module implements a simple adaptive learning system that learns from
trading mistakes and successes to improve the strategy over time.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


class AdaptiveLearningSystem:
    """
    Adaptive learning system that learns from trading mistakes and successes.
    
    This class implements a system that identifies patterns in trading mistakes
    and successes, and adapts the strategy parameters accordingly.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the adaptive learning system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_config = config.get("adaptive_ml", {})
        self.enable = self.ml_config.get("enable", True)
        
        if not self.enable:
            logger.info("Adaptive learning is disabled")
            return
        
        # Learning parameters
        self.learning_rate = self.ml_config.get("learning_rate", 0.05)
        self.memory_window = self.ml_config.get("memory_window", 1000)
        self.retrain_interval = self.ml_config.get("retrain_interval", 5)
        
        # Initialize memory
        self.trade_memory = []
        self.parameter_adjustments = {}
        self.mistake_patterns = defaultdict(int)
        self.success_patterns = defaultdict(int)
        
        # Initialize counters
        self.trades_since_last_update = 0
        
        # Load previous state if available
        self.state_file = os.path.join(
            self.ml_config.get("model_path", "models"),
            f"{self.ml_config.get('model_name', 'adaptive_ml')}_state.json"
        )
        self._load_state()
        
        logger.info("Initialized adaptive learning system")
    
    def analyze_trade(self, trade_data: Dict) -> Dict:
        """
        Analyze a completed trade to identify patterns and learn from it.
        
        Args:
            trade_data: Dictionary with trade information
            
        Returns:
            Dict: Analysis results with identified patterns and parameter adjustments
        """
        if not self.enable:
            return {"patterns": [], "parameter_adjustments": {}}
        
        # Add trade to memory
        self.trade_memory.append(trade_data)
        
        # Limit memory size
        if len(self.trade_memory) > self.memory_window:
            self.trade_memory.pop(0)
        
        # Identify patterns
        patterns = self._identify_patterns(trade_data)
        
        # Calculate parameter adjustments
        parameter_adjustments = self._calculate_parameter_adjustments(patterns, trade_data)
        
        # Update counters
        self.trades_since_last_update += 1
        
        # Check if it's time to update parameters
        if self.trades_since_last_update >= self.retrain_interval:
            self._update_parameters()
            self.trades_since_last_update = 0
        
        # Save state
        self._save_state()
        
        return {
            "patterns": patterns,
            "parameter_adjustments": parameter_adjustments
        }
    
    def _identify_patterns(self, trade_data: Dict) -> List[str]:
        """
        Identify patterns in a trade.
        
        Args:
            trade_data: Dictionary with trade information
            
        Returns:
            List[str]: Identified patterns
        """
        patterns = []
        
        # Extract trade information
        profit_loss = trade_data.get("profit_loss", 0)
        direction = trade_data.get("direction", "long")
        entry_price = trade_data.get("entry_price", 0)
        exit_price = trade_data.get("exit_price", 0)
        exit_reason = trade_data.get("exit_reason", "unknown")
        
        # Entry timing patterns
        if "entry_data" in trade_data:
            entry_data = trade_data["entry_data"]
            
            # Early entry
            if (direction == "long" and entry_data.get("rsi", 50) < 40) or \
               (direction == "short" and entry_data.get("rsi", 50) > 60):
                patterns.append("early_entry")
            
            # Late entry
            if (direction == "long" and entry_data.get("rsi", 50) > 70) or \
               (direction == "short" and entry_data.get("rsi", 50) < 30):
                patterns.append("late_entry")
            
            # Counter trend trade
            if (direction == "long" and entry_data.get("trend", "sideways") == "downtrend") or \
               (direction == "short" and entry_data.get("trend", "sideways") == "uptrend"):
                patterns.append("counter_trend_trade")
        
        # Exit timing patterns
        if exit_reason == "stop_loss":
            patterns.append("stop_loss_hit")
        elif exit_reason == "take_profit":
            patterns.append("take_profit_hit")
        
        # Profit/loss patterns
        if profit_loss > 0:
            patterns.append("profitable_trade")
            
            # Record success patterns
            for pattern in patterns:
                self.success_patterns[pattern] += 1
        else:
            patterns.append("losing_trade")
            
            # Record mistake patterns
            for pattern in patterns:
                self.mistake_patterns[pattern] += 1
        
        return patterns
    
    def _calculate_parameter_adjustments(self, patterns: List[str], trade_data: Dict) -> Dict:
        """
        Calculate parameter adjustments based on identified patterns.
        
        Args:
            patterns: List of identified patterns
            trade_data: Dictionary with trade information
            
        Returns:
            Dict: Parameter adjustments
        """
        adjustments = {}
        
        # Extract trade information
        profit_loss = trade_data.get("profit_loss", 0)
        
        # Adjustment strength based on profit/loss
        strength = self.learning_rate
        if profit_loss != 0:
            # Stronger adjustments for larger profits/losses
            strength *= min(abs(profit_loss) * 2, 3)
        
        # Adjustments based on patterns
        for pattern in patterns:
            if pattern == "early_entry":
                # Adjust entry parameters to be more conservative
                adjustments["rsi_oversold"] = -2 * strength  # Lower oversold threshold
                adjustments["rsi_overbought"] = -2 * strength  # Lower overbought threshold
            
            elif pattern == "late_entry":
                # Adjust entry parameters to be more aggressive
                adjustments["rsi_oversold"] = 2 * strength  # Raise oversold threshold
                adjustments["rsi_overbought"] = 2 * strength  # Raise overbought threshold
            
            elif pattern == "counter_trend_trade" and profit_loss < 0:
                # Strengthen trend following for losing counter-trend trades
                adjustments["trend_ma_period"] = -2 * strength  # Shorter trend MA for responsiveness
            
            elif pattern == "stop_loss_hit":
                # Adjust stop loss parameters
                adjustments["atr_multiplier"] = 0.1 * strength  # Increase ATR multiplier for wider stops
            
            elif pattern == "take_profit_hit":
                # Adjust take profit parameters
                adjustments["risk_reward_ratio"] = 0.1 * strength  # Increase risk-reward ratio
        
        # Update parameter adjustments
        for param, adjustment in adjustments.items():
            if param not in self.parameter_adjustments:
                self.parameter_adjustments[param] = 0
            self.parameter_adjustments[param] += adjustment
        
        return adjustments
    
    def _update_parameters(self):
        """Update strategy parameters based on accumulated adjustments."""
        if not self.parameter_adjustments:
            return
        
        logger.info("Updating strategy parameters based on adaptive learning")
        
        # Log parameter adjustments
        for param, adjustment in self.parameter_adjustments.items():
            logger.info(f"Adjusting {param} by {adjustment:.4f}")
        
        # Reset parameter adjustments
        self.parameter_adjustments = {}
    
    def get_learning_stats(self) -> Dict:
        """
        Get statistics about the learning process.
        
        Returns:
            Dict: Learning statistics
        """
        if not self.enable:
            return {}
        
        # Calculate win rate
        profitable_trades = sum(1 for trade in self.trade_memory if trade.get("profit_loss", 0) > 0)
        win_rate = profitable_trades / len(self.trade_memory) if self.trade_memory else 0
        
        # Get top mistake patterns
        top_mistakes = sorted(self.mistake_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get top success patterns
        top_successes = sorted(self.success_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "trades_analyzed": len(self.trade_memory),
            "win_rate": win_rate,
            "top_mistake_patterns": dict(top_mistakes),
            "top_success_patterns": dict(top_successes),
            "parameter_adjustments": self.parameter_adjustments,
            "trades_since_last_update": self.trades_since_last_update
        }
    
    def _save_state(self):
        """Save the current state of the adaptive learning system."""
        if not self.enable:
            return
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "parameter_adjustments": self.parameter_adjustments,
            "mistake_patterns": dict(self.mistake_patterns),
            "success_patterns": dict(self.success_patterns),
            "trades_since_last_update": self.trades_since_last_update
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        try:
            with open(self.state_file, 'w') as file:
                json.dump(state, file, indent=4)
            logger.info(f"Adaptive learning state saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving adaptive learning state: {e}")
    
    def _load_state(self):
        """Load the state of the adaptive learning system."""
        if not self.enable or not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, 'r') as file:
                state = json.load(file)
            
            self.parameter_adjustments = state.get("parameter_adjustments", {})
            self.mistake_patterns = defaultdict(int, state.get("mistake_patterns", {}))
            self.success_patterns = defaultdict(int, state.get("success_patterns", {}))
            self.trades_since_last_update = state.get("trades_since_last_update", 0)
            
            logger.info(f"Adaptive learning state loaded from {self.state_file}")
        except Exception as e:
            logger.error(f"Error loading adaptive learning state: {e}")


class TradingMistakeRecognizer:
    """
    System for recognizing and learning from trading mistakes.
    
    This class identifies common trading mistakes and provides feedback
    to help improve trading performance.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trading mistake recognizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_config = config.get("adaptive_ml", {})
        self.enable = self.ml_config.get("enable", True)
        
        if not self.enable:
            logger.info("Trading mistake recognizer is disabled")
            return
        
        # Initialize mistake patterns
        self.mistake_patterns = {
            "early_entry": "Entered the trade too early before confirmation",
            "late_entry": "Entered the trade too late after the move has started",
            "counter_trend_trade": "Traded against the prevailing trend",
            "stop_loss_too_tight": "Stop loss was set too close to entry price",
            "stop_loss_too_wide": "Stop loss was set too far from entry price",
            "take_profit_too_close": "Take profit was set too close to entry price",
            "take_profit_too_far": "Take profit was set too far from entry price",
            "overtrading": "Trading too frequently without clear signals",
            "revenge_trading": "Entering a trade to recover a previous loss",
            "position_sizing_error": "Position size was too large for the risk"
        }
        
        logger.info("Initialized trading mistake recognizer")
    
    def analyze_trade(self, trade_data: Dict) -> Dict:
        """
        Analyze a trade to identify mistakes.
        
        Args:
            trade_data: Dictionary with trade information
            
        Returns:
            Dict: Analysis results with identified mistakes and feedback
        """
        if not self.enable:
            return {"mistakes": [], "feedback": ""}
        
        # Identify mistakes
        mistakes = []
        
        # Extract trade information
        profit_loss = trade_data.get("profit_loss", 0)
        direction = trade_data.get("direction", "long")
        entry_price = trade_data.get("entry_price", 0)
        exit_price = trade_data.get("exit_price", 0)
        stop_loss = trade_data.get("stop_loss", 0)
        take_profit = trade_data.get("take_profit", 0)
        exit_reason = trade_data.get("exit_reason", "unknown")
        
        # Only analyze losing trades
        if profit_loss >= 0:
            return {"mistakes": [], "feedback": "This was a profitable trade. Good job!"}
        
        # Check for early entry
        if "entry_data" in trade_data:
            entry_data = trade_data["entry_data"]
            
            if (direction == "long" and entry_data.get("rsi", 50) < 40) or \
               (direction == "short" and entry_data.get("rsi", 50) > 60):
                mistakes.append("early_entry")
            
            # Check for late entry
            if (direction == "long" and entry_data.get("rsi", 50) > 70) or \
               (direction == "short" and entry_data.get("rsi", 50) < 30):
                mistakes.append("late_entry")
            
            # Check for counter trend trade
            if (direction == "long" and entry_data.get("trend", "sideways") == "downtrend") or \
               (direction == "short" and entry_data.get("trend", "sideways") == "uptrend"):
                mistakes.append("counter_trend_trade")
        
        # Check for stop loss issues
        if exit_reason == "stop_loss":
            # Calculate stop loss distance as percentage
            if direction == "long":
                stop_loss_pct = (entry_price - stop_loss) / entry_price * 100
            else:
                stop_loss_pct = (stop_loss - entry_price) / entry_price * 100
            
            if stop_loss_pct < 0.5:
                mistakes.append("stop_loss_too_tight")
            elif stop_loss_pct > 2.0:
                mistakes.append("stop_loss_too_wide")
        
        # Check for take profit issues
        if exit_reason == "take_profit":
            # Calculate take profit distance as percentage
            if direction == "long":
                take_profit_pct = (take_profit - entry_price) / entry_price * 100
            else:
                take_profit_pct = (entry_price - take_profit) / entry_price * 100
            
            if take_profit_pct < 1.0:
                mistakes.append("take_profit_too_close")
            elif take_profit_pct > 5.0:
                mistakes.append("take_profit_too_far")
        
        # Generate feedback
        feedback = self._generate_feedback(mistakes)
        
        return {
            "mistakes": mistakes,
            "feedback": feedback
        }
    
    def _generate_feedback(self, mistakes: List[str]) -> str:
        """
        Generate feedback based on identified mistakes.
        
        Args:
            mistakes: List of identified mistakes
            
        Returns:
            str: Feedback message
        """
        if not mistakes:
            return "No specific mistakes identified. Review the trade for other factors."
        
        feedback = "Trade analysis feedback:\n"
        
        for mistake in mistakes:
            if mistake in self.mistake_patterns:
                feedback += f"- {self.mistake_patterns[mistake]}\n"
        
        feedback += "\nSuggestions for improvement:\n"
        
        if "early_entry" in mistakes:
            feedback += "- Wait for stronger confirmation signals before entering trades\n"
            feedback += "- Look for multiple indicators aligning before entry\n"
        
        if "late_entry" in mistakes:
            feedback += "- Set up alerts for potential entry points\n"
            feedback += "- Develop a watchlist of stocks to monitor for earlier entries\n"
        
        if "counter_trend_trade" in mistakes:
            feedback += "- Focus on trading with the trend, not against it\n"
            feedback += "- Use higher timeframes to confirm the overall trend\n"
        
        if "stop_loss_too_tight" in mistakes:
            feedback += "- Set wider stop losses based on volatility (e.g., using ATR)\n"
            feedback += "- Consider reducing position size to accommodate wider stops\n"
        
        if "stop_loss_too_wide" in mistakes:
            feedback += "- Use more precise stop loss levels based on support/resistance\n"
            feedback += "- Consider increasing position size with tighter stops\n"
        
        return feedback
