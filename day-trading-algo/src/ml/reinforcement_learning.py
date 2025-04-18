#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reinforcement Learning module for the day trading algorithm.

This module implements reinforcement learning techniques to adapt the trading
strategy based on past performance and market conditions.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from collections import deque
import random
import joblib
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class ReinforcementLearner:
    """
    Reinforcement Learning system that adapts trading parameters based on performance.
    
    This class implements a Q-learning approach to optimize trading decisions
    and parameters based on rewards from trading actions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the reinforcement learning system.
        
        Args:
            config: Configuration dictionary with RL parameters
        """
        self.config = config.get("adaptive_ml", {}).get("reinforcement_learning", {})
        self.enable = self.config.get("enable", True)
        
        if not self.enable:
            logger.info("Reinforcement learning is disabled")
            return
        
        # RL parameters
        self.algorithm = self.config.get("algorithm", "dqn")
        self.reward_function = self.config.get("reward_function", "sharpe")
        self.state_representation = self.config.get("state_representation", "features")
        self.action_space = self.config.get("action_space", "discrete")
        
        # Learning parameters
        self.gamma = self.config.get("gamma", 0.99)  # Discount factor
        self.epsilon = self.config.get("epsilon_start", 1.0)  # Exploration rate
        self.epsilon_min = self.config.get("epsilon_end", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        
        # Memory parameters
        self.memory_size = self.config.get("replay_buffer_size", 10000)
        self.batch_size = self.config.get("batch_size", 64)
        self.memory = deque(maxlen=self.memory_size)
        
        # Model parameters
        self.model_path = os.path.join(
            config.get("adaptive_ml", {}).get("model_path", "models"),
            f"rl_model_{self.algorithm}.joblib"
        )
        
        # Initialize Q-table or model based on algorithm
        self._initialize_model()
        
        # Track performance
        self.total_rewards = []
        self.episode_rewards = 0
        self.trades_count = 0
        
        logger.info(f"Initialized {self.algorithm} reinforcement learning system")
    
    def _initialize_model(self):
        """Initialize the RL model based on the selected algorithm."""
        if self.algorithm == "q_learning":
            # Simple Q-table for discrete state-action spaces
            self.q_table = {}
        elif self.algorithm == "dqn":
            # For Deep Q-Network, we'll use a simple neural network
            # This is a placeholder - in a real implementation, you would use a proper DQN
            self.model = self._create_dqn_model()
            self.target_model = self._create_dqn_model()
            self.update_target_model()
        else:
            logger.warning(f"Unknown RL algorithm: {self.algorithm}, defaulting to q_learning")
            self.algorithm = "q_learning"
            self.q_table = {}
    
    def _create_dqn_model(self):
        """
        Create a simple neural network for DQN.
        
        This is a placeholder. In a real implementation, you would use a proper
        deep learning framework like TensorFlow or PyTorch.
        """
        # Placeholder for a neural network model
        model = {
            "weights": np.random.random((10, 5)),  # Random weights for demonstration
            "bias": np.random.random(5)
        }
        return model
    
    def update_target_model(self):
        """Update target model with weights from the main model."""
        if self.algorithm == "dqn":
            self.target_model = self.model.copy()
    
    def get_state(self, observation: Dict) -> Tuple:
        """
        Convert observation to a state representation.
        
        Args:
            observation: Dictionary of market data and indicators
            
        Returns:
            Tuple: State representation
        """
        if self.state_representation == "features":
            # Extract relevant features for state representation
            state = (
                self._discretize(observation.get("rsi", 50), 0, 100, 10),
                self._discretize(observation.get("macd_hist", 0), -2, 2, 10),
                self._discretize(observation.get("bb_width", 0), 0, 5, 10),
                self._discretize(observation.get("trend_strength", 0), -1, 1, 10),
                self._discretize(observation.get("volume_relative_to_avg", 1), 0, 3, 10)
            )
        else:
            # Default to a simple state representation
            state = (
                self._discretize(observation.get("close_pct_change", 0), -0.05, 0.05, 10),
                self._discretize(observation.get("rsi", 50), 0, 100, 10)
            )
        
        return state
    
    def _discretize(self, value: float, min_val: float, max_val: float, bins: int) -> int:
        """
        Discretize a continuous value into bins.
        
        Args:
            value: Continuous value to discretize
            min_val: Minimum expected value
            max_val: Maximum expected value
            bins: Number of bins
            
        Returns:
            int: Discretized value (bin index)
        """
        if value is None:
            return 0
        
        # Clip value to range
        value = max(min_val, min(value, max_val))
        
        # Discretize
        bin_size = (max_val - min_val) / bins
        if bin_size == 0:
            return 0
        
        bin_index = int((value - min_val) / bin_size)
        
        # Handle edge case
        if bin_index == bins:
            bin_index = bins - 1
            
        return bin_index
    
    def get_action(self, state: Tuple) -> int:
        """
        Get action based on current state using epsilon-greedy policy.
        
        Args:
            state: Current state representation
            
        Returns:
            int: Action index
        """
        if not self.enable:
            return 0  # Default action if RL is disabled
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, 3)  # 0: no action, 1: buy, 2: sell
        
        # Exploit: best known action
        if self.algorithm == "q_learning":
            if state not in self.q_table:
                self.q_table[state] = np.zeros(3)
            return np.argmax(self.q_table[state])
        elif self.algorithm == "dqn":
            # Placeholder for DQN prediction
            # In a real implementation, you would use your neural network to predict Q-values
            state_vector = np.array([list(state)])
            q_values = np.dot(state_vector, self.model["weights"]) + self.model["bias"]
            return np.argmax(q_values[0])
        
        return 0  # Default action
    
    def remember(self, state: Tuple, action: int, reward: float, next_state: Tuple, done: bool):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if not self.enable:
            return
        
        self.memory.append((state, action, reward, next_state, done))
        self.episode_rewards += reward
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def replay(self, batch_size: Optional[int] = None):
        """
        Train the model using experiences from replay memory.
        
        Args:
            batch_size: Number of experiences to sample (default: self.batch_size)
        """
        if not self.enable:
            return
        
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        if self.algorithm == "q_learning":
            # Update Q-table
            for state, action, reward, next_state, done in minibatch:
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(3)
                
                if next_state not in self.q_table:
                    self.q_table[next_state] = np.zeros(3)
                
                # Q-learning update rule
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(self.q_table[next_state])
                
                # Update Q-value
                self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + \
                                             self.learning_rate * target
        
        elif self.algorithm == "dqn":
            # Placeholder for DQN training
            # In a real implementation, you would use your neural network framework's training methods
            pass
    
    def calculate_reward(self, trade_result: Dict) -> float:
        """
        Calculate reward based on trade result.
        
        Args:
            trade_result: Dictionary with trade result information
            
        Returns:
            float: Calculated reward
        """
        if self.reward_function == "profit":
            # Simple profit-based reward
            return trade_result.get("profit_loss", 0)
        
        elif self.reward_function == "sharpe":
            # Sharpe ratio inspired reward
            profit = trade_result.get("profit_loss", 0)
            risk = trade_result.get("max_drawdown", 0.01)  # Avoid division by zero
            
            # Avoid division by zero
            if abs(risk) < 0.0001:
                risk = 0.0001
                
            return profit / abs(risk)
        
        elif self.reward_function == "custom":
            # Custom reward function that considers multiple factors
            profit = trade_result.get("profit_loss", 0)
            duration = trade_result.get("duration_minutes", 1)
            followed_trend = trade_result.get("followed_trend", False)
            good_entry = trade_result.get("good_entry", False)
            good_exit = trade_result.get("good_exit", False)
            
            # Base reward is profit
            reward = profit
            
            # Adjust for trade duration (prefer shorter profitable trades)
            if profit > 0:
                reward *= (1 + 1/max(duration, 1))
            
            # Bonus for following the trend
            if followed_trend:
                reward *= 1.2
            
            # Bonus for good entry and exit
            if good_entry:
                reward *= 1.1
            if good_exit:
                reward *= 1.1
                
            return reward
        
        else:
            # Default to profit
            return trade_result.get("profit_loss", 0)
    
    def end_episode(self):
        """End the current episode and record performance."""
        if not self.enable:
            return
        
        self.total_rewards.append(self.episode_rewards)
        logger.info(f"Episode {len(self.total_rewards)} ended with total reward: {self.episode_rewards:.2f}")
        
        # Reset episode rewards
        self.episode_rewards = 0
        
        # Update target model periodically
        if self.algorithm == "dqn" and len(self.total_rewards) % self.config.get("target_update", 10) == 0:
            self.update_target_model()
    
    def save_model(self):
        """Save the RL model to disk."""
        if not self.enable:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model based on algorithm
        if self.algorithm == "q_learning":
            joblib.dump(self.q_table, self.model_path)
        elif self.algorithm == "dqn":
            joblib.dump(self.model, self.model_path)
        
        logger.info(f"Saved {self.algorithm} model to {self.model_path}")
    
    def load_model(self):
        """Load the RL model from disk."""
        if not self.enable:
            return
        
        if not os.path.exists(self.model_path):
            logger.warning(f"No saved model found at {self.model_path}")
            return False
        
        try:
            # Load model based on algorithm
            if self.algorithm == "q_learning":
                self.q_table = joblib.load(self.model_path)
            elif self.algorithm == "dqn":
                self.model = joblib.load(self.model_path)
                self.update_target_model()
            
            logger.info(f"Loaded {self.algorithm} model from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics of the RL system.
        
        Returns:
            Dict: Performance statistics
        """
        if not self.enable or not self.total_rewards:
            return {}
        
        return {
            "total_episodes": len(self.total_rewards),
            "avg_reward": np.mean(self.total_rewards),
            "max_reward": np.max(self.total_rewards),
            "min_reward": np.min(self.total_rewards),
            "recent_avg_reward": np.mean(self.total_rewards[-10:]) if len(self.total_rewards) >= 10 else np.mean(self.total_rewards),
            "epsilon": self.epsilon
        }


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
            config: Configuration dictionary with adaptive learning parameters
        """
        self.config = config.get("paper_trading", {}).get("learning", {})
        self.enable = self.config.get("enable", True)
        
        if not self.enable:
            logger.info("Adaptive learning is disabled")
            return
        
        # Learning parameters
        self.learn_from_mistakes = self.config.get("learn_from_mistakes", True)
        self.learn_from_successes = self.config.get("learn_from_successes", True)
        self.adaptation_speed = self.config.get("adaptation_speed", 0.1)
        
        # Reward system
        self.reward_system = self.config.get("reward_system", {})
        self.reward_enable = self.reward_system.get("enable", True)
        
        # Mistake recognition
        self.mistake_recognition = self.config.get("mistake_recognition", {})
        self.mistake_enable = self.mistake_recognition.get("enable", True)
        self.mistake_patterns = self.mistake_recognition.get("patterns_to_detect", [])
        
        # Success recognition
        self.success_recognition = self.config.get("success_recognition", {})
        self.success_enable = self.success_recognition.get("enable", True)
        self.success_patterns = self.success_recognition.get("patterns_to_detect", [])
        
        # Feedback loop
        self.feedback_loop = self.config.get("feedback_loop", {})
        self.feedback_enable = self.feedback_loop.get("enable", True)
        self.parameter_adjustment_rate = self.feedback_loop.get("parameter_adjustment_rate", 0.05)
        
        # Initialize memory
        self.mistakes_memory = []
        self.successes_memory = []
        self.parameter_adjustments = {}
        
        logger.info("Initialized adaptive learning system")
    
    def analyze_trade(self, trade_data: Dict) -> Dict:
        """
        Analyze a completed trade to identify mistakes and successes.
        
        Args:
            trade_data: Dictionary with trade information
            
        Returns:
            Dict: Analysis results with identified patterns
        """
        if not self.enable:
            return {"patterns": []}
        
        result = {
            "patterns": [],
            "is_mistake": False,
            "is_success": False,
            "parameter_adjustments": {}
        }
        
        # Basic trade information
        profit_loss = trade_data.get("profit_loss", 0)
        entry_price = trade_data.get("entry_price", 0)
        exit_price = trade_data.get("exit_price", 0)
        direction = trade_data.get("direction", "long")
        entry_time = trade_data.get("entry_time")
        exit_time = trade_data.get("exit_time")
        exit_reason = trade_data.get("exit_reason", "unknown")
        
        # Calculate trade duration
        duration_minutes = 0
        if entry_time and exit_time:
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            if isinstance(exit_time, str):
                exit_time = pd.to_datetime(exit_time)
            
            duration = exit_time - entry_time
            duration_minutes = duration.total_seconds() / 60
        
        # Market data at entry and exit
        entry_data = trade_data.get("entry_data", {})
        exit_data = trade_data.get("exit_data", {})
        
        # Identify mistakes
        if self.learn_from_mistakes and profit_loss < 0:
            result["is_mistake"] = True
            patterns = self._identify_mistake_patterns(
                trade_data, entry_data, exit_data, duration_minutes
            )
            result["patterns"].extend(patterns)
            
            # Record mistake
            self.mistakes_memory.append({
                "trade_id": trade_data.get("id", len(self.mistakes_memory)),
                "timestamp": datetime.now(),
                "patterns": patterns,
                "trade_data": trade_data
            })
        
        # Identify successes
        if self.learn_from_successes and profit_loss > 0:
            result["is_success"] = True
            patterns = self._identify_success_patterns(
                trade_data, entry_data, exit_data, duration_minutes
            )
            result["patterns"].extend(patterns)
            
            # Record success
            self.successes_memory.append({
                "trade_id": trade_data.get("id", len(self.successes_memory)),
                "timestamp": datetime.now(),
                "patterns": patterns,
                "trade_data": trade_data
            })
        
        # Calculate parameter adjustments
        if self.feedback_enable:
            result["parameter_adjustments"] = self._calculate_parameter_adjustments(
                result["patterns"], profit_loss
            )
            
            # Update parameter adjustments
            for param, adjustment in result["parameter_adjustments"].items():
                if param not in self.parameter_adjustments:
                    self.parameter_adjustments[param] = 0
                self.parameter_adjustments[param] += adjustment
        
        return result
    
    def _identify_mistake_patterns(self, trade_data: Dict, entry_data: Dict, 
                                 exit_data: Dict, duration_minutes: float) -> List[str]:
        """
        Identify patterns in trading mistakes.
        
        Args:
            trade_data: Dictionary with trade information
            entry_data: Market data at entry
            exit_data: Market data at exit
            duration_minutes: Trade duration in minutes
            
        Returns:
            List[str]: Identified mistake patterns
        """
        if not self.mistake_enable:
            return []
        
        patterns = []
        
        # Extract trade information
        profit_loss = trade_data.get("profit_loss", 0)
        direction = trade_data.get("direction", "long")
        exit_reason = trade_data.get("exit_reason", "unknown")
        
        # Early entry
        if "early_entry" in self.mistake_patterns:
            # Check if entered too early (before confirmation)
            if (direction == "long" and entry_data.get("rsi", 50) < 40) or \
               (direction == "short" and entry_data.get("rsi", 50) > 60):
                patterns.append("early_entry")
        
        # Late entry
        if "late_entry" in self.mistake_patterns:
            # Check if entered too late (after move has started)
            if (direction == "long" and entry_data.get("rsi", 50) > 70) or \
               (direction == "short" and entry_data.get("rsi", 50) < 30):
                patterns.append("late_entry")
        
        # Early exit
        if "early_exit" in self.mistake_patterns:
            # Check if exited too early (before target)
            if exit_reason == "manual" and abs(profit_loss) < 0.2:
                patterns.append("early_exit")
        
        # Late exit
        if "late_exit" in self.mistake_patterns:
            # Check if exited too late (after reversal)
            if exit_reason == "stop_loss" and abs(profit_loss) > 1.0:
                patterns.append("late_exit")
        
        # Counter trend trade
        if "counter_trend_trade" in self.mistake_patterns:
            # Check if traded against the trend
            if (direction == "long" and entry_data.get("trend", "sideways") == "downtrend") or \
               (direction == "short" and entry_data.get("trend", "sideways") == "uptrend"):
                patterns.append("counter_trend_trade")
        
        # Overtrading
        if "overtrading" in self.mistake_patterns:
            # Check if trading too frequently
            if duration_minutes < 5:
                patterns.append("overtrading")
        
        # Position sizing error
        if "position_sizing_error" in self.mistake_patterns:
            # Check if position size was too large
            if trade_data.get("size", 0) > trade_data.get("recommended_size", float('inf')):
                patterns.append("position_sizing_error")
        
        return patterns
    
    def _identify_success_patterns(self, trade_data: Dict, entry_data: Dict, 
                                 exit_data: Dict, duration_minutes: float) -> List[str]:
        """
        Identify patterns in trading successes.
        
        Args:
            trade_data: Dictionary with trade information
            entry_data: Market data at entry
            exit_data: Market data at exit
            duration_minutes: Trade duration in minutes
            
        Returns:
            List[str]: Identified success patterns
        """
        if not self.success_enable:
            return []
        
        patterns = []
        
        # Extract trade information
        profit_loss = trade_data.get("profit_loss", 0)
        direction = trade_data.get("direction", "long")
        exit_reason = trade_data.get("exit_reason", "unknown")
        
        # Optimal entry
        if "optimal_entry" in self.success_patterns:
            # Check if entry was near optimal point
            if (direction == "long" and entry_data.get("rsi", 50) > 30 and entry_data.get("rsi", 50) < 40) or \
               (direction == "short" and entry_data.get("rsi", 50) < 70 and entry_data.get("rsi", 50) > 60):
                patterns.append("optimal_entry")
        
        # Optimal exit
        if "optimal_exit" in self.success_patterns:
            # Check if exit was near optimal point
            if exit_reason == "take_profit" or \
               (exit_reason == "manual" and profit_loss > 0.5):
                patterns.append("optimal_exit")
        
        # Trend following
        if "trend_following" in self.success_patterns:
            # Check if traded with the trend
            if (direction == "long" and entry_data.get("trend", "sideways") == "uptrend") or \
               (direction == "short" and entry_data.get("trend", "sideways") == "downtrend"):
                patterns.append("trend_following")
        
        # Proper position sizing
        if "proper_position_sizing" in self.success_patterns:
            # Check if position size was appropriate
            if abs(trade_data.get("size", 0) - trade_data.get("recommended_size", 0)) < 0.1:
                patterns.append("proper_position_sizing")
        
        # Good risk management
        if "good_risk_management" in self.success_patterns:
            # Check if risk management was good
            if trade_data.get("risk_reward_ratio", 0) > 2.0:
                patterns.append("good_risk_management")
        
        return patterns
    
    def _calculate_parameter_adjustments(self, patterns: List[str], profit_loss: float) -> Dict:
        """
        Calculate parameter adjustments based on identified patterns.
        
        Args:
            patterns: List of identified patterns
            profit_loss: Profit or loss from the trade
            
        Returns:
            Dict: Parameter adjustments
        """
        if not self.feedback_enable:
            return {}
        
        adjustments = {}
        
        # Adjustment strength based on profit/loss
        strength = self.parameter_adjustment_rate
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
            
            elif pattern == "early_exit":
                # Adjust exit parameters to hold longer
                adjustments["profit_taking_threshold"] = 0.1 * strength  # Increase profit target
                adjustments["stop_loss_threshold"] = 0.05 * strength  # Increase stop loss distance
            
            elif pattern == "late_exit":
                # Adjust exit parameters to exit earlier
                adjustments["profit_taking_threshold"] = -0.1 * strength  # Decrease profit target
                adjustments["stop_loss_threshold"] = -0.05 * strength  # Decrease stop loss distance
            
            elif pattern == "counter_trend_trade":
                # Strengthen trend following
                adjustments["trend_ma_period"] = -2 * strength  # Shorter trend MA for responsiveness
            
            elif pattern == "overtrading":
                # Reduce trading frequency
                adjustments["avoid_first_minutes"] = 5 * strength  # Increase time to avoid at open
                adjustments["avoid_last_minutes"] = 5 * strength  # Increase time to avoid at close
            
            elif pattern == "position_sizing_error":
                # Adjust position sizing
                adjustments["max_risk_per_trade"] = -0.001 * strength  # Decrease risk per trade
            
            elif pattern == "optimal_entry":
                # Reinforce good entry parameters
                adjustments["fast_ma_period"] = 0.5 * strength  # Fine-tune MA periods
                adjustments["slow_ma_period"] = 0.5 * strength
            
            elif pattern == "optimal_exit":
                # Reinforce good exit parameters
                adjustments["profit_taking_threshold"] = 0.05 * strength  # Fine-tune profit target
            
            elif pattern == "trend_following":
                # Reinforce trend following
                adjustments["trend_ma_period"] = 1 * strength  # Fine-tune trend MA
            
            elif pattern == "proper_position_sizing":
                # Reinforce good position sizing
                adjustments["max_risk_per_trade"] = 0.0005 * strength  # Fine-tune risk per trade
            
            elif pattern == "good_risk_management":
                # Reinforce good risk management
                adjustments["risk_reward_ratio"] = 0.1 * strength  # Increase risk-reward ratio
        
        return adjustments
    
    def get_parameter_adjustments(self) -> Dict:
        """
        Get the current parameter adjustments.
        
        Returns:
            Dict: Current parameter adjustments
        """
        return self.parameter_adjustments
    
    def reset_parameter_adjustments(self):
        """Reset parameter adjustments."""
        self.parameter_adjustments = {}
    
    def get_learning_stats(self) -> Dict:
        """
        Get statistics about the learning process.
        
        Returns:
            Dict: Learning statistics
        """
        if not self.enable:
            return {}
        
        return {
            "mistakes_count": len(self.mistakes_memory),
            "successes_count": len(self.successes_memory),
            "parameter_adjustments": self.parameter_adjustments,
            "top_mistake_patterns": self._get_top_patterns(self.mistakes_memory),
            "top_success_patterns": self._get_top_patterns(self.successes_memory)
        }
    
    def _get_top_patterns(self, memory: List[Dict], top_n: int = 3) -> Dict:
        """
        Get the top N most frequent patterns.
        
        Args:
            memory: List of memory entries
            top_n: Number of top patterns to return
            
        Returns:
            Dict: Top patterns and their counts
        """
        pattern_counts = {}
        
        for entry in memory:
            for pattern in entry.get("patterns", []):
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = 0
                pattern_counts[pattern] += 1
        
        # Sort by count
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N
        return dict(sorted_patterns[:top_n])
