#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine learning module for trading strategy improvement.

This module provides functionality to train and use machine learning models
to improve trading strategy performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.data.models import Stock, TimeFrame, TradeDirection

logger = logging.getLogger(__name__)


class MLModel:
    """
    Machine learning model for trading strategy improvement.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the machine learning model with configuration parameters.
        
        Args:
            config: ML model configuration dictionary
        """
        self.config = config
        
        # Extract configuration parameters
        self.model_type = config.get("model_type", "random_forest")
        self.test_size = config.get("test_size", 0.2)
        self.random_state = config.get("random_state", 42)
        self.n_estimators = config.get("n_estimators", 100)
        self.max_depth = config.get("max_depth", 10)
        self.min_samples_split = config.get("min_samples_split", 2)
        self.min_samples_leaf = config.get("min_samples_leaf", 1)
        self.prediction_threshold = config.get("prediction_threshold", 0.6)
        self.feature_importance_threshold = config.get("feature_importance_threshold", 0.01)
        
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        logger.info(f"Initialized MLModel with model type: {self.model_type}")
    
    def _create_model(self):
        """
        Create a machine learning model based on the configuration.
        
        Returns:
            object: Machine learning model
        """
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _prepare_features(self, stock: Stock, timeframe: TimeFrame) -> pd.DataFrame:
        """
        Prepare features for the machine learning model.
        
        Args:
            stock: Stock object with price data
            timeframe: Timeframe for analysis
            
        Returns:
            pd.DataFrame: DataFrame with features
        """
        # Get price data as DataFrame
        df = stock.get_dataframe(timeframe)
        
        if df.empty:
            logger.warning(f"No data available for {stock.symbol} on {timeframe.value} timeframe")
            return pd.DataFrame()
        
        # Create features DataFrame
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['close'] = df['close']
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume']
        
        # Calculate returns
        features['return_1d'] = df['close'].pct_change(1)
        features['return_5d'] = df['close'].pct_change(5)
        features['return_10d'] = df['close'].pct_change(10)
        features['return_20d'] = df['close'].pct_change(20)
        
        # Calculate moving averages
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Calculate moving average ratios
        features['sma_ratio_5_20'] = features['sma_5'] / features['sma_20']
        features['sma_ratio_5_50'] = features['sma_5'] / features['sma_50']
        features['sma_ratio_20_50'] = features['sma_20'] / features['sma_50']
        features['sma_ratio_50_200'] = features['sma_50'] / features['sma_200']
        
        # Calculate price relative to moving averages
        features['price_to_sma_5'] = df['close'] / features['sma_5']
        features['price_to_sma_20'] = df['close'] / features['sma_20']
        features['price_to_sma_50'] = df['close'] / features['sma_50']
        features['price_to_sma_200'] = df['close'] / features['sma_200']
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        features['bb_middle_20'] = features['sma_20']
        features['bb_std_20'] = df['close'].rolling(window=20).std()
        features['bb_upper_20'] = features['bb_middle_20'] + (features['bb_std_20'] * 2)
        features['bb_lower_20'] = features['bb_middle_20'] - (features['bb_std_20'] * 2)
        features['bb_width_20'] = (features['bb_upper_20'] - features['bb_lower_20']) / features['bb_middle_20']
        features['bb_position_20'] = (df['close'] - features['bb_lower_20']) / (features['bb_upper_20'] - features['bb_lower_20'])
        
        # Calculate MACD
        features['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        features['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(window=14).mean()
        
        # Calculate volume features
        features['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        features['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        features['volume_ratio_5_20'] = features['volume_sma_5'] / features['volume_sma_20']
        features['volume_to_sma_5'] = df['volume'] / features['volume_sma_5']
        features['volume_to_sma_20'] = df['volume'] / features['volume_sma_20']
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _prepare_labels(self, stock: Stock, timeframe: TimeFrame, forward_days: int = 5) -> pd.Series:
        """
        Prepare labels for the machine learning model.
        
        Args:
            stock: Stock object with price data
            timeframe: Timeframe for analysis
            forward_days: Number of days to look forward for return calculation
            
        Returns:
            pd.Series: Series with labels
        """
        # Get price data as DataFrame
        df = stock.get_dataframe(timeframe)
        
        if df.empty:
            logger.warning(f"No data available for {stock.symbol} on {timeframe.value} timeframe")
            return pd.Series()
        
        # Calculate forward returns
        forward_return = df['close'].pct_change(forward_days).shift(-forward_days)
        
        # Create labels (1 for positive return, 0 for negative return)
        labels = (forward_return > 0).astype(int)
        
        # Drop NaN values
        labels = labels.dropna()
        
        return labels
    
    def train(self, stocks: Dict[str, Stock], timeframe: TimeFrame) -> Dict:
        """
        Train the machine learning model on historical data.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
            timeframe: Timeframe for analysis
            
        Returns:
            Dict: Dictionary of training results
        """
        logger.info("Starting model training")
        
        # Prepare features and labels
        all_features = []
        all_labels = []
        
        for symbol, stock in stocks.items():
            logger.info(f"Preparing data for {symbol}")
            
            # Prepare features
            features = self._prepare_features(stock, timeframe)
            if features.empty:
                continue
            
            # Prepare labels
            labels = self._prepare_labels(stock, timeframe)
            if labels.empty:
                continue
            
            # Align features and labels
            common_index = features.index.intersection(labels.index)
            if len(common_index) == 0:
                logger.warning(f"No common index between features and labels for {symbol}")
                continue
            
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            
            # Add to combined data
            all_features.append(features)
            all_labels.append(labels)
        
        if not all_features or not all_labels:
            logger.error("No data available for training")
            return {"success": False, "error": "No data available for training"}
        
        # Combine data from all stocks
        X = pd.concat(all_features)
        y = pd.concat(all_labels)
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importances = dict(zip(self.feature_names, importances))
            sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            top_features = [f for f, i in sorted_importances if i >= self.feature_importance_threshold]
        else:
            top_features = []
        
        logger.info(f"Model training completed with accuracy: {accuracy:.4f}")
        
        return {
            "success": True,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "top_features": top_features
        }
    
    def predict(self, stock: Stock, timeframe: TimeFrame) -> pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Args:
            stock: Stock object with price data
            timeframe: Timeframe for analysis
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        if self.model is None:
            logger.error("Model not trained")
            return pd.DataFrame()
        
        # Prepare features
        features = self._prepare_features(stock, timeframe)
        if features.empty:
            logger.warning(f"No features available for {stock.symbol}")
            return pd.DataFrame()
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        probabilities = self.model.predict_proba(features_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame(index=features.index)
        results['probability'] = probabilities[:, 1]  # Probability of positive return
        results['prediction'] = (results['probability'] > self.prediction_threshold).astype(int)
        results['signal'] = results['prediction'].diff()  # 1 for new buy signal, -1 for new sell signal
        
        return results
    
    def evaluate_strategy(self, stock: Stock, timeframe: TimeFrame, initial_balance: float = 10000.0) -> Dict:
        """
        Evaluate a trading strategy based on model predictions.
        
        Args:
            stock: Stock object with price data
            timeframe: Timeframe for analysis
            initial_balance: Initial account balance
            
        Returns:
            Dict: Dictionary of strategy performance metrics
        """
        if self.model is None:
            logger.error("Model not trained")
            return {"success": False, "error": "Model not trained"}
        
        # Get predictions
        predictions = self.predict(stock, timeframe)
        if predictions.empty:
            logger.warning(f"No predictions available for {stock.symbol}")
            return {"success": False, "error": "No predictions available"}
        
        # Get price data
        df = stock.get_dataframe(timeframe)
        
        # Align price data with predictions
        common_index = df.index.intersection(predictions.index)
        if len(common_index) == 0:
            logger.warning(f"No common index between price data and predictions for {stock.symbol}")
            return {"success": False, "error": "No common index between price data and predictions"}
        
        df = df.loc[common_index]
        predictions = predictions.loc[common_index]
        
        # Simulate trading
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []
        
        for i in range(1, len(predictions)):
            date = predictions.index[i]
            signal = predictions['signal'].iloc[i]
            close = df['close'].iloc[i]
            
            # Buy signal
            if signal == 1 and position == 0:
                position = balance / close
                entry_price = close
                balance = 0
                trades.append({
                    'entry_date': date,
                    'entry_price': close,
                    'direction': 'long',
                    'position_size': position
                })
            
            # Sell signal
            elif (signal == -1 or i == len(predictions) - 1) and position > 0:
                balance = position * close
                profit_loss = (close - entry_price) * position
                profit_loss_percent = (close / entry_price - 1) * 100
                
                trades[-1].update({
                    'exit_date': date,
                    'exit_price': close,
                    'profit_loss': profit_loss,
                    'profit_loss_percent': profit_loss_percent
                })
                
                position = 0
                entry_price = 0
        
        # Calculate performance metrics
        if not trades:
            logger.warning(f"No trades generated for {stock.symbol}")
            return {"success": False, "error": "No trades generated"}
        
        # Filter completed trades
        completed_trades = [t for t in trades if 'exit_date' in t]
        
        if not completed_trades:
            logger.warning(f"No completed trades for {stock.symbol}")
            return {"success": False, "error": "No completed trades"}
        
        # Calculate metrics
        total_trades = len(completed_trades)
        winning_trades = sum(1 for t in completed_trades if t['profit_loss'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t['profit_loss'] for t in completed_trades if t['profit_loss'] > 0)
        total_loss = sum(abs(t['profit_loss']) for t in completed_trades if t['profit_loss'] < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_profit = total_profit / winning_trades if winning_trades > 0 else 0
        average_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        final_balance = balance if position == 0 else position * df['close'].iloc[-1]
        total_return = final_balance - initial_balance
        total_return_percent = (final_balance / initial_balance - 1) * 100
        
        logger.info(f"Strategy evaluation completed with return: {total_return_percent:.2f}%")
        
        return {
            "success": True,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "total_return": total_return,
            "total_return_percent": total_return_percent,
            "final_balance": final_balance,
            "trades": completed_trades
        }
