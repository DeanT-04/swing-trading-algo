#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive machine learning module.

This module provides functionality for adaptive machine learning models
that improve themselves after each trading iteration.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from pathlib import Path

from src.data.models import Stock, TimeFrame, TradeDirection

logger = logging.getLogger(__name__)


class AdaptiveMLModel:
    """
    Adaptive machine learning model that improves itself after each trading iteration.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the adaptive machine learning model with configuration parameters.
        
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
        self.learning_rate = config.get("learning_rate", 0.1)
        self.online_learning = config.get("online_learning", True)
        self.model_path = config.get("model_path", "models")
        self.model_name = config.get("model_name", "adaptive_ml_model")
        self.retrain_interval = config.get("retrain_interval", 10)  # Retrain after this many trades
        self.memory_window = config.get("memory_window", 500)  # Number of samples to keep in memory
        
        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Initialize training data storage
        self.X_train_buffer = []
        self.y_train_buffer = []
        self.trade_counter = 0
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load existing model if available
        self._load_model()
        
        logger.info(f"Initialized AdaptiveMLModel with model type: {self.model_type}")
    
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
                random_state=self.random_state,
                warm_start=self.online_learning  # Enable warm start for incremental learning
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                warm_start=self.online_learning  # Enable warm start for incremental learning
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
        
        # Calculate Stochastic Oscillator
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        features['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
        
        # Calculate ADX
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff(-1).abs()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        features['adx'] = dx.rolling(window=14).mean()
        
        # Calculate On-Balance Volume (OBV)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_sma_20'] = obv.rolling(window=20).mean()
        
        # Add day of week (0 = Monday, 6 = Sunday)
        if isinstance(df.index[0], pd.Timestamp):
            features['day_of_week'] = df.index.dayofweek
        
        # Add month
        if isinstance(df.index[0], pd.Timestamp):
            features['month'] = df.index.month
        
        # Add volatility features
        features['volatility_5d'] = df['close'].pct_change().rolling(window=5).std()
        features['volatility_10d'] = df['close'].pct_change().rolling(window=10).std()
        features['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
        
        # Add momentum features
        features['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        features['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
        
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
        
        # Add to training buffer
        self.X_train_buffer.extend(X.values.tolist())
        self.y_train_buffer.extend(y.values.tolist())
        
        # Limit buffer size to memory window
        if len(self.X_train_buffer) > self.memory_window:
            self.X_train_buffer = self.X_train_buffer[-self.memory_window:]
            self.y_train_buffer = self.y_train_buffer[-self.memory_window:]
        
        # Convert buffer to numpy arrays
        X_train = np.array(self.X_train_buffer)
        y_train = np.array(self.y_train_buffer)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train model
        if self.model is None:
            self.model = self._create_model()
        
        self.model.fit(X_train_scaled, y_train)
        
        # Save model
        self._save_model()
        
        # Evaluate model on training data
        y_pred = self.model.predict(X_train_scaled)
        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred, zero_division=0)
        recall = recall_score(y_train, y_pred, zero_division=0)
        f1 = f1_score(y_train, y_pred, zero_division=0)
        
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
            "top_features": top_features
        }
    
    def update_model(self, features: pd.DataFrame, actual_outcome: int) -> Dict:
        """
        Update the model with new data after a trade.
        
        Args:
            features: Features at the time of the trade
            actual_outcome: Actual outcome (1 for profit, 0 for loss)
            
        Returns:
            Dict: Dictionary of update results
        """
        if self.model is None:
            logger.error("Model not trained")
            return {"success": False, "error": "Model not trained"}
        
        # Increment trade counter
        self.trade_counter += 1
        
        # Add to training buffer
        self.X_train_buffer.append(features.values.tolist())
        self.y_train_buffer.append(actual_outcome)
        
        # Limit buffer size to memory window
        if len(self.X_train_buffer) > self.memory_window:
            self.X_train_buffer = self.X_train_buffer[-self.memory_window:]
            self.y_train_buffer = self.y_train_buffer[-self.memory_window:]
        
        # Check if it's time to retrain
        if self.trade_counter >= self.retrain_interval:
            # Convert buffer to numpy arrays
            X_train = np.array(self.X_train_buffer)
            y_train = np.array(self.y_train_buffer)
            
            # Scale features
            X_train_scaled = self.scaler.transform(X_train)
            
            # Retrain model
            self.model.fit(X_train_scaled, y_train)
            
            # Save model
            self._save_model()
            
            # Reset trade counter
            self.trade_counter = 0
            
            # Evaluate model on training data
            y_pred = self.model.predict(X_train_scaled)
            accuracy = accuracy_score(y_train, y_pred)
            
            logger.info(f"Model updated with accuracy: {accuracy:.4f}")
            
            return {
                "success": True,
                "updated": True,
                "accuracy": accuracy
            }
        
        return {
            "success": True,
            "updated": False
        }
    
    def predict(self, stock: Stock, timeframe: TimeFrame, current_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Args:
            stock: Stock object with price data
            timeframe: Timeframe for analysis
            current_date: Current date for prediction (if None, use latest data)
            
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
        
        # Filter features up to current_date if specified
        if current_date is not None:
            features = features[features.index <= current_date]
        
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model.
        
        Returns:
            Dict[str, float]: Dictionary of feature importance
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def _save_model(self):
        """Save the model to disk."""
        if self.model is None:
            return
        
        model_file = os.path.join(self.model_path, f"{self.model_name}.joblib")
        scaler_file = os.path.join(self.model_path, f"{self.model_name}_scaler.joblib")
        metadata_file = os.path.join(self.model_path, f"{self.model_name}_metadata.joblib")
        
        try:
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            
            metadata = {
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "last_updated": datetime.now().isoformat()
            }
            joblib.dump(metadata, metadata_file)
            
            logger.info(f"Model saved to {model_file}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load the model from disk."""
        model_file = os.path.join(self.model_path, f"{self.model_name}.joblib")
        scaler_file = os.path.join(self.model_path, f"{self.model_name}_scaler.joblib")
        metadata_file = os.path.join(self.model_path, f"{self.model_name}_metadata.joblib")
        
        if not os.path.exists(model_file):
            logger.info(f"No existing model found at {model_file}")
            return
        
        try:
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            
            metadata = joblib.load(metadata_file)
            self.feature_names = metadata["feature_names"]
            
            logger.info(f"Model loaded from {model_file}")
            logger.info(f"Last updated: {metadata['last_updated']}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Create a new model if loading fails
            self.model = self._create_model()
