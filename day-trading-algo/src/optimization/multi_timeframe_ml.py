#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-timeframe machine learning module for trading strategy improvement.

This module provides functionality to train and use machine learning models
across multiple timeframes to improve trading strategy performance.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Optional
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.data.models import Stock, TimeFrame, TradeDirection

logger = logging.getLogger(__name__)


class MultiTimeframeMLModel:
    """
    Machine learning model that analyzes multiple timeframes for trading decisions.
    
    This class implements a machine learning model that combines features from
    multiple timeframes to make more accurate trading predictions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the machine learning model with configuration parameters.
        
        Args:
            config: ML model configuration dictionary
        """
        self.config = config
        
        # Extract configuration parameters
        self.model_type = config.get("model_type", "gradient_boosting")
        self.test_size = config.get("test_size", 0.2)
        self.random_state = config.get("random_state", 42)
        self.n_estimators = config.get("n_estimators", 200)
        self.max_depth = config.get("max_depth", 8)
        self.min_samples_split = config.get("min_samples_split", 2)
        self.min_samples_leaf = config.get("min_samples_leaf", 1)
        self.prediction_threshold = config.get("prediction_threshold", 0.6)
        self.feature_importance_threshold = config.get("feature_importance_threshold", 0.01)
        self.learning_rate = config.get("learning_rate", 0.1)
        self.online_learning = config.get("online_learning", True)
        self.model_path = config.get("model_path", "models")
        self.model_name = config.get("model_name", "multi_timeframe_ml_model")
        self.retrain_interval = config.get("retrain_interval", 5)
        self.memory_window = config.get("memory_window", 1000)
        
        # Timeframes to analyze
        self.timeframes = [
            TimeFrame.HOUR_1,
            TimeFrame.MINUTE_15,
            TimeFrame.MINUTE_5,
            TimeFrame.MINUTE_1
        ]
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Initialize training buffers for online learning
        self.X_train_buffer = []
        self.y_train_buffer = []
        self.trade_counter = 0
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
        
        logger.info(f"Initialized {self.model_type} multi-timeframe ML model")
    
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
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _prepare_features_single_timeframe(self, stock: Stock, timeframe: TimeFrame) -> pd.DataFrame:
        """
        Prepare features for a single timeframe.
        
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
        features['return_1'] = df['close'].pct_change(1)
        features['return_3'] = df['close'].pct_change(3)
        features['return_5'] = df['close'].pct_change(5)
        
        # Calculate moving averages
        for period in [8, 13, 21, 50]:
            features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        features['rsi'] = 100 - (100 / (1 + rs))
        
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
        features['atr'] = tr.rolling(window=14).mean()
        
        # Calculate Bollinger Bands
        features['bb_middle'] = df['close'].rolling(window=20).mean()
        features['bb_std'] = df['close'].rolling(window=20).std()
        features['bb_upper'] = features['bb_middle'] + 2 * features['bb_std']
        features['bb_lower'] = features['bb_middle'] - 2 * features['bb_std']
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Calculate volume features
        features['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        features['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        features['volume_ratio'] = features['volume_sma_5'] / features['volume_sma_10']
        features['volume_change'] = df['volume'].pct_change(1)
        
        # Add time-based features if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour'] = df.index.hour
            features['minute'] = df.index.minute
            features['day_of_week'] = df.index.dayofweek
            
            # Market session features (assuming US market hours)
            features['is_market_open'] = ((df.index.hour >= 9) & (df.index.hour < 16) | 
                                         ((df.index.hour == 9) & (df.index.minute >= 30)) |
                                         ((df.index.hour == 16) & (df.index.minute == 0))) & \
                                         (df.index.dayofweek < 5)
            
            # Time to market close in minutes
            features['time_to_close'] = 0
            market_open_mask = features['is_market_open'] == 1
            features.loc[market_open_mask, 'time_to_close'] = (
                (16 - df.index[market_open_mask].hour) * 60 - df.index[market_open_mask].minute
            )
        
        # Add prefix to column names to identify timeframe
        features = features.add_prefix(f'{timeframe.value}_')
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _prepare_features(self, stock: Stock) -> pd.DataFrame:
        """
        Prepare features from multiple timeframes for the machine learning model.
        
        Args:
            stock: Stock object with price data
            
        Returns:
            pd.DataFrame: DataFrame with features from all timeframes
        """
        all_features = {}
        
        # Prepare features for each timeframe
        for timeframe in self.timeframes:
            features = self._prepare_features_single_timeframe(stock, timeframe)
            if not features.empty:
                all_features[timeframe] = features
        
        if not all_features:
            logger.warning(f"No features available for {stock.symbol} on any timeframe")
            return pd.DataFrame()
        
        # Use the lowest timeframe as the base for alignment
        base_timeframe = self.timeframes[-1]  # MINUTE_1
        if base_timeframe not in all_features:
            # If 1-minute data is not available, use the lowest available timeframe
            for tf in reversed(self.timeframes):
                if tf in all_features:
                    base_timeframe = tf
                    break
        
        base_features = all_features[base_timeframe]
        
        # Align and merge features from all timeframes
        merged_features = base_features.copy()
        
        for timeframe, features in all_features.items():
            if timeframe == base_timeframe:
                continue
            
            # Resample higher timeframe features to match the base timeframe
            for idx in merged_features.index:
                # Find the closest timestamp in the higher timeframe that is <= current timestamp
                closest_idx = features.index[features.index <= idx]
                if len(closest_idx) > 0:
                    closest_idx = closest_idx[-1]
                    # Add higher timeframe features to the merged features
                    for col in features.columns:
                        merged_features.loc[idx, col] = features.loc[closest_idx, col]
        
        # Drop rows with NaN values
        merged_features = merged_features.dropna()
        
        # Store feature names
        self.feature_names = merged_features.columns.tolist()
        
        return merged_features
    
    def _prepare_labels(self, stock: Stock, primary_timeframe: TimeFrame, forward_bars: int = 10) -> pd.Series:
        """
        Prepare labels for the machine learning model.
        
        Args:
            stock: Stock object with price data
            primary_timeframe: Primary timeframe for analysis
            forward_bars: Number of bars to look ahead for return calculation
            
        Returns:
            pd.Series: Series with labels (1 for positive return, 0 for negative return)
        """
        # Get price data as DataFrame
        df = stock.get_dataframe(primary_timeframe)
        
        if df.empty:
            logger.warning(f"No data available for {stock.symbol} on {primary_timeframe.value} timeframe")
            return pd.Series()
        
        # Calculate forward returns
        forward_return = df['close'].shift(-forward_bars) / df['close'] - 1
        
        # Create labels (1 for positive return, 0 for negative or zero return)
        labels = (forward_return > 0).astype(int)
        
        # Drop NaN values
        labels = labels.dropna()
        
        return labels
    
    def train(self, stocks: Dict[str, Stock]) -> Dict:
        """
        Train the machine learning model on historical data from multiple timeframes.
        
        Args:
            stocks: Dictionary of Stock objects by symbol
            
        Returns:
            Dict: Dictionary of training results
        """
        logger.info("Starting multi-timeframe model training")
        
        # Prepare features and labels
        all_features = []
        all_labels = []
        
        for symbol, stock in stocks.items():
            logger.info(f"Preparing data for {symbol}")
            
            # Prepare features from all timeframes
            features = self._prepare_features(stock)
            if features.empty:
                logger.warning(f"No features available for {symbol}")
                continue
            
            # Prepare labels using the primary timeframe (1-minute for day trading)
            primary_timeframe = TimeFrame.MINUTE_1
            labels = self._prepare_labels(stock, primary_timeframe)
            if labels.empty:
                logger.warning(f"No labels available for {symbol}")
                continue
            
            # Align features and labels
            common_index = features.index.intersection(labels.index)
            if len(common_index) == 0:
                logger.warning(f"No common timestamps between features and labels for {symbol}")
                continue
            
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            
            all_features.append(features)
            all_labels.append(labels)
        
        if not all_features:
            logger.error("No data available for training")
            return {"success": False, "error": "No data available for training"}
        
        # Combine data from all stocks
        X = pd.concat(all_features)
        y = pd.concat(all_labels)
        
        # Store feature names
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
        
        # Initialize training buffers for online learning
        if self.online_learning:
            self.X_train_buffer = X_train_scaled.tolist()[-self.memory_window:]
            self.y_train_buffer = y_train.tolist()[-self.memory_window:]
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importances = dict(zip(self.feature_names, importances))
            sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            top_features = [f for f, i in sorted_importances if i >= self.feature_importance_threshold]
        else:
            top_features = []
        
        # Save model
        self._save_model()
        
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
        
        if not self.online_learning:
            return {"success": True, "updated": False, "message": "Online learning disabled"}
        
        # Convert features to numpy array
        if isinstance(features, pd.DataFrame):
            features = features.values.reshape(1, -1)
        else:
            features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Add to training buffer
        self.X_train_buffer.append(features_scaled[0])
        self.y_train_buffer.append(actual_outcome)
        
        # Limit buffer size
        if len(self.X_train_buffer) > self.memory_window:
            self.X_train_buffer.pop(0)
            self.y_train_buffer.pop(0)
        
        # Increment trade counter
        self.trade_counter += 1
        
        # Check if it's time to retrain
        if self.trade_counter >= self.retrain_interval:
            # Convert buffer to numpy arrays
            X_train = np.array(self.X_train_buffer)
            y_train = np.array(self.y_train_buffer)
            
            # Retrain model
            self.model.fit(X_train, y_train)
            
            # Save model
            self._save_model()
            
            # Reset trade counter
            self.trade_counter = 0
            
            # Evaluate model on training data
            y_pred = self.model.predict(X_train)
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
    
    def predict(self, stock: Stock, current_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Make predictions using the trained model.
        
        Args:
            stock: Stock object with price data
            current_date: Current date for prediction (if None, use latest data)
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        if self.model is None:
            logger.error("Model not trained")
            return pd.DataFrame()
        
        # Prepare features from all timeframes
        features = self._prepare_features(stock)
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
        """Save the trained model to disk."""
        if self.model is None:
            return
        
        model_file = os.path.join(self.model_path, f"{self.model_name}.pkl")
        scaler_file = os.path.join(self.model_path, f"{self.model_name}_scaler.pkl")
        metadata_file = os.path.join(self.model_path, f"{self.model_name}_metadata.pkl")
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            metadata = {
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Model saved to {model_file}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load a trained model from disk."""
        model_file = os.path.join(self.model_path, f"{self.model_name}.pkl")
        scaler_file = os.path.join(self.model_path, f"{self.model_name}_scaler.pkl")
        metadata_file = os.path.join(self.model_path, f"{self.model_name}_metadata.pkl")
        
        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            logger.info("No existing model found, will train a new one")
            return
        
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self.feature_names = metadata.get("feature_names", [])
            
            logger.info(f"Loaded model from {model_file}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.scaler = StandardScaler()
