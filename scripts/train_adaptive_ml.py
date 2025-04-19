#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train and evaluate an adaptive machine learning model for trading.

This script trains an adaptive machine learning model on historical data and
evaluates its performance for trading.
"""

import os
import sys
import yaml
import logging
import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.data.models import TimeFrame
from src.data.provider import create_data_provider
from src.optimization.adaptive_ml import AdaptiveMLModel
from src.utils.logger import setup_logging


def load_config(config_path="config/config.yaml"):
    """Load configuration from config file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def create_output_dirs(config):
    """Create output directories for logs, data, and reports."""
    # Create logs directory
    logs_dir = Path("logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create data cache directory
    if "cache_dir" in config["data"]:
        cache_dir = Path(config["data"]["cache_dir"])
        os.makedirs(cache_dir, exist_ok=True)
    
    # Create reports directory
    reports_dir = Path("reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create models directory
    models_dir = Path("models")
    os.makedirs(models_dir, exist_ok=True)
    
    return {
        "logs_dir": logs_dir,
        "reports_dir": reports_dir,
        "models_dir": models_dir
    }


def plot_feature_importance(feature_importance, output_path, top_n=20):
    """Plot feature importance and save to file."""
    plt.figure(figsize=(12, 8))
    
    # Sort feature importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Take top N features
    top_features = sorted_features[:top_n]
    
    # Extract feature names and importance values
    feature_names = [f[0] for f in top_features]
    importance_values = [f[1] for f in top_features]
    
    # Create horizontal bar plot
    plt.barh(range(len(feature_names)), importance_values, align='center')
    plt.yticks(range(len(feature_names)), feature_names)
    
    # Add labels and title
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save to file
    plt.savefig(output_path)
    plt.close()


def plot_predictions(predictions, stock_data, output_path):
    """Plot predictions and stock price and save to file."""
    plt.figure(figsize=(12, 8))
    
    # Create subplot for stock price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(stock_data.index, stock_data['close'], label='Close Price')
    ax1.set_ylabel('Price')
    ax1.set_title('Stock Price and Predictions')
    ax1.legend()
    
    # Create subplot for predictions
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(predictions.index, predictions['probability'], label='Probability')
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    
    # Add buy/sell signals
    buy_signals = predictions[predictions['signal'] == 1]
    sell_signals = predictions[predictions['signal'] == -1]
    
    ax2.scatter(buy_signals.index, buy_signals['probability'], color='green', marker='^', s=100, label='Buy Signal')
    ax2.scatter(sell_signals.index, sell_signals['probability'], color='red', marker='v', s=100, label='Sell Signal')
    
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Date')
    ax2.legend()
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Save to file
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Main function to train and evaluate the adaptive ML model."""
    parser = argparse.ArgumentParser(description="Train and evaluate an adaptive machine learning model for trading")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model-type", choices=["random_forest", "gradient_boosting"],
                        help="Type of machine learning model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the model on test data")
    parser.add_argument("--symbol", help="Symbol to evaluate the model on")
    parser.add_argument("--history-days", type=int, default=365,
                        help="Number of days of historical data to use")
    parser.add_argument("--simulate-trades", action="store_true",
                        help="Simulate trades with adaptive learning")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    output_dirs = create_output_dirs(config)
    
    # Set up logging
    logger = setup_logging(config["general"], output_dirs["logs_dir"])
    logger.info("Starting Adaptive ML Model Training")
    
    # Initialize ML configuration
    if "adaptive_ml" not in config:
        config["adaptive_ml"] = {
            "model_type": "random_forest",
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "prediction_threshold": 0.6,
            "feature_importance_threshold": 0.01,
            "learning_rate": 0.1,
            "online_learning": True,
            "model_path": "models",
            "model_name": "adaptive_ml_model",
            "retrain_interval": 10,
            "memory_window": 500
        }
    
    # Override model type if specified
    if args.model_type:
        config["adaptive_ml"]["model_type"] = args.model_type
    
    # Override history days if specified
    if args.history_days:
        config["data"]["history_days"] = args.history_days
    
    # Initialize data provider
    try:
        data_provider = create_data_provider(config["data"])
        logger.info(f"Data provider initialized: {data_provider.__class__.__name__}")
    except Exception as e:
        logger.error(f"Error initializing data provider: {e}")
        return
    
    # Retrieve historical data for symbols
    symbols = config["data"]["symbols"]
    timeframe = TimeFrame.from_string(config["data"]["timeframes"][0])  # Use first timeframe
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config["data"]["history_days"])
    
    stocks = {}
    for symbol in symbols:
        try:
            logger.info(f"Retrieving data for {symbol}")
            stock = data_provider.get_historical_data(symbol, timeframe, start_date, end_date)
            stocks[symbol] = stock
            logger.info(f"Retrieved {len(stock.data.get(timeframe, []))} data points for {symbol}")
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
    
    if not stocks:
        logger.error("No data retrieved for any symbols. Exiting.")
        return
    
    # Initialize adaptive ML model
    ml_model = AdaptiveMLModel(config["adaptive_ml"])
    
    # Train model
    logger.info(f"Training {config['adaptive_ml']['model_type']} model")
    result = ml_model.train(stocks, timeframe)
    
    if not result["success"]:
        logger.error(f"Error training model: {result.get('error', 'Unknown error')}")
        return
    
    # Print training results
    logger.info(f"Model training completed")
    logger.info(f"Accuracy: {result['accuracy']:.4f}")
    logger.info(f"Precision: {result['precision']:.4f}")
    logger.info(f"Recall: {result['recall']:.4f}")
    logger.info(f"F1 Score: {result['f1']:.4f}")
    logger.info(f"Top features: {', '.join(result['top_features'][:10])}")
    
    # Save training results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dirs["reports_dir"] / f"adaptive_ml_training_result_{timestamp}.json"
    
    with open(result_file, 'w') as file:
        json.dump(result, file, indent=4)
    
    logger.info(f"Training results saved to {result_file}")
    
    # Plot feature importance
    feature_importance = ml_model.get_feature_importance()
    if feature_importance:
        feature_importance_file = output_dirs["reports_dir"] / f"feature_importance_{timestamp}.png"
        plot_feature_importance(feature_importance, feature_importance_file)
        logger.info(f"Feature importance plot saved to {feature_importance_file}")
    
    # Evaluate model on specific symbol if requested
    if args.evaluate:
        symbol_to_evaluate = args.symbol if args.symbol else symbols[0]
        
        if symbol_to_evaluate not in stocks:
            logger.error(f"Symbol {symbol_to_evaluate} not found in data")
            return
        
        logger.info(f"Evaluating model on {symbol_to_evaluate}")
        
        # Make predictions
        predictions = ml_model.predict(stocks[symbol_to_evaluate], timeframe)
        
        if predictions.empty:
            logger.error(f"No predictions generated for {symbol_to_evaluate}")
            return
        
        # Plot predictions
        stock_data = stocks[symbol_to_evaluate].get_dataframe(timeframe)
        predictions_file = output_dirs["reports_dir"] / f"predictions_{symbol_to_evaluate}_{timestamp}.png"
        plot_predictions(predictions, stock_data, predictions_file)
        logger.info(f"Predictions plot saved to {predictions_file}")
        
        # Save predictions to CSV
        predictions_csv = output_dirs["reports_dir"] / f"predictions_{symbol_to_evaluate}_{timestamp}.csv"
        predictions.to_csv(predictions_csv)
        logger.info(f"Predictions saved to {predictions_csv}")
    
    # Simulate trades with adaptive learning if requested
    if args.simulate_trades:
        symbol_to_simulate = args.symbol if args.symbol else symbols[0]
        
        if symbol_to_simulate not in stocks:
            logger.error(f"Symbol {symbol_to_simulate} not found in data")
            return
        
        logger.info(f"Simulating trades with adaptive learning for {symbol_to_simulate}")
        
        # Get stock data
        stock_data = stocks[symbol_to_simulate].get_dataframe(timeframe)
        
        # Initialize simulation variables
        balance = config["general"]["initial_capital"]
        position = 0
        entry_price = 0
        trades = []
        
        # Split data into training and testing
        train_size = int(len(stock_data) * 0.7)
        train_data = stock_data.iloc[:train_size]
        test_data = stock_data.iloc[train_size:]
        
        # Create a temporary stock object with training data
        train_stock = stocks[symbol_to_simulate].copy()
        train_stock.data[timeframe] = [p for p in train_stock.data[timeframe] if p.timestamp <= train_data.index[-1]]
        
        # Train model on training data
        logger.info("Training model on training data")
        ml_model.train({symbol_to_simulate: train_stock}, timeframe)
        
        # Simulate trades on test data
        logger.info("Simulating trades on test data")
        
        for i in range(len(test_data)):
            current_date = test_data.index[i]
            
            # Create a temporary stock object with data up to current date
            temp_stock = stocks[symbol_to_simulate].copy()
            temp_stock.data[timeframe] = [p for p in temp_stock.data[timeframe] if p.timestamp <= current_date]
            
            # Make prediction
            prediction = ml_model.predict(temp_stock, timeframe, current_date)
            
            if prediction.empty or current_date not in prediction.index:
                continue
            
            signal = prediction.loc[current_date, 'signal']
            close = test_data.iloc[i]['close']
            
            # Buy signal
            if signal == 1 and position == 0:
                position = balance / close
                entry_price = close
                balance = 0
                
                logger.info(f"Buy signal at {current_date}: {close}")
                
                trades.append({
                    'entry_date': current_date,
                    'entry_price': close,
                    'direction': 'long',
                    'position_size': position
                })
            
            # Sell signal
            elif (signal == -1 or i == len(test_data) - 1) and position > 0:
                balance = position * close
                profit_loss = (close - entry_price) * position
                profit_loss_percent = (close / entry_price - 1) * 100
                
                logger.info(f"Sell signal at {current_date}: {close}, Profit/Loss: {profit_loss:.2f} ({profit_loss_percent:.2f}%)")
                
                trades[-1].update({
                    'exit_date': current_date,
                    'exit_price': close,
                    'profit_loss': profit_loss,
                    'profit_loss_percent': profit_loss_percent
                })
                
                # Update model with trade outcome
                outcome = 1 if profit_loss > 0 else 0
                features = ml_model._prepare_features(temp_stock, timeframe).iloc[-1]
                ml_model.update_model(features, outcome)
                
                position = 0
                entry_price = 0
        
        # Calculate performance metrics
        if not trades:
            logger.warning(f"No trades generated for {symbol_to_simulate}")
            return
        
        # Filter completed trades
        completed_trades = [t for t in trades if 'exit_date' in t]
        
        if not completed_trades:
            logger.warning(f"No completed trades for {symbol_to_simulate}")
            return
        
        # Calculate metrics
        total_trades = len(completed_trades)
        winning_trades = sum(1 for t in completed_trades if t['profit_loss'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t['profit_loss'] for t in completed_trades if t['profit_loss'] > 0)
        total_loss = sum(abs(t['profit_loss']) for t in completed_trades if t['profit_loss'] < 0)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        final_balance = balance
        total_return = final_balance - config["general"]["initial_capital"]
        total_return_percent = (final_balance / config["general"]["initial_capital"] - 1) * 100
        
        # Print simulation results
        logger.info(f"Simulation completed")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Winning trades: {winning_trades} ({win_rate:.2%})")
        logger.info(f"Losing trades: {losing_trades}")
        logger.info(f"Profit factor: {profit_factor:.2f}")
        logger.info(f"Final balance: {final_balance:.2f}")
        logger.info(f"Total return: {total_return:.2f} ({total_return_percent:.2f}%)")
        
        # Save simulation results
        simulation_result = {
            "symbol": symbol_to_simulate,
            "initial_balance": config["general"]["initial_capital"],
            "final_balance": final_balance,
            "total_return": total_return,
            "total_return_percent": total_return_percent,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "trades": completed_trades
        }
        
        simulation_file = output_dirs["reports_dir"] / f"adaptive_ml_simulation_{symbol_to_simulate}_{timestamp}.json"
        
        with open(simulation_file, 'w') as file:
            json.dump(simulation_result, file, indent=4, default=str)
        
        logger.info(f"Simulation results saved to {simulation_file}")


if __name__ == "__main__":
    main()
