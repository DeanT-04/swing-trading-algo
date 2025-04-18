#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train and evaluate a machine learning model for trading.

This script trains a machine learning model on historical data and
evaluates its performance for trading.
"""

import os
import sys
import yaml
import logging
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.data.models import TimeFrame
from src.data.provider import create_data_provider
from src.optimization.ml_model import MLModel
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


def main():
    """Main function to train and evaluate the ML model."""
    parser = argparse.ArgumentParser(description="Train and evaluate a machine learning model for trading")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model-type", choices=["random_forest", "gradient_boosting"],
                        help="Type of machine learning model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the model on test data")
    parser.add_argument("--symbol", help="Symbol to evaluate the model on")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    output_dirs = create_output_dirs(config)
    
    # Set up logging
    logger = setup_logging(config["general"], output_dirs["logs_dir"])
    logger.info("Starting ML Model Training")
    
    # Initialize ML configuration
    if "ml" not in config:
        config["ml"] = {
            "model_type": "random_forest",
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "prediction_threshold": 0.6,
            "feature_importance_threshold": 0.01
        }
    
    # Override model type if specified
    if args.model_type:
        config["ml"]["model_type"] = args.model_type
    
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
    
    # Initialize ML model
    ml_model = MLModel(config["ml"])
    
    # Train model
    logger.info(f"Training {config['ml']['model_type']} model")
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
    result_file = output_dirs["reports_dir"] / f"ml_training_result_{timestamp}.json"
    
    with open(result_file, 'w') as file:
        json.dump(result, file, indent=4)
    
    logger.info(f"Training results saved to {result_file}")
    
    # Evaluate model on specific symbol if requested
    if args.evaluate:
        symbol_to_evaluate = args.symbol if args.symbol else symbols[0]
        
        if symbol_to_evaluate not in stocks:
            logger.error(f"Symbol {symbol_to_evaluate} not found in data")
            return
        
        logger.info(f"Evaluating model on {symbol_to_evaluate}")
        eval_result = ml_model.evaluate_strategy(stocks[symbol_to_evaluate], timeframe, config["general"]["initial_capital"])
        
        if not eval_result["success"]:
            logger.error(f"Error evaluating model: {eval_result.get('error', 'Unknown error')}")
            return
        
        # Print evaluation results
        logger.info(f"Strategy evaluation completed")
        logger.info(f"Total trades: {eval_result['total_trades']}")
        logger.info(f"Win rate: {eval_result['win_rate']:.4f}")
        logger.info(f"Profit factor: {eval_result['profit_factor']:.4f}")
        logger.info(f"Total return: {eval_result['total_return_percent']:.2f}%")
        logger.info(f"Final balance: {eval_result['final_balance']:.2f}")
        
        # Save evaluation results
        eval_result_file = output_dirs["reports_dir"] / f"ml_evaluation_result_{symbol_to_evaluate}_{timestamp}.json"
        
        with open(eval_result_file, 'w') as file:
            json.dump(eval_result, file, indent=4)
        
        logger.info(f"Evaluation results saved to {eval_result_file}")


if __name__ == "__main__":
    main()
