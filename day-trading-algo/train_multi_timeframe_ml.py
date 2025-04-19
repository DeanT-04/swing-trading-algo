#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the multi-timeframe machine learning model.

This script trains a machine learning model on historical data across multiple timeframes
and evaluates its performance.
"""

import os
import sys
import yaml
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.logging_utils import setup_logging
from src.data.models import Stock, TimeFrame
from src.data.provider import YahooFinanceProvider
from src.optimization.multi_timeframe_ml import MultiTimeframeMLModel


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)


def create_output_dirs(config: dict) -> dict:
    """
    Create output directories for models, logs, and reports.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Dictionary of output directories
    """
    from pathlib import Path

    # Create output directories
    model_path = Path(config.get("adaptive_ml", {}).get("model_path", "models"))
    logs_dir = Path("logs")
    reports_dir = Path("reports")

    model_path.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    return {
        "model_path": model_path,
        "logs_dir": logs_dir,
        "reports_dir": reports_dir
    }


def get_historical_data(config: dict, symbols: List[str], timeframes: List[TimeFrame],
                        start_date: datetime, end_date: datetime) -> Dict[str, Stock]:
    """
    Get historical data for multiple symbols and timeframes.

    Args:
        config: Configuration dictionary
        symbols: List of symbols to get data for
        timeframes: List of timeframes to get data for
        start_date: Start date for historical data
        end_date: End date for historical data

    Returns:
        Dict[str, Stock]: Dictionary of Stock objects by symbol
    """
    # Initialize data provider
    data_provider = YahooFinanceProvider(config)

    # Convert datetime objects to strings for the data provider
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Get historical data for each symbol and timeframe
    stocks = {}
    for symbol in symbols:
        try:
            # Get stock data for all timeframes
            stock = data_provider.get_stock_data(
                symbol=symbol,
                timeframes=timeframes,
                start_date=start_date_str,
                end_date=end_date_str
            )

            if stock and stock.data:
                stocks[symbol] = stock
                for timeframe in timeframes:
                    if timeframe in stock.data:
                        logging.info(f"Got {len(stock.data[timeframe])} data points for {symbol} on {timeframe.value} timeframe")
                    else:
                        logging.warning(f"No data available for {symbol} on {timeframe.value} timeframe")
            else:
                logging.warning(f"No data available for {symbol}")

        except Exception as e:
            logging.error(f"Error getting data for {symbol}: {e}")

    return stocks


def simulate_trades(ml_model: MultiTimeframeMLModel, stocks: Dict[str, Stock],
                    symbol_to_simulate: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    """
    Simulate trades using the trained model.

    Args:
        ml_model: Trained ML model
        stocks: Dictionary of Stock objects by symbol
        symbol_to_simulate: Symbol to simulate trades for
        start_date: Start date for simulation
        end_date: End date for simulation

    Returns:
        List[Dict]: List of simulated trades
    """
    if symbol_to_simulate not in stocks:
        logging.error(f"Symbol {symbol_to_simulate} not found in stocks")
        return []

    stock = stocks[symbol_to_simulate]

    # Get 1-minute data for simulation
    timeframe = TimeFrame.MINUTE_1
    if timeframe not in stock.data:
        logging.error(f"No {timeframe.value} data available for {symbol_to_simulate}")
        return []

    # Filter data to simulation period
    data = [p for p in stock.data[timeframe] if start_date <= p.timestamp <= end_date]
    if not data:
        logging.error(f"No data available for {symbol_to_simulate} in simulation period")
        return []

    # Initialize simulation variables
    position = 0
    entry_price = 0
    trades = []
    balance = 1000.0  # Initial balance

    # Simulate trades
    for i in range(len(data)):
        current_date = data[i].timestamp

        # Create a temporary stock object with data up to current date
        temp_stock = Stock(symbol=symbol_to_simulate)
        for tf in stock.data:
            temp_data = [p for p in stock.data[tf] if p.timestamp <= current_date]
            if temp_data:
                temp_stock.add_data(tf, temp_data)

        # Make prediction
        prediction = ml_model.predict(temp_stock, current_date)

        if prediction.empty or current_date not in prediction.index:
            continue

        # Get prediction for current date
        signal = prediction.loc[current_date, 'signal']
        close = data[i].close

        # Buy signal
        if signal == 1 and position == 0:
            position = balance / close
            entry_price = close

            trades.append({
                'type': 'buy',
                'entry_date': current_date,
                'entry_price': close,
                'position_size': position,
                'balance': balance
            })

            logging.info(f"Buy signal at {current_date}: {close}")

        # Sell signal
        elif (signal == -1 or i == len(data) - 1) and position > 0:
            balance = position * close
            profit_loss = (close - entry_price) * position
            profit_loss_percent = (close / entry_price - 1) * 100

            logging.info(f"Sell signal at {current_date}: {close}, Profit/Loss: {profit_loss:.2f} ({profit_loss_percent:.2f}%)")

            trades[-1].update({
                'exit_date': current_date,
                'exit_price': close,
                'profit_loss': profit_loss,
                'profit_loss_percent': profit_loss_percent
            })

            # Update model with trade outcome
            outcome = 1 if profit_loss > 0 else 0
            features = ml_model._prepare_features(temp_stock)
            if not features.empty and current_date in features.index:
                ml_model.update_model(features.loc[current_date], outcome)

            position = 0
            entry_price = 0

    return trades


def main():
    """Main function to train and evaluate the multi-timeframe ML model."""
    parser = argparse.ArgumentParser(description="Train and evaluate a multi-timeframe machine learning model for trading")
    parser.add_argument("--config", default="config/best_config_95plus_20250418_223546.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model-type", choices=["random_forest", "gradient_boosting"],
                        help="Type of machine learning model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the model on test data")
    parser.add_argument("--symbol", help="Symbol to evaluate the model on")
    parser.add_argument("--history-days", type=int, default=365,
                        help="Number of days of historical data to use")
    parser.add_argument("--simulate-trades", action="store_true",
                        help="Simulate trades using the trained model")
    parser.add_argument("--start-date", help="Start date for simulation (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for simulation (YYYY-MM-DD)")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create output directories
    output_dirs = create_output_dirs(config)

    # Set up logging
    log_level = config.get("general", {}).get("log_level", "INFO")
    log_file = os.path.join(output_dirs["logs_dir"], f"multi_timeframe_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_level, log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting Multi-Timeframe ML Model Training")

    # Initialize ML configuration
    if "adaptive_ml" not in config:
        config["adaptive_ml"] = {
            "model_type": "gradient_boosting",
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "prediction_threshold": 0.6,
            "feature_importance_threshold": 0.01,
            "learning_rate": 0.1,
            "online_learning": True,
            "model_path": "models",
            "model_name": "multi_timeframe_ml_model",
            "retrain_interval": 5,
            "memory_window": 1000
        }

    # Override model type if specified
    if args.model_type:
        config["adaptive_ml"]["model_type"] = args.model_type

    # Set model name to include timeframes
    config["adaptive_ml"]["model_name"] = "multi_timeframe_ml_model"

    # Get symbols from config
    symbols = config["data"]["symbols"]
    if args.symbol:
        symbols = [args.symbol]

    # Get timeframes
    timeframes = [
        TimeFrame.HOUR_1,
        TimeFrame.MINUTE_15,
        TimeFrame.MINUTE_5,
        TimeFrame.MINUTE_1
    ]

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.history_days)

    # Get historical data
    logger.info(f"Getting historical data for {len(symbols)} symbols from {start_date} to {end_date}")
    stocks = get_historical_data(config, symbols, timeframes, start_date, end_date)

    if not stocks:
        logger.error("No data available for training")
        return

    logger.info(f"Got data for {len(stocks)} symbols")

    # Initialize multi-timeframe ML model
    ml_model = MultiTimeframeMLModel(config["adaptive_ml"])

    # Train model
    logger.info(f"Training {config['adaptive_ml']['model_type']} model on multiple timeframes")
    result = ml_model.train(stocks)

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
    result_file = os.path.join(output_dirs["reports_dir"], f"multi_timeframe_ml_training_result_{timestamp}.json")

    with open(result_file, 'w') as file:
        json.dump(result, file, indent=4)

    logger.info(f"Training results saved to {result_file}")

    # Get feature importance
    feature_importance = ml_model.get_feature_importance()
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    logger.info("Feature importance:")
    for feature, importance in sorted_importance[:20]:
        logger.info(f"{feature}: {importance:.4f}")

    # Simulate trades if requested
    if args.simulate_trades:
        if args.symbol:
            symbol_to_simulate = args.symbol
        else:
            # Use the first symbol with data
            symbol_to_simulate = list(stocks.keys())[0]

        logger.info(f"Simulating trades for {symbol_to_simulate}")

        # Set simulation date range
        sim_start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else start_date
        sim_end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else end_date

        # Simulate trades
        trades = simulate_trades(ml_model, stocks, symbol_to_simulate, sim_start_date, sim_end_date)

        if not trades:
            logger.error("No trades generated in simulation")
            return

        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('profit_loss', 0) > 0)
        losing_trades = sum(1 for t in trades if t.get('profit_loss', 0) <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_profit = sum(t.get('profit_loss', 0) for t in trades)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        # Print performance metrics
        logger.info(f"Simulation results for {symbol_to_simulate}:")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Winning trades: {winning_trades} ({win_rate:.2%})")
        logger.info(f"Losing trades: {losing_trades} ({1-win_rate:.2%})")
        logger.info(f"Total profit: {total_profit:.2f}")
        logger.info(f"Average profit per trade: {avg_profit:.2f}")

        # Save simulation results
        sim_result_file = os.path.join(output_dirs["reports_dir"], f"multi_timeframe_ml_simulation_{symbol_to_simulate}_{timestamp}.json")

        with open(sim_result_file, 'w') as file:
            json.dump({
                "symbol": symbol_to_simulate,
                "start_date": sim_start_date.isoformat(),
                "end_date": sim_end_date.isoformat(),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "avg_profit": avg_profit,
                "trades": trades
            }, file, indent=4, default=str)

        logger.info(f"Simulation results saved to {sim_result_file}")


if __name__ == "__main__":
    main()
