# Multi-Timeframe Machine Learning Model Training

This document explains how to train and use the multi-timeframe machine learning model for day trading.

## Overview

The multi-timeframe ML model analyzes data across multiple timeframes (1h, 15m, 5m, 1m) to make more accurate trading predictions. By combining features from different timeframes, the model can identify patterns that would be missed when looking at a single timeframe.

## Training the Model

### Prerequisites

- Python 3.8 or higher
- Required Python packages (install with `pip install -r requirements.txt`)
- Historical data for the symbols you want to trade

### Training Process

1. **Run the Training Script**

   You can train the model using the provided PowerShell script:

   ```
   .\train_all_timeframes.ps1
   ```

   Or run the Python script directly:

   ```
   python train_multi_timeframe_ml.py --config "config/best_config_95plus_20250418_223546.yaml" --history-days 365 --model-type "gradient_boosting"
   ```

2. **Command-Line Arguments**

   The training script accepts the following arguments:

   - `--config`: Path to the configuration file
   - `--model-type`: Type of machine learning model to use (`random_forest` or `gradient_boosting`)
   - `--history-days`: Number of days of historical data to use for training
   - `--symbol`: Symbol to evaluate the model on (optional)
   - `--simulate-trades`: Simulate trades using the trained model
   - `--start-date`: Start date for simulation (YYYY-MM-DD)
   - `--end-date`: End date for simulation (YYYY-MM-DD)

3. **Training Output**

   The training process will output:
   
   - Training metrics (accuracy, precision, recall, F1 score)
   - Feature importance
   - Simulation results (if `--simulate-trades` is specified)
   
   All results are saved to the `reports` directory.

## Using the Trained Model

Once trained, the model is automatically integrated with the multi-timeframe strategy. To use it:

1. **Run the Day Trading Algorithm with ML**

   ```
   python day_trading_main.py --timeframe 1m --strategy multi_timeframe
   ```

2. **Run the Auto Trader with ML**

   ```
   python auto_trader.py --timeframe 1m --strategy multi_timeframe --start-now
   ```

## Continuous Learning

The model is designed to continuously learn from new trading data. After each trade, the model updates its knowledge based on the outcome (profit or loss). This allows the model to adapt to changing market conditions.

## Model Configuration

You can configure the ML model by editing the `adaptive_ml` section in the configuration file:

```yaml
adaptive_ml:
  model_type: "gradient_boosting"  # or "random_forest"
  test_size: 0.2
  random_state: 42
  n_estimators: 200
  max_depth: 8
  min_samples_split: 2
  min_samples_leaf: 1
  prediction_threshold: 0.6
  feature_importance_threshold: 0.01
  learning_rate: 0.1
  online_learning: true
  model_path: "models"
  model_name: "multi_timeframe_ml_model"
  retrain_interval: 5
  memory_window: 1000
```

## Performance Considerations

Training on multiple timeframes requires more computational resources than single-timeframe training. For optimal performance:

1. Use a machine with sufficient RAM (8GB+)
2. Limit the number of symbols during initial training
3. Start with a smaller history period (e.g., 90 days) and increase as needed

## Troubleshooting

If you encounter issues during training:

1. Check the logs in the `logs` directory
2. Ensure you have sufficient historical data for all timeframes
3. Try reducing the number of symbols or the history period
4. Check that all required Python packages are installed

## Advanced Usage

### Backtesting

To backtest the model on historical data:

```
python train_multi_timeframe_ml.py --config "config/best_config_95plus_20250418_223546.yaml" --simulate-trades --symbol "AAPL" --start-date "2023-01-01" --end-date "2023-12-31"
```

### Feature Importance Analysis

To analyze which features are most important for the model:

```
python train_multi_timeframe_ml.py --config "config/best_config_95plus_20250418_223546.yaml" --history-days 365
```

The feature importance will be printed to the console and saved in the training report.
