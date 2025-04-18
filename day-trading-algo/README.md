# Day Trading Algorithm

An algorithmic day trading system that uses technical indicators, market regime detection, and adaptive machine learning to make intraday trading decisions.

## Features

- **Intraday Momentum Strategy**: Uses MACD, RSI, Bollinger Bands, and volume patterns to identify potential entry and exit points
- **Market Regime Detection**: Adapts trading parameters based on the current market conditions
- **Real-time Paper Trading**: Tests the strategy with real-time market data
- **Parameter Optimization**: Uses genetic algorithms to find optimal strategy parameters
- **Adaptive Machine Learning**: Continuously improves the strategy based on recent market data
- **Risk Management**: Implements position sizing, stop losses, and take profits
- **Performance Tracking**: Tracks and reports trading performance metrics

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (install with `pip install -r requirements.txt`)

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Configuration

The algorithm is configured using the `config/config.yaml` file. Key configuration sections include:

- **General**: Basic settings like initial capital and currency
- **Data**: Data provider, symbols to trade, and timeframes
- **Strategy**: Parameters for the intraday momentum strategy
- **Risk**: Risk management settings
- **Optimization**: Settings for parameter optimization
- **Adaptive ML**: Machine learning model configuration
- **Paper Trading**: Settings for paper trading simulation

## Usage

### Running Paper Trading

To run the algorithm in paper trading mode:

```
python run_day_trading.py --mode paper
```

### Running Backtests

To run a backtest:

```
python run_day_trading.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31 --timeframe 5m
```

### Running Optimization

To optimize strategy parameters:

```
python run_day_trading.py --mode optimize --method genetic --metric sharpe_ratio --generations 20 --population 50
```

## Strategy Details

The Intraday Momentum Strategy uses the following indicators and signals:

1. **Moving Averages**: Fast, slow, and trend EMAs to identify trend direction
2. **MACD**: For momentum and trend confirmation
3. **RSI**: To identify overbought and oversold conditions
4. **Bollinger Bands**: To identify price extremes and potential reversals
5. **Volume Analysis**: To confirm price movements with volume
6. **Market Regime Detection**: To adapt parameters based on market conditions

### Entry Signals

- **Long Entry**: Uptrend, MACD bullish crossover, RSI rising from oversold, price near lower Bollinger Band, volume confirmation
- **Short Entry**: Downtrend, MACD bearish crossover, RSI falling from overbought, price near upper Bollinger Band, volume confirmation

### Exit Signals

- **Take Profit**: When price reaches the take profit level
- **Stop Loss**: When price reaches the stop loss level
- **Trend Change**: When the trend changes direction
- **End of Day**: All positions are closed at the end of the trading day

## Performance Metrics

The algorithm tracks the following performance metrics:

- **Total Return**: Overall profit/loss
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Average time in a trade

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Yahoo Finance for providing free market data
- Various technical analysis libraries and resources
