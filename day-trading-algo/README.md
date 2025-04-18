# Ultimate Day Trading Algorithm

A sophisticated day trading algorithm designed to make intraday trading decisions using technical indicators, market regime detection, and adaptive machine learning.

## Features

- **200+ Trading Symbols**: Configured with over 200 different stocks from various sectors and market caps
- **Intraday Momentum Strategy**: Uses MACD, RSI, Bollinger Bands, and volume patterns to identify potential entry and exit points
- **Market Regime Detection**: Adapts trading parameters based on the current market conditions
- **Real-time Paper Trading**: Tests the strategy with real-time market data
- **Adaptive Learning**: Learns from mistakes and successes to improve over time
- **Extensive Backtesting**: Tests the strategy over historical data with detailed performance metrics
- **Risk Management**: Implements sophisticated risk management with position sizing, stop losses, and take profits

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/day-trading-algo.git
cd day-trading-algo
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

### Configuration

The algorithm is configured using the `config/config.yaml` file. You can modify this file to change:

- Trading symbols
- Strategy parameters
- Risk management settings
- Backtesting parameters
- Paper trading settings
- Adaptive learning parameters

## Usage

### Running Backtests

To run a backtest over a specific date range:

```
python run_day_trading.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

You can also specify specific symbols to test:

```
python run_day_trading.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31 --symbols AAPL,MSFT,GOOGL
```

### Running Paper Trading

To run paper trading with real-time data:

```
python run_day_trading.py --mode paper
```

## Adaptive Learning

The algorithm includes an adaptive learning system that:

1. **Learns from Mistakes**: Identifies patterns in losing trades and adjusts strategy parameters
2. **Learns from Successes**: Reinforces patterns in winning trades
3. **Provides Feedback**: Generates detailed feedback on trading mistakes
4. **Adapts Parameters**: Automatically adjusts strategy parameters based on performance

## Directory Structure

- `config/`: Configuration files
- `src/`: Source code
  - `data/`: Data handling modules
  - `strategy/`: Trading strategy implementations
  - `risk/`: Risk management modules
  - `simulation/`: Backtesting and paper trading modules
  - `ml/`: Machine learning and adaptive learning modules
  - `utils/`: Utility functions
- `logs/`: Log files
- `reports/`: Performance reports and backtest results
- `models/`: Saved machine learning models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Yahoo Finance for providing free market data
- Various open-source Python libraries that made this project possible
