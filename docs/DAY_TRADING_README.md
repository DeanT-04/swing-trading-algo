# Multi-Timeframe Day Trading Algorithm

A sophisticated day trading algorithm that analyzes multiple timeframes to achieve a 95%+ win rate. This algorithm uses machine learning and technical analysis to identify high-probability trading opportunities across various market conditions.

## Features

- **Multi-Timeframe Analysis**: Analyzes 1h, 15m, 5m, and 1m timeframes to identify the strongest trading opportunities
- **Adaptive Machine Learning**: Continuously learns from market conditions and past trades to improve performance
- **Risk Management**: Implements strict risk management with position sizing limited to 10% of available balance
- **Market Condition Detection**: Automatically adapts to volatile, bullish, bearish, or neutral market conditions
- **Day-of-Week Optimization**: Uses different stock selections optimized for each day of the week
- **Real-Time Monitoring**: Provides a clean, informative UI showing account status, active trades, and performance metrics
- **Detailed Reporting**: Generates comprehensive trade reports and performance analytics

## Installation

### Quick Installation

Run the installation script to set up everything automatically:

```powershell
.\install.ps1
```

This will:
1. Create a virtual environment and install dependencies
2. Set up the necessary directories
3. Create a global PowerShell command
4. Add a desktop shortcut
5. Configure everything for immediate use

### Manual Installation

If you prefer to install manually:

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```
   mkdir -p logs data/cache reports models
   ```

## Usage

### Running the Algorithm

You can run the day trading algorithm in several ways:

1. **Using the PowerShell command** (after installation):
   ```
   trade
   ```

2. **Using the batch file**:
   ```
   .\trade.bat
   ```

3. **Using the desktop shortcut** created during installation

4. **Directly with Python**:
   ```
   python day-trading-algo\day_trading_main.py --mode paper --timeframe 1m --strategy multi_timeframe
   ```

### Command Options

The following options are available:

- `-Test`: Run in test mode with predefined stocks (AAPL, MSFT, GOOGL, AMZN, META)
- `-5m`: Use 5-minute timeframe (default is 1m)
- `-Once`: Run once and exit (don't continue monitoring)

Examples:
```
trade -Test
trade -5m
trade -Test -5m
```

## How It Works

1. **Market Analysis**:
   - Analyzes SPY (S&P 500 ETF) to determine current market conditions
   - Classifies the market as volatile, bullish, bearish, or neutral
   - Selects appropriate stocks based on market conditions and day of week

2. **Multi-Timeframe Analysis**:
   - Analyzes higher timeframes (1h, 15m) to identify the primary trend
   - Uses lower timeframes (5m, 1m) for precise entry and exit points
   - Confirms signals across multiple timeframes to reduce false signals

3. **Machine Learning Integration**:
   - Uses a gradient boosting model to predict trade outcomes
   - Continuously learns from new market data and completed trades
   - Requires a minimum 70% confidence score to enter trades

4. **Risk Management**:
   - Limits position size to 10% of available balance
   - Uses ATR (Average True Range) for dynamic stop-loss placement
   - Implements a minimum 1.5:1 reward-to-risk ratio

5. **Trade Execution**:
   - Executes trades in paper trading mode
   - Tracks all trades with detailed entry/exit information
   - Generates performance reports and analytics

## Performance Metrics

The algorithm is designed to achieve:
- 95%+ win rate in various market conditions
- Consistent daily profits with £50 starting capital
- Growth to £100/day as capital increases
- Ability to trade 50-200 different stocks

## Project Structure

```
swing-trading-algo/
├── day-trading-algo/           # Day trading algorithm
│   ├── config/                 # Configuration files
│   ├── src/                    # Source code
│   │   ├── data/               # Data retrieval and storage
│   │   ├── analysis/           # Technical analysis and indicators
│   │   ├── strategy/           # Trading strategy implementation
│   │   ├── risk/               # Risk management and position sizing
│   │   ├── simulation/         # Trade simulation engine
│   │   ├── performance/        # Performance tracking and analysis
│   │   ├── optimization/       # Strategy optimization and machine learning
│   │   └── utils/              # Utility functions and helpers
│   ├── day_trading_main.py     # Main entry point
│   └── README.md               # Documentation
├── install.ps1                 # Installation script
├── trade.bat                   # Quick launch batch file
├── trade-global.ps1            # Global trading script
└── README.md                   # This file
```

## Requirements

- Python 3.8 or higher
- PowerShell 5.1 or higher
- Internet connection for real-time data
- Windows operating system

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended to be used for actual trading. Always consult with a licensed financial advisor before making investment decisions. The creators of this software are not responsible for any financial losses incurred from using this software.
