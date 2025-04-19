# Swing Trading Algorithm

A Python-based system for identifying and simulating potentially profitable swing trades in the stock market over periods typically lasting from a few hours to several days.

## Overview

This project aims to create a comprehensive trading system that:
- Analyzes market data across multiple timeframes
- Identifies potential trading opportunities based on technical indicators
- Generates entry and exit signals according to predefined rules
- Manages risk through calculated position sizing and stop-loss placement
- Tracks and analyzes performance to enable continuous improvement

## Project Structure

```
swing-trading-algo/
├── src/                    # Source code for swing trading
│   ├── data/               # Data retrieval and storage
│   ├── analysis/           # Technical analysis and indicators
│   ├── strategy/           # Trading strategy implementation
│   ├── risk/               # Risk management and position sizing
│   ├── simulation/         # Trade simulation engine
│   ├── performance/        # Performance tracking and analysis
│   ├── optimization/       # Strategy optimization and machine learning
│   └── utils/              # Utility functions and helpers
├── day-trading-algo/       # Day trading algorithm
│   ├── src/                # Source code for day trading
│   │   ├── analysis/       # Technical analysis for day trading
│   │   ├── data/           # Data handling for day trading
│   │   ├── ml/             # Machine learning models
│   │   ├── optimization/   # Strategy optimization
│   │   ├── performance/    # Performance tracking
│   │   ├── risk/           # Risk management
│   │   ├── simulation/     # Trade simulation
│   │   ├── strategy/       # Trading strategies
│   │   ├── ui/             # User interface components
│   │   └── utils/          # Utility functions
│   ├── config/             # Configuration files
│   ├── data/               # Market data storage
│   ├── logs/               # Log files
│   ├── models/             # Trained ML models
│   └── reports/            # Performance reports
├── ui/                     # Enhanced UI components
├── scripts/                # PowerShell and batch scripts
├── tests/                  # Test suite
├── docs/                   # Documentation
└── README.md               # Project overview
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Unix/MacOS
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Configuration
1. Copy `config/config_template.yaml` to `config/config.yaml`
2. Edit `config.yaml` to set your preferences and API keys

### Running the System

#### Enhanced UI (Recommended)
```
trade-ui
```
This command will run the enhanced UI from anywhere in your PowerShell terminal.

Options:
- `-Test`: Run in test mode with simulated trades
- `-5m`: Use 5-minute timeframe (default is 1-minute)

#### Basic Command Line
```
powershell -ExecutionPolicy Bypass -File "scripts\trade.ps1"
```

## Features

### Swing Trading Algorithm
- Multi-timeframe analysis (Daily, 4-hour, 1-hour)
- Technical indicators (Moving Averages, RSI, ATR)
- Customizable trading strategies
- Risk-based position sizing
- Detailed performance tracking
- Strategy optimization

### Day Trading Algorithm
- Multi-timeframe analysis (1-minute, 5-minute, 15-minute, 1-hour, 4-hour)
- Real-time market data processing
- Machine learning-based trade predictions
- Adaptive learning from past trades
- Risk management with 10% allocation per trade
- Enhanced UI with market status information
- Paper trading mode for testing
- Detailed trade reporting and visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.
