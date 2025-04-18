# Usage Guide

This guide explains how to use the Swing Trading Algorithm.

## Basic Usage

### Running the Algorithm

To run the Swing Trading Algorithm with the default configuration:

```bash
python main.py
```

This will:
1. Load the configuration from `config/config.yaml`
2. Retrieve historical data for the specified symbols
3. Analyze the data using the configured strategy
4. Generate trading signals
5. Simulate trades based on the signals
6. Generate a performance report

### Configuration

Before running the algorithm, you should configure it according to your preferences. See the [Configuration Guide](configuration.md) for details.

## Command-Line Options

The algorithm supports the following command-line options:

```bash
python main.py --config path/to/config.yaml  # Use a different configuration file
python main.py --verbose                     # Enable verbose logging
python main.py --backtest 2023-01-01 2023-12-31  # Run a backtest for a specific period
```

## Workflow Examples

### Basic Swing Trading Simulation

```bash
# Edit the configuration file
nano config/config.yaml

# Run the algorithm
python main.py

# View the performance report
cat reports/performance_report_YYYYMMDD_HHMMSS.md
```

### Backtesting a Strategy

```bash
# Edit the configuration file to set up your strategy
nano config/config.yaml

# Run a backtest for a specific period
python main.py --backtest 2022-01-01 2022-12-31

# View the performance report
cat reports/performance_report_YYYYMMDD_HHMMSS.md
```

### Optimizing Strategy Parameters

```bash
# Edit the configuration file to enable optimization
nano config/config.yaml

# Run the optimization
python main.py --optimize

# View the optimization results
cat reports/optimization_report_YYYYMMDD_HHMMSS.md
```

## Understanding the Output

### Console Output

The algorithm outputs information to the console, including:
- Configuration details
- Data retrieval progress
- Trading signals
- Trade executions
- Performance summary

Example:
```
INFO - Starting Swing Trading Algorithm
INFO - Configuration loaded: Swing Trading Algorithm
INFO - Data provider initialized: AlphaVantageProvider
INFO - Retrieving data for AAPL
INFO - Retrieved 252 data points for AAPL
INFO - Strategy initialized: Basic Swing Strategy
INFO - Simulator initialized with balance: 1000.0 GBP
INFO - Performance analyzer initialized
INFO - Starting simulation
INFO - Analyzing AAPL
INFO - Generated 5 signals for AAPL
INFO - Opened long trade for AAPL at 150.25
INFO - Closed 1 trades
INFO - Simulation completed
INFO - Final balance: 1052.36 GBP
INFO - Total profit/loss: 52.36 GBP (5.24%)
INFO - Total trades: 1
INFO - Open positions: 0
INFO - Swing Trading Algorithm completed
```

### Log Files

Detailed logs are saved to the `logs` directory. These logs include:
- Debug information
- Error messages
- Detailed trade information
- System events

### Performance Reports

Performance reports are saved to the `reports` directory in Markdown format. These reports include:
- Performance metrics (win rate, profit factor, etc.)
- Trade list with entry/exit details
- Equity curve chart (if enabled)

## Advanced Usage

### Custom Strategies

To implement a custom strategy:

1. Create a new Python file in the `src/strategy` directory
2. Implement your strategy class, inheriting from a base strategy or implementing the required methods
3. Update your configuration to use your custom strategy

### Data Sources

The algorithm supports multiple data sources:
- Alpha Vantage API
- Yahoo Finance API
- CSV files

To use a different data source, update the `provider` setting in the `data` section of your configuration file.

### Risk Management

You can customize risk management parameters in the `risk` section of your configuration file:
- Maximum risk per trade
- Risk-reward ratio
- Stop loss calculation method
- Position sizing

## Troubleshooting

### Common Issues

1. **No Signals Generated**: Check that your strategy parameters are appropriate for the market conditions and that you have sufficient historical data.

2. **Data Retrieval Errors**: Check your API key and internet connection. Consider using cached data if API access is limited.

3. **Performance Issues**: If the algorithm is running slowly, consider reducing the number of symbols or the amount of historical data.

### Getting Help

If you encounter any issues not covered here, please open an issue on the GitHub repository or contact the project maintainers.
