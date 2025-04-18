# Configuration Guide

This guide explains how to configure the Swing Trading Algorithm.

## Configuration File

The algorithm is configured using a YAML file located at `config/config.yaml`. A template is provided at `config/config_template.yaml`.

To create your configuration file:

```bash
cp config/config_template.yaml config/config.yaml
```

Then edit `config/config.yaml` with your preferred text editor.

## Configuration Sections

The configuration file is divided into several sections:

### General

General settings for the algorithm:

```yaml
general:
  name: "Swing Trading Algorithm"
  version: "0.1.0"
  initial_capital: 50.0  # Starting capital in GBP
  currency: "GBP"
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

- `name`: Name of the algorithm (for display purposes)
- `version`: Version of the algorithm (for display purposes)
- `initial_capital`: Starting capital for the simulation
- `currency`: Currency for the simulation
- `log_level`: Logging level

### Data

Settings for data retrieval and storage:

```yaml
data:
  provider: "alpha_vantage"  # Options: alpha_vantage, yahoo_finance, csv
  api_key: "YOUR_API_KEY_HERE"
  symbols:
    - "AAPL"  # Apple
    - "MSFT"  # Microsoft
    - "AMZN"  # Amazon
    - "GOOGL" # Alphabet
    - "META"  # Meta Platforms
  timeframes:
    - "daily"
    - "4h"
    - "1h"
  cache_dir: "data/cache"
  history_days: 365  # Days of historical data to retrieve
```

- `provider`: Data provider to use
- `api_key`: API key for the data provider (if required)
- `symbols`: List of stock symbols to analyze
- `timeframes`: List of timeframes to analyze
- `cache_dir`: Directory to cache data
- `history_days`: Number of days of historical data to retrieve

### Analysis

Settings for technical analysis:

```yaml
analysis:
  indicators:
    moving_averages:
      - type: "SMA"  # Simple Moving Average
        periods: [20, 50, 200]
      - type: "EMA"  # Exponential Moving Average
        periods: [9, 21]
    rsi:
      period: 14
      overbought: 70
      oversold: 30
    atr:
      period: 14
```

- `indicators`: Technical indicators to calculate
  - `moving_averages`: Moving average settings
  - `rsi`: Relative Strength Index settings
  - `atr`: Average True Range settings

### Strategy

Settings for the trading strategy:

```yaml
strategy:
  name: "basic_swing"
  rules:
    trend_following: true
    counter_trend: false
    breakout: true
    pullback: true
  entry_confirmation:
    - "price_above_ma"
    - "rsi_oversold_bullish"
    - "volume_increase"
  exit_rules:
    - "stop_loss_hit"
    - "take_profit_hit"
    - "trend_reversal"
```

- `name`: Name of the strategy to use
- `rules`: Strategy-specific rules
- `entry_confirmation`: Conditions required for entry
- `exit_rules`: Conditions for exit

### Risk

Settings for risk management:

```yaml
risk:
  max_risk_per_trade: 0.02  # 2% of account per trade
  risk_reward_ratio: 2.0  # Minimum reward:risk ratio
  max_open_positions: 3
  stop_loss:
    method: "atr"  # Options: atr, percent, support_resistance
    atr_multiplier: 2.0
    percent: 0.05  # 5% from entry price (if method is percent)
```

- `max_risk_per_trade`: Maximum risk per trade as a percentage of account balance
- `risk_reward_ratio`: Minimum reward-to-risk ratio for trades
- `max_open_positions`: Maximum number of open positions
- `stop_loss`: Stop loss settings
  - `method`: Method to calculate stop loss
  - `atr_multiplier`: Multiplier for ATR-based stop loss
  - `percent`: Percentage for percentage-based stop loss

### Simulation

Settings for trade simulation:

```yaml
simulation:
  slippage: 0.001  # 0.1% slippage on entries and exits
  commission: 0.002  # 0.2% commission per trade
  enable_fractional_shares: true
```

- `slippage`: Slippage to apply to entries and exits
- `commission`: Commission to apply to trades
- `enable_fractional_shares`: Whether to allow fractional shares

### Performance

Settings for performance tracking:

```yaml
performance:
  metrics:
    - "win_rate"
    - "profit_factor"
    - "max_drawdown"
    - "sharpe_ratio"
  reporting:
    frequency: "daily"  # Options: daily, weekly, monthly
    save_to_file: true
    plot_equity_curve: true
```

- `metrics`: Performance metrics to calculate
- `reporting`: Reporting settings
  - `frequency`: Frequency of performance reports
  - `save_to_file`: Whether to save reports to file
  - `plot_equity_curve`: Whether to plot equity curve

### Optimization

Settings for strategy optimization:

```yaml
optimization:
  enable: false
  method: "grid_search"  # Options: grid_search, random_search, genetic
  parameters:
    - "moving_average_periods"
    - "rsi_levels"
    - "atr_multiplier"
  metric: "sharpe_ratio"  # Metric to optimize for
```

- `enable`: Whether to enable optimization
- `method`: Optimization method
- `parameters`: Parameters to optimize
- `metric`: Metric to optimize for

## Environment Variables

Some configuration options can be overridden using environment variables:

- `TRADING_ALGO_CONFIG`: Path to the configuration file
- `TRADING_ALGO_LOG_LEVEL`: Logging level
- `TRADING_ALGO_API_KEY`: API key for the data provider

Example:

```bash
export TRADING_ALGO_API_KEY="your-api-key"
python main.py
```

## Configuration Examples

### Basic Configuration

```yaml
general:
  name: "Swing Trading Algorithm"
  initial_capital: 50.0
  currency: "GBP"
  log_level: "INFO"

data:
  provider: "alpha_vantage"
  api_key: "YOUR_API_KEY_HERE"
  symbols:
    - "AAPL"
    - "MSFT"
  timeframes:
    - "daily"
  cache_dir: "data/cache"
  history_days: 365

strategy:
  name: "basic_swing"

risk:
  max_risk_per_trade: 0.02
  risk_reward_ratio: 2.0
  max_open_positions: 3

simulation:
  slippage: 0.001
  commission: 0.002
  enable_fractional_shares: true

performance:
  metrics:
    - "win_rate"
    - "profit_factor"
    - "max_drawdown"
  reporting:
    save_to_file: true
    plot_equity_curve: true
```

### Advanced Configuration

```yaml
general:
  name: "Advanced Swing Trading Algorithm"
  initial_capital: 1000.0
  currency: "GBP"
  log_level: "DEBUG"

data:
  provider: "alpha_vantage"
  api_key: "YOUR_API_KEY_HERE"
  symbols:
    - "AAPL"
    - "MSFT"
    - "AMZN"
    - "GOOGL"
    - "META"
  timeframes:
    - "daily"
    - "4h"
    - "1h"
  cache_dir: "data/cache"
  history_days: 365

analysis:
  indicators:
    moving_averages:
      - type: "SMA"
        periods: [20, 50, 200]
      - type: "EMA"
        periods: [9, 21]
    rsi:
      period: 14
      overbought: 70
      oversold: 30
    atr:
      period: 14

strategy:
  name: "basic_swing"
  ma_type: "EMA"
  fast_ma_period: 9
  slow_ma_period: 21
  trend_ma_period: 50
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  atr_period: 14
  atr_multiplier: 2.0
  volume_ma_period: 20
  risk_reward_ratio: 2.0

risk:
  max_risk_per_trade: 0.02
  risk_reward_ratio: 2.0
  max_open_positions: 3
  stop_loss:
    method: "atr"
    atr_multiplier: 2.0
    percent: 0.05

simulation:
  slippage: 0.001
  commission: 0.002
  enable_fractional_shares: true

performance:
  metrics:
    - "win_rate"
    - "profit_factor"
    - "max_drawdown"
    - "sharpe_ratio"
  reporting:
    frequency: "daily"
    save_to_file: true
    plot_equity_curve: true

optimization:
  enable: true
  method: "grid_search"
  parameters:
    - "fast_ma_period"
    - "slow_ma_period"
    - "rsi_oversold"
    - "rsi_overbought"
    - "atr_multiplier"
  metric: "sharpe_ratio"
```

## Validation

The algorithm validates the configuration file when it starts. If there are any errors, it will log them and exit.

Common validation errors:
- Missing required fields
- Invalid values
- Incompatible settings

## Best Practices

1. **Start Simple**: Begin with a simple configuration and gradually add complexity.
2. **Test Changes**: Test configuration changes on historical data before using them for live trading.
3. **Document Changes**: Keep track of configuration changes and their effects on performance.
4. **Use Version Control**: Store your configuration files in version control to track changes over time.
5. **Separate Environments**: Use different configuration files for development, testing, and production.
