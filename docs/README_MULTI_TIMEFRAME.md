# Multi-Timeframe Day Trading Strategy

This document explains how to use the new multi-timeframe day trading strategy that has been added to the day trading algorithm.

## Overview

The multi-timeframe strategy analyzes price action across multiple timeframes to identify high-probability trading setups. By considering multiple timeframes, the strategy can filter out noise and identify stronger trading signals.

The strategy uses the following timeframes:
- 1-hour (1h) - For overall trend direction
- 15-minute (15m) - For intermediate trend confirmation
- 5-minute (5m) - For entry timing
- 1-minute (1m) - For precise entry and exit points

## Benefits of Multi-Timeframe Analysis

1. **Better Trend Identification**: Higher timeframes help identify the primary trend direction
2. **Reduced False Signals**: Confirmation across multiple timeframes reduces false signals
3. **Better Entry and Exit Points**: Lower timeframes provide more precise entry and exit points
4. **Improved Risk Management**: Multiple timeframe analysis helps set better stop-loss and take-profit levels

## How to Use

### Command Line Usage

To use the multi-timeframe strategy, run the day trading algorithm with the `--strategy` parameter set to `multi_timeframe`:

```
python day_trading_main.py --strategy multi_timeframe --timeframe 1m
```

You can also specify the timeframe to use as the primary timeframe. For day trading, the 1-minute (1m) or 5-minute (5m) timeframes are recommended:

```
python day_trading_main.py --strategy multi_timeframe --timeframe 1m
```

### Using with Auto Trader

To use the multi-timeframe strategy with the auto trader script:

```
python auto_trader.py --strategy multi_timeframe --timeframe 1m --start-now
```

## Strategy Logic

The multi-timeframe strategy works as follows:

1. **Analyze Higher Timeframes**: First, it analyzes higher timeframes (1h, 15m) to determine the overall trend direction.

2. **Confirm with Lower Timeframes**: Then, it looks for confirmation on lower timeframes (5m, 1m) to identify potential entry points.

3. **Entry Conditions**:
   - For long entries: Higher timeframes must show an uptrend, and lower timeframes must show a bullish signal (RSI oversold bounce, MA crossover, etc.)
   - For short entries: Higher timeframes must show a downtrend, and lower timeframes must show a bearish signal (RSI overbought drop, MA crossover, etc.)

4. **Exit Conditions**:
   - Take profit at predefined levels
   - Stop loss at predefined levels
   - Trailing stop loss to lock in profits

## Configuration

The multi-timeframe strategy uses the same configuration parameters as the intraday strategy. You can adjust these parameters in the configuration file:

```yaml
strategy:
  name: "multi_timeframe"
  ma_type: "EMA"
  fast_ma_period: 8
  slow_ma_period: 21
  trend_ma_period: 50
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  volume_ma_period: 10
  session_start: "09:30"
  session_end: "16:00"
  avoid_first_minutes: 15
  avoid_last_minutes: 15
```

## Performance Considerations

The multi-timeframe strategy may be more computationally intensive than the single-timeframe strategy, as it needs to analyze data across multiple timeframes. However, the improved signal quality often outweighs the additional computational cost.

For optimal performance, ensure that your system has sufficient memory and processing power, especially when trading a large number of symbols simultaneously.
