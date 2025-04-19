#!/usr/bin/env pwsh
# Simple Trading Script - Multi-Timeframe Day Trading Algorithm
# This script can be run from anywhere and will find the trading algorithm

# Set error action preference
$ErrorActionPreference = "Stop"

# Define the path to the trading algorithm
$tradingAlgoPath = "C:\Users\deana\Documents\Coding projects\trading-algo\swing-trading-algo"

# Check if the trading algorithm exists
if (-not (Test-Path $tradingAlgoPath)) {
    Write-Host "Trading algorithm not found at $tradingAlgoPath" -ForegroundColor Red
    Write-Host "Please update the script with the correct path" -ForegroundColor Red
    exit 1
}

# Display banner
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "          Multi-Timeframe Day Trading Algorithm        " -ForegroundColor Cyan
Write-Host "                    95%+ Win Rate                      " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Parse command line arguments
$testMode = $false
$timeframe = "1m"
$runOnce = $false

foreach ($arg in $args) {
    if ($arg -eq "-Test") {
        $testMode = $true
    }
    elseif ($arg -eq "-Once") {
        $runOnce = $true
    }
    elseif ($arg -eq "-5m") {
        $timeframe = "5m"
    }
}

# Set the trading algorithm directory
$tradingAlgoDir = Join-Path $tradingAlgoPath "day-trading-algo"

# Run in test mode if specified
if ($testMode) {
    Write-Host "=======================================================" -ForegroundColor Magenta
    Write-Host "                    TEST SIMULATION                    " -ForegroundColor Magenta
    Write-Host "=======================================================" -ForegroundColor Magenta
    Write-Host "Running test simulation with the following parameters:" -ForegroundColor Yellow
    Write-Host "Timeframe: $timeframe" -ForegroundColor Yellow
    Write-Host "Duration:  1.0 hours" -ForegroundColor Yellow
    Write-Host "Symbols:   AAPL,MSFT,GOOGL,AMZN,META" -ForegroundColor Yellow
    Write-Host ""

    # Run the test simulation
    python "$tradingAlgoDir\day_trading_main.py" --mode "paper" --config "$tradingAlgoDir\config\best_config_95plus_20250418_223546.yaml" --symbols "AAPL,MSFT,GOOGL,AMZN,META" --timeframe $timeframe --strategy "multi_timeframe" --duration 1.0

    Write-Host "Test simulation completed!" -ForegroundColor Green
    exit
}

# Define stocks for different market conditions
$volatileStocks = "TSLA,NVDA,AMD,COIN,PLTR,RBLX,GME,AMC,BBBY,SPCE,DKNG,MARA,RIOT,MSTR,ARKK,SQQQ,TQQQ,SPXU,VXX,UVXY"
$bullishStocks = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,NFLX,PYPL,ADBE,CRM,UBER,ABNB,PLTR,COIN,RBLX,TQQQ,SPY,QQQ"
$bearishStocks = "VXX,UVXY,SQQQ,SPXU,SH,PSQ,DOG,MUB,TLT,GLD,SLV,USO,UNG,XLP,XLU,JNJ,PG,KO,PEP,WMT"
$neutralStocks = "SPY,QQQ,DIA,IWM,XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,GLD,SLV,USO,UNG,TLT,MUB"

# Get day of week
$dayOfWeek = (Get-Date).DayOfWeek

# Define stocks for different days of the week
$dayStocks = @{
    "Monday" = "TSLA,NVDA,AMD,AAPL,MSFT,GOOGL,AMZN,META,NFLX,PYPL,UBER,ABNB,PLTR,COIN,RBLX,GME,AMC,BBBY,SPCE,DKNG"
    "Tuesday" = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,JPM,BAC,GS,MS,HD,WMT,JNJ,PFE,XOM,CVX"
    "Wednesday" = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,SPY,QQQ,DIA,IWM,XLF,XLE,XLV,XLY,GLD,SLV"
    "Thursday" = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,JPM,BAC,WFC,C,GS,MS,V,MA,AXP,BLK"
    "Friday" = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,SPY,QQQ,IWM,VXX,UVXY,SQQQ,TQQQ,SPXU,USO,GLD"
}

# Determine market condition (simplified)
$marketCondition = "neutral"
try {
    # Get SPY data for the last few days
    $spyData = Invoke-RestMethod -Uri "https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1d&range=5d"

    # Get the closing prices
    $closePrices = $spyData.chart.result[0].indicators.quote[0].close

    # Calculate daily returns
    $returns = @()
    for ($i = 1; $i -lt $closePrices.Count; $i++) {
        $returns += ($closePrices[$i] - $closePrices[$i-1]) / $closePrices[$i-1]
    }

    # Calculate average return and volatility
    $avgReturn = ($returns | Measure-Object -Average).Average
    $volatility = ($returns | Measure-Object -StandardDeviation).StandardDeviation

    # Determine market condition
    if ($volatility -gt 0.015) {
        $marketCondition = "volatile"
    }
    elseif ($avgReturn -gt 0.005) {
        $marketCondition = "bullish"
    }
    elseif ($avgReturn -lt -0.005) {
        $marketCondition = "bearish"
    }
}
catch {
    Write-Host "Could not determine market condition. Defaulting to neutral." -ForegroundColor Yellow
}

# Get stocks based on market condition
$conditionStocks = switch ($marketCondition) {
    "volatile" { $volatileStocks }
    "bullish" { $bullishStocks }
    "bearish" { $bearishStocks }
    default { $neutralStocks }
}

# Get stocks based on day of week
$dayOfWeekStocks = $dayStocks[$dayOfWeek.ToString()]

# Combine and deduplicate stocks
$allStocksArray = "$conditionStocks,$dayOfWeekStocks" -split ','
$uniqueStocks = $allStocksArray | Select-Object -Unique | Where-Object { $_ -ne "" }
$symbols = $uniqueStocks -join ','

# Display trading information
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "                 TRADING INFORMATION                   " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "Market Condition: $marketCondition" -ForegroundColor Yellow
Write-Host "Day of Week:      $dayOfWeek" -ForegroundColor Yellow
Write-Host "Timeframe:        $timeframe" -ForegroundColor Yellow
Write-Host "Number of Stocks: $($uniqueStocks.Count)" -ForegroundColor Yellow
Write-Host ""

# Run the trading algorithm
Write-Host "Starting multi-timeframe day trading algorithm..." -ForegroundColor Green
python "$tradingAlgoDir\day_trading_main.py" --mode "paper" --config "$tradingAlgoDir\config\best_config_95plus_20250418_223546.yaml" --symbols $symbols --timeframe $timeframe --strategy "multi_timeframe" --duration 6.5

Write-Host "Trading session completed!" -ForegroundColor Green
