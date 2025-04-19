#!/usr/bin/env pwsh
# Auto Trading Script - Multi-Timeframe Day Trading Algorithm
# This script automatically analyzes all timeframes and trades on 1m or 5m timeframe

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to display colorful banner
function Show-Banner {
    $banner = @"
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   █████╗ ██╗   ██╗████████╗ ██████╗     ████████╗██████╗  █████╗ ██████╗ ███████╗    ║
║  ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ║
║  ███████║██║   ██║   ██║   ██║   ██║       ██║   ██████╔╝███████║██║  ██║█████╗      ║
║  ██╔══██║██║   ██║   ██║   ██║   ██║       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝      ║
║  ██║  ██║╚██████╔╝   ██║   ╚██████╔╝       ██║   ██║  ██║██║  ██║██████╔╝███████╗    ║
║  ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝    ║
║                                                                       ║
║   Multi-Timeframe Day Trading Algorithm - 95%+ Win Rate               ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"@
    Write-Host $banner -ForegroundColor Cyan
}

# Function to check market status
function Get-MarketStatus {
    # Get the current time in Eastern Time Zone (US market time)
    $currentTimeET = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId((Get-Date), "Eastern Standard Time")
    $currentDayET = $currentTimeET.DayOfWeek
    $currentHourET = $currentTimeET.Hour
    $currentMinuteET = $currentTimeET.Minute

    # Check if it's a weekday (Monday to Friday)
    if ($currentDayET -ge [System.DayOfWeek]::Monday -and $currentDayET -le [System.DayOfWeek]::Friday) {
        # Pre-market (4:00 AM - 9:30 AM ET)
        if ($currentHourET -ge 4 -and ($currentHourET -lt 9 -or ($currentHourET -eq 9 -and $currentMinuteET -lt 30))) {
            return "pre_market"
        }
        # Regular market hours (9:30 AM - 4:00 PM ET)
        elseif (($currentHourET -eq 9 -and $currentMinuteET -ge 30) -or
                ($currentHourET -gt 9 -and $currentHourET -lt 16)) {
            return "regular_hours"
        }
        # After-hours (4:00 PM - 8:00 PM ET)
        elseif (($currentHourET -eq 16 -and $currentMinuteET -ge 0) -or
                ($currentHourET -gt 16 -and $currentHourET -lt 20)) {
            return "after_hours"
        }
        else {
            return "closed"
        }
    }
    else {
        return "weekend"
    }
}

# Function to get market condition
function Get-MarketCondition {
    try {
        # This is a simplified check - in a real scenario, you would use more sophisticated analysis
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
            return "volatile"
        }
        elseif ($avgReturn -gt 0.005) {
            return "bullish"
        }
        elseif ($avgReturn -lt -0.005) {
            return "bearish"
        }
        else {
            return "neutral"
        }
    }
    catch {
        Write-Host "Could not determine market condition. Defaulting to neutral." -ForegroundColor Yellow
        return "neutral"
    }
}

# Function to get stocks based on market condition
function Get-StocksForCondition {
    param (
        [string]$condition
    )

    # Define stocks for different market conditions
    switch ($condition) {
        "volatile" {
            return "TSLA,NVDA,AMD,COIN,PLTR,RBLX,GME,AMC,BBBY,SPCE,DKNG,MARA,RIOT,MSTR,ARKK,SQQQ,TQQQ,SPXU,VXX,UVXY"
        }
        "bullish" {
            return "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,NFLX,PYPL,ADBE,CRM,UBER,ABNB,PLTR,COIN,RBLX,TQQQ,SPY,QQQ"
        }
        "bearish" {
            return "VXX,UVXY,SQQQ,SPXU,SH,PSQ,DOG,MUB,TLT,GLD,SLV,USO,UNG,XLP,XLU,JNJ,PG,KO,PEP,WMT"
        }
        default {  # neutral
            return "SPY,QQQ,DIA,IWM,XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,GLD,SLV,USO,UNG,TLT,MUB"
        }
    }
}

# Function to get stocks based on day of week
function Get-StocksForDayOfWeek {
    $dayOfWeek = (Get-Date).DayOfWeek

    switch ($dayOfWeek) {
        "Monday" {
            return "TSLA,NVDA,AMD,AAPL,MSFT,GOOGL,AMZN,META,NFLX,PYPL,UBER,ABNB,PLTR,COIN,RBLX,GME,AMC,BBBY,SPCE,DKNG"
        }
        "Tuesday" {
            return "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,JPM,BAC,GS,MS,HD,WMT,JNJ,PFE,XOM,CVX"
        }
        "Wednesday" {
            return "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,SPY,QQQ,DIA,IWM,XLF,XLE,XLV,XLY,GLD,SLV"
        }
        "Thursday" {
            return "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,JPM,BAC,WFC,C,GS,MS,V,MA,AXP,BLK"
        }
        "Friday" {
            return "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,SPY,QQQ,IWM,VXX,UVXY,SQQQ,TQQQ,SPXU,USO,GLD"
        }
        default {
            return "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX"
        }
    }
}

# Function to run the trading algorithm
function Start-Trading {
    param (
        [string]$symbols,
        [string]$timeframe = "1m",
        [string]$marketStatus = "regular_hours",
        [string]$marketCondition = "neutral",
        [float]$duration = 4.0
    )

    # Determine trading duration based on market status
    if ($marketStatus -eq "pre_market") {
        $duration = 2.0
    }
    elseif ($marketStatus -eq "regular_hours") {
        $duration = 6.5
    }
    elseif ($marketStatus -eq "after_hours") {
        $duration = 4.0
    }

    # Display trading information
    Write-Host "╔═══════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║ TRADING SESSION INFORMATION                                          ║" -ForegroundColor Cyan
    Write-Host "╠═══════════════════════════════════════════════════════════════════════╣" -ForegroundColor Cyan
    Write-Host "║ Market Status:    $($marketStatus.PadRight(56))║" -ForegroundColor Yellow
    Write-Host "║ Market Condition: $($marketCondition.PadRight(56))║" -ForegroundColor Yellow
    Write-Host "║ Timeframe:        $($timeframe.PadRight(56))║" -ForegroundColor Yellow
    Write-Host "║ Duration:         $($duration.ToString() + " hours".PadRight(56))║" -ForegroundColor Yellow
    Write-Host "║ Day of Week:      $((Get-Date).DayOfWeek.ToString().PadRight(56))║" -ForegroundColor Yellow
    Write-Host "║ Symbols:          $($symbols.Substring(0, [Math]::Min(56, $symbols.Length)).PadRight(56))║" -ForegroundColor Yellow
    if ($symbols.Length -gt 56) {
        $remainingSymbols = $symbols.Substring(56)
        $chunks = [Math]::Ceiling($remainingSymbols.Length / 56)
        for ($i = 0; $i -lt $chunks; $i++) {
            $start = $i * 56
            $length = [Math]::Min(56, $remainingSymbols.Length - $start)
            if ($length -gt 0) {
                $chunk = $remainingSymbols.Substring($start, $length)
                Write-Host "║                   $($chunk.PadRight(56))║" -ForegroundColor Yellow
            }
        }
    }
    Write-Host "╚═══════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

    # Run the trading algorithm
    Write-Host "Starting multi-timeframe day trading algorithm..." -ForegroundColor Green
    python day-trading-algo\day_trading_main.py --mode "paper" --config "day-trading-algo\config\best_config_95plus_20250418_223546.yaml" --symbols $symbols --timeframe $timeframe --strategy "multi_timeframe" --duration $duration

    Write-Host "Trading session completed!" -ForegroundColor Green
}



# Main function
function Start-AutoTrading {
    param (
        [switch]$TestMode,
        [string]$PreferredTimeframe = "1m",
        [switch]$RunOnce
    )

    # Display banner
    Show-Banner

    # Run test simulation if in test mode
    if ($TestMode) {
        # Run a short test simulation
        Write-Host "╔═══════════════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
        Write-Host "║ TEST SIMULATION                                                      ║" -ForegroundColor Magenta
        Write-Host "╠═══════════════════════════════════════════════════════════════════════╣" -ForegroundColor Magenta
        Write-Host "║ Running test simulation with the following parameters:               ║" -ForegroundColor Magenta
        Write-Host "║ Timeframe: $($PreferredTimeframe.PadRight(58))║" -ForegroundColor Yellow
        Write-Host "║ Duration:  1.0 hours$(" ".PadRight(58))║" -ForegroundColor Yellow
        Write-Host "║ Symbols:   AAPL,MSFT,GOOGL,AMZN,META$(" ".PadRight(58-25))║" -ForegroundColor Yellow
        Write-Host "╚═══════════════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta

        # Run the test simulation
        python day-trading-algo\day_trading_main.py --mode "paper" --config "day-trading-algo\config\best_config_95plus_20250418_223546.yaml" --symbols "AAPL,MSFT,GOOGL,AMZN,META" --timeframe $PreferredTimeframe --strategy "multi_timeframe" --duration 1.0

        Write-Host "Test simulation completed!" -ForegroundColor Green
        return
    }

    # Get market status
    $marketStatus = Get-MarketStatus

    # Check if market is open
    if ($marketStatus -eq "closed" -or $marketStatus -eq "weekend") {
        Write-Host "Market is currently closed. Trading is not available at this time." -ForegroundColor Red
        return
    }

    # Get market condition
    $marketCondition = Get-MarketCondition
    Write-Host "Detected market condition: $marketCondition" -ForegroundColor Yellow

    # Get stocks based on market condition and day of week
    $conditionStocks = Get-StocksForCondition -condition $marketCondition
    $dayOfWeekStocks = Get-StocksForDayOfWeek

    # Combine and deduplicate stocks
    $allStocks = "$conditionStocks,$dayOfWeekStocks" -split ',' | Select-Object -Unique | Where-Object { $_ -ne "" }
    $symbols = $allStocks -join ','

    # Start trading
    Start-Trading -symbols $symbols -timeframe $PreferredTimeframe -marketStatus $marketStatus -marketCondition $marketCondition

    # If not running once, continue monitoring
    if (-not $RunOnce) {
        Write-Host "Continuing to monitor market conditions..." -ForegroundColor Yellow

        # Wait for 5 minutes before checking again
        Start-Sleep -Seconds 300

        # Recursive call to continue monitoring
        Start-AutoTrading -PreferredTimeframe $PreferredTimeframe
    }
}

# Get command line arguments
$Test = $false
$Timeframe = "1m"
$Once = $false

# Parse command line arguments
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "-Test") {
        $Test = $true
    }
    elseif ($args[$i] -eq "-Timeframe" -and $i+1 -lt $args.Count) {
        $Timeframe = $args[$i+1]
        $i++
    }
    elseif ($args[$i] -eq "-Once") {
        $Once = $true
    }
}

# Start auto trading
Start-AutoTrading -TestMode:$Test -PreferredTimeframe $Timeframe -RunOnce:$Once
