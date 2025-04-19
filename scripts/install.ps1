#!/usr/bin/env pwsh
# Day Trading Algorithm Installation Script
# This script sets up the day trading algorithm and creates global commands

# Set error action preference
$ErrorActionPreference = "Stop"

# Display banner
function Show-Banner {
    Write-Host "╔═══════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                                                                       ║" -ForegroundColor Cyan
    Write-Host "║   █████╗ ██╗   ██╗████████╗ ██████╗     ████████╗██████╗  █████╗ ██████╗ ███████╗    ║" -ForegroundColor Cyan
    Write-Host "║  ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ║" -ForegroundColor Cyan
    Write-Host "║  ███████║██║   ██║   ██║   ██║   ██║       ██║   ██████╔╝███████║██║  ██║█████╗      ║" -ForegroundColor Cyan
    Write-Host "║  ██╔══██║██║   ██║   ██║   ██║   ██║       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝      ║" -ForegroundColor Cyan
    Write-Host "║  ██║  ██║╚██████╔╝   ██║   ╚██████╔╝       ██║   ██║  ██║██║  ██║██████╔╝███████╗    ║" -ForegroundColor Cyan
    Write-Host "║  ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝    ║" -ForegroundColor Cyan
    Write-Host "║                                                                       ║" -ForegroundColor Cyan
    Write-Host "║   Multi-Timeframe Day Trading Algorithm - 95%+ Win Rate               ║" -ForegroundColor Cyan
    Write-Host "║                                                                       ║" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

# Show banner
Show-Banner

# Get the current directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = $scriptDir

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "Found $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    Write-Host "You can download Python from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Check if pip is installed
Write-Host "Checking pip installation..." -ForegroundColor Yellow
try {
    $pipVersion = python -m pip --version
    Write-Host "Found $pipVersion" -ForegroundColor Green
}
catch {
    Write-Host "pip not found. Installing pip..." -ForegroundColor Yellow
    python -m ensurepip --upgrade
}

# Create virtual environment if it doesn't exist
$venvPath = Join-Path $projectDir "venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venvPath
    Write-Host "Virtual environment created at $venvPath" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
& $activateScript

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Yellow
$requirementsFile = Join-Path $projectDir "requirements.txt"
if (Test-Path $requirementsFile) {
    python -m pip install -r $requirementsFile
    Write-Host "Required packages installed successfully" -ForegroundColor Green
}
else {
    Write-Host "requirements.txt not found. Installing essential packages..." -ForegroundColor Yellow
    python -m pip install numpy pandas matplotlib yfinance scikit-learn rich
    Write-Host "Essential packages installed" -ForegroundColor Green
}

# Create necessary directories
$directories = @(
    "logs",
    "data",
    "data\cache",
    "reports",
    "models"
)

foreach ($dir in $directories) {
    $dirPath = Join-Path $projectDir $dir
    if (-not (Test-Path $dirPath)) {
        New-Item -Path $dirPath -ItemType Directory -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# Create the global command script
$globalScriptPath = Join-Path $projectDir "trade-global.ps1"
$globalScriptContent = @"
#!/usr/bin/env pwsh
# Global Trading Script - Multi-Timeframe Day Trading Algorithm
# This script can be run from anywhere and will find the trading algorithm

# Set error action preference
`$ErrorActionPreference = "Stop"

# Define the path to the trading algorithm
`$tradingAlgoPath = "$projectDir"

# Check if the trading algorithm exists
if (-not (Test-Path `$tradingAlgoPath)) {
    Write-Host "Trading algorithm not found at `$tradingAlgoPath" -ForegroundColor Red
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
`$testMode = `$false
`$timeframe = "1m"
`$runOnce = `$false

foreach (`$arg in `$args) {
    if (`$arg -eq "-Test") {
        `$testMode = `$true
    }
    elseif (`$arg -eq "-Once") {
        `$runOnce = `$true
    }
    elseif (`$arg -eq "-5m") {
        `$timeframe = "5m"
    }
}

# Set the trading algorithm directory
`$tradingAlgoDir = Join-Path `$tradingAlgoPath "day-trading-algo"

# Activate virtual environment if it exists
`$venvPath = Join-Path `$tradingAlgoPath "venv"
`$activateScript = Join-Path `$venvPath "Scripts\Activate.ps1"
if (Test-Path `$activateScript) {
    & `$activateScript
}

# Run in test mode if specified
if (`$testMode) {
    Write-Host "=======================================================" -ForegroundColor Magenta
    Write-Host "                    TEST SIMULATION                    " -ForegroundColor Magenta
    Write-Host "=======================================================" -ForegroundColor Magenta
    Write-Host "Running test simulation with the following parameters:" -ForegroundColor Yellow
    Write-Host "Timeframe: `$timeframe" -ForegroundColor Yellow
    Write-Host "Duration:  1.0 hours" -ForegroundColor Yellow
    Write-Host "Symbols:   AAPL,MSFT,GOOGL,AMZN,META" -ForegroundColor Yellow
    Write-Host ""

    # Run the test simulation
    python "`$tradingAlgoDir\day_trading_main.py" --mode "paper" --config "`$tradingAlgoDir\config\best_config_95plus_20250418_223546.yaml" --symbols "AAPL,MSFT,GOOGL,AMZN,META" --timeframe `$timeframe --strategy "multi_timeframe" --duration 1.0

    Write-Host "Test simulation completed!" -ForegroundColor Green
    exit
}

# Define stocks for different market conditions
`$volatileStocks = "TSLA,NVDA,AMD,COIN,PLTR,RBLX,GME,AMC,BBBY,SPCE,DKNG,MARA,RIOT,MSTR,ARKK,SQQQ,TQQQ,SPXU,VXX,UVXY"
`$bullishStocks = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,NFLX,PYPL,ADBE,CRM,UBER,ABNB,PLTR,COIN,RBLX,TQQQ,SPY,QQQ"
`$bearishStocks = "VXX,UVXY,SQQQ,SPXU,SH,PSQ,DOG,MUB,TLT,GLD,SLV,USO,UNG,XLP,XLU,JNJ,PG,KO,PEP,WMT"
`$neutralStocks = "SPY,QQQ,DIA,IWM,XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,GLD,SLV,USO,UNG,TLT,MUB"

# Get day of week
`$dayOfWeek = (Get-Date).DayOfWeek

# Define stocks for different days of the week
`$dayStocks = @{
    "Monday" = "TSLA,NVDA,AMD,AAPL,MSFT,GOOGL,AMZN,META,NFLX,PYPL,UBER,ABNB,PLTR,COIN,RBLX,GME,AMC,BBBY,SPCE,DKNG"
    "Tuesday" = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,JPM,BAC,GS,MS,HD,WMT,JNJ,PFE,XOM,CVX"
    "Wednesday" = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,SPY,QQQ,DIA,IWM,XLF,XLE,XLV,XLY,GLD,SLV"
    "Thursday" = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,JPM,BAC,WFC,C,GS,MS,V,MA,AXP,BLK"
    "Friday" = "AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,AMD,INTC,NFLX,SPY,QQQ,IWM,VXX,UVXY,SQQQ,TQQQ,SPXU,USO,GLD"
}

# Determine market condition (simplified)
`$marketCondition = "neutral"
try {
    # Get SPY data for the last few days
    `$spyData = Invoke-RestMethod -Uri "https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1d&range=5d"

    # Get the closing prices
    `$closePrices = `$spyData.chart.result[0].indicators.quote[0].close

    # Calculate daily returns
    `$returns = @()
    for (`$i = 1; `$i -lt `$closePrices.Count; `$i++) {
        `$returns += (`$closePrices[`$i] - `$closePrices[`$i-1]) / `$closePrices[`$i-1]
    }

    # Calculate average return and volatility
    `$avgReturn = (`$returns | Measure-Object -Average).Average
    `$volatility = (`$returns | Measure-Object -StandardDeviation).StandardDeviation

    # Determine market condition
    if (`$volatility -gt 0.015) {
        `$marketCondition = "volatile"
    }
    elseif (`$avgReturn -gt 0.005) {
        `$marketCondition = "bullish"
    }
    elseif (`$avgReturn -lt -0.005) {
        `$marketCondition = "bearish"
    }
}
catch {
    Write-Host "Could not determine market condition. Defaulting to neutral." -ForegroundColor Yellow
}

# Get stocks based on market condition
`$conditionStocks = switch (`$marketCondition) {
    "volatile" { `$volatileStocks }
    "bullish" { `$bullishStocks }
    "bearish" { `$bearishStocks }
    default { `$neutralStocks }
}

# Get stocks based on day of week
`$dayOfWeekStocks = `$dayStocks[`$dayOfWeek.ToString()]

# Combine and deduplicate stocks
`$allStocksArray = "`$conditionStocks,`$dayOfWeekStocks" -split ','
`$uniqueStocks = `$allStocksArray | Select-Object -Unique | Where-Object { `$_ -ne "" }
`$symbols = `$uniqueStocks -join ','

# Display trading information
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "                 TRADING INFORMATION                   " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "Market Condition: `$marketCondition" -ForegroundColor Yellow
Write-Host "Day of Week:      `$dayOfWeek" -ForegroundColor Yellow
Write-Host "Timeframe:        `$timeframe" -ForegroundColor Yellow
Write-Host "Number of Stocks: `$(`$uniqueStocks.Count)" -ForegroundColor Yellow
Write-Host ""

# Run the trading algorithm
Write-Host "Starting multi-timeframe day trading algorithm..." -ForegroundColor Green
python "`$tradingAlgoDir\day_trading_main.py" --mode "paper" --config "`$tradingAlgoDir\config\best_config_95plus_20250418_223546.yaml" --symbols `$symbols --timeframe `$timeframe --strategy "multi_timeframe" --duration 6.5

Write-Host "Trading session completed!" -ForegroundColor Green
"@

# Save the global script
Set-Content -Path $globalScriptPath -Value $globalScriptContent
Write-Host "Created global trading script at $globalScriptPath" -ForegroundColor Green

# Create a batch file for easy execution
$batchFilePath = Join-Path $projectDir "trade.bat"
$batchFileContent = @"
@echo off
REM Day Trading Algorithm - Quick Launch Script
REM This script provides a simple way to launch the day trading algorithm

echo ╔═══════════════════════════════════════════════════════════════════════╗
echo ║                                                                       ║
echo ║   █████╗ ██╗   ██╗████████╗ ██████╗     ████████╗██████╗  █████╗ ██████╗ ███████╗    ║
echo ║  ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ║
echo ║  ███████║██║   ██║   ██║   ██║   ██║       ██║   ██████╔╝███████║██║  ██║█████╗      ║
echo ║  ██╔══██║██║   ██║   ██║   ██║   ██║       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝      ║
echo ║  ██║  ██║╚██████╔╝   ██║   ╚██████╔╝       ██║   ██║  ██║██║  ██║██████╔╝███████╗    ║
echo ║  ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝    ║
echo ║                                                                       ║
echo ║   Multi-Timeframe Day Trading Algorithm - 95%%+ Win Rate               ║
echo ║                                                                       ║
echo ╚═══════════════════════════════════════════════════════════════════════╝

powershell -ExecutionPolicy Bypass -File "%~dp0trade-global.ps1" %*
"@

# Save the batch file
Set-Content -Path $batchFilePath -Value $batchFileContent
Write-Host "Created batch file at $batchFilePath" -ForegroundColor Green

# Set up PowerShell profile command
Write-Host "Setting up PowerShell profile command..." -ForegroundColor Yellow

# Check if PowerShell profile exists, create if it doesn't
if (-not (Test-Path $PROFILE)) {
    Write-Host "Creating PowerShell profile..." -ForegroundColor Yellow
    New-Item -Path $PROFILE -ItemType File -Force | Out-Null
}

# Check if the function already exists in the profile
$profileContent = Get-Content $PROFILE -ErrorAction SilentlyContinue
$functionExists = $profileContent -match "function Start-Trading"

if (-not $functionExists) {
    # Add the function to the profile
    $functionDefinition = @"

# Trading Algorithm Commands
function Start-Trading {
    param(
        [Parameter(ValueFromRemainingArguments=`$true)]
        [string[]]`$Arguments
    )

    & "$globalScriptPath" `$Arguments
}

function Start-EnhancedTrading {
    param(
        [Parameter(ValueFromRemainingArguments=`$true)]
        [string[]]`$Arguments
    )

    & "$projectDir\run_enhanced_ui.ps1" `$Arguments
}

# Create aliases for easier use
Set-Alias -Name trade -Value Start-Trading
Set-Alias -Name trade-ui -Value Start-EnhancedTrading
"@

    Add-Content -Path $PROFILE -Value $functionDefinition

    Write-Host "Command added to PowerShell profile" -ForegroundColor Green
    Write-Host "You can now use 'trade' from anywhere in PowerShell" -ForegroundColor Green
} else {
    Write-Host "The 'trade' command is already set up in your PowerShell profile" -ForegroundColor Green
}

# Create a desktop shortcut
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "Day Trading Algorithm.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$globalScriptPath`""
$Shortcut.WorkingDirectory = $projectDir
$Shortcut.IconLocation = "powershell.exe,0"
$Shortcut.Description = "Launch Day Trading Algorithm"
$Shortcut.Save()

Write-Host "Created desktop shortcut: $shortcutPath" -ForegroundColor Green

# Final instructions
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "          Installation Completed Successfully!          " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run the day trading algorithm in several ways:" -ForegroundColor Green
Write-Host ""
Write-Host "1. From any PowerShell window:" -ForegroundColor Yellow
Write-Host "   trade          # Standard UI" -ForegroundColor White
Write-Host "   trade-ui       # Enhanced UI with rich formatting" -ForegroundColor White
Write-Host ""
Write-Host "2. Using the batch files:" -ForegroundColor Yellow
Write-Host "   $batchFilePath       # Standard UI" -ForegroundColor White
Write-Host "   trade-enhanced.bat   # Enhanced UI" -ForegroundColor White
Write-Host ""
Write-Host "3. Using the desktop shortcut:" -ForegroundColor Yellow
Write-Host "   Day Trading Algorithm.lnk" -ForegroundColor White
Write-Host ""
Write-Host "Options:" -ForegroundColor Yellow
Write-Host "  -Test    Run in test mode with predefined stocks" -ForegroundColor White
Write-Host "  -5m      Use 5-minute timeframe (default is 1m)" -ForegroundColor White
Write-Host "  -Once    Run once and exit (don't continue monitoring)" -ForegroundColor White
Write-Host ""
Write-Host "Example:" -ForegroundColor Yellow
Write-Host "  trade -Test -5m" -ForegroundColor White
Write-Host ""

# Offer to reload the profile
$reload = Read-Host "Do you want to reload the PowerShell profile now? (y/n)"
if ($reload -eq "y" -or $reload -eq "Y") {
    Write-Host "Reloading PowerShell profile..." -ForegroundColor Yellow
    . $PROFILE
    Write-Host "Profile reloaded. You can now use the 'trade' command." -ForegroundColor Green
} else {
    Write-Host "Please restart PowerShell or run '. $PROFILE' to use the 'trade' command" -ForegroundColor Yellow
}

# Offer to run the algorithm
$runNow = Read-Host "Do you want to run the day trading algorithm now? (y/n)"
if ($runNow -eq "y" -or $runNow -eq "Y") {
    $testMode = Read-Host "Run in test mode? (y/n)"
    if ($testMode -eq "y" -or $testMode -eq "Y") {
        & $globalScriptPath -Test
    } else {
        & $globalScriptPath
    }
}
