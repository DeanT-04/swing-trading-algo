# PowerShell script to start the day trading algorithm with enhanced console UI

# Get the directory where the day trading algorithm is located
$tradingAlgoPath = "C:\Users\deana\Documents\Coding projects\trading-algo\swing-trading-algo\day-trading-algo"

# Change to the trading algorithm directory
Set-Location -Path $tradingAlgoPath

# Check if virtual environment exists and activate it
if (Test-Path "..\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    & "..\venv\Scripts\Activate.ps1"
} else {
    Write-Host "Virtual environment not found, using system Python"
}

# Check if rich library is installed
$richInstalled = python -c "try: import rich; print('installed'); except ImportError: print('not installed')"
if ($richInstalled -eq "not installed") {
    Write-Host "Installing rich library for enhanced console output..."
    pip install rich
}

# Start the day trading algorithm with enhanced console UI
Write-Host "Starting day trading algorithm with enhanced console UI..."
python auto_trader.py --no-ui --start-now

# Keep the window open if there was an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error occurred. Press any key to close this window."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
