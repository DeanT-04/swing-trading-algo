# start-trading.ps1
# PowerShell script to start the day trading algorithm

# Navigate to the day-trading-algo directory
Set-Location -Path "$PSScriptRoot"

# Display startup message
Write-Host "Starting Day Trading Algorithm..." -ForegroundColor Green
Write-Host "Time: $(Get-Date)" -ForegroundColor Cyan
Write-Host "Working Directory: $PSScriptRoot" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor DarkGray

# Check if Python is available
try {
    $pythonVersion = python --version
    Write-Host "Using $pythonVersion" -ForegroundColor Cyan
}
catch {
    Write-Host "Error: Python not found. Please make sure Python is installed and in your PATH." -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Check if virtual environment exists and activate it
if (Test-Path -Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & ".\venv\Scripts\Activate.ps1"
}
else {
    Write-Host "No virtual environment found. Using system Python." -ForegroundColor Yellow
}

# Check and install required packages
Write-Host "Checking and installing required packages..." -ForegroundColor Cyan
try {
    python -m pip install -r requirements.txt
    Write-Host "All required packages installed successfully." -ForegroundColor Green
}
catch {
    Write-Host "Error installing packages: $_" -ForegroundColor Red
    Write-Host "Press any key to continue anyway..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Start the trading algorithm
try {
    Write-Host "Launching day trading algorithm..." -ForegroundColor Green

    # Check if we should run in headless mode (no UI)
    $headless = $args -contains "--headless"

    if ($headless) {
        Write-Host "Running in headless mode (no UI)..." -ForegroundColor Yellow
        python auto_trader.py --start-now --no-ui
    } else {
        python auto_trader.py --start-now
    }
}
catch {
    Write-Host "Error starting trading algorithm: $_" -ForegroundColor Red
}

# Keep the window open if there was an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "Trading algorithm exited with error code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
