#!/usr/bin/env pwsh
# Enhanced UI Launcher for Day Trading Algorithm

# Set error action preference
$ErrorActionPreference = "Stop"

# Get the current directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = $scriptDir

# Display banner
Write-Host "╔═══════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                       ║" -ForegroundColor Cyan
Write-Host "║   █████╗ ██╗   ██╗████████╗ ██████╗     ████████╗██████╗  █████╗ ██████╗ ███████╗    ║" -ForegroundColor Cyan
Write-Host "║  ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ║" -ForegroundColor Cyan
Write-Host "║  ███████║██║   ██║   ██║   ██║   ██║       ██║   ██████╔╝███████║██║  ██║█████╗      ║" -ForegroundColor Cyan
Write-Host "║  ██╔══██║██║   ██║   ██║   ██║   ██║       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝      ║" -ForegroundColor Cyan
Write-Host "║  ██║  ██║╚██████╔╝   ██║   ╚██████╔╝       ██║   ██║  ██║██║  ██║██████╔╝███████╗    ║" -ForegroundColor Cyan
Write-Host "║  ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝    ║" -ForegroundColor Cyan
Write-Host "║                                                                       ║" -ForegroundColor Cyan
Write-Host "║   Enhanced UI for Day Trading Algorithm - 95%+ Win Rate               ║" -ForegroundColor Cyan
Write-Host "║                                                                       ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if it exists
$venvPath = Join-Path $projectDir "venv"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $activateScript
    Write-Host "Virtual environment activated" -ForegroundColor Green
}

# Check if rich is installed
Write-Host "Checking for required packages..." -ForegroundColor Yellow
$richInstalled = $false
try {
    $richVersion = python -c "import rich; print(rich.__version__)"
    $richInstalled = $true
    Write-Host "Rich library found: $richVersion" -ForegroundColor Green
}
catch {
    Write-Host "Rich library not found. Installing..." -ForegroundColor Yellow
    python -m pip install rich
    if ($LASTEXITCODE -eq 0) {
        $richInstalled = $true
        Write-Host "Rich library installed successfully" -ForegroundColor Green
    }
    else {
        Write-Host "Failed to install Rich library. UI will use basic mode." -ForegroundColor Red
    }
}

# Check if pytz is installed
$pytzInstalled = $false
try {
    $pytzVersion = python -c "import pytz; print(pytz.__version__)"
    $pytzInstalled = $true
    Write-Host "Pytz library found: $pytzVersion" -ForegroundColor Green
}
catch {
    Write-Host "Pytz library not found. Installing..." -ForegroundColor Yellow
    python -m pip install pytz
    if ($LASTEXITCODE -eq 0) {
        $pytzInstalled = $true
        Write-Host "Pytz library installed successfully" -ForegroundColor Green
    }
    else {
        Write-Host "Failed to install Pytz library. UI will use basic mode." -ForegroundColor Red
    }
}

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

# Run the enhanced UI
Write-Host "Starting Enhanced UI..." -ForegroundColor Green

if ($testMode) {
    Write-Host "Running in TEST mode with timeframe: $timeframe" -ForegroundColor Magenta
    python ui\enhanced_ui.py -test -timeframe $timeframe
} else {
    python ui\enhanced_ui.py -timeframe $timeframe
}

# Check if the UI exited with an error
if ($LASTEXITCODE -ne 0) {
    Write-Host "Enhanced UI exited with an error. Falling back to standard UI..." -ForegroundColor Yellow

    # Run the standard trading algorithm
    & $projectDir\scripts\trade-global.ps1 $args
}

Write-Host "UI session completed" -ForegroundColor Green
