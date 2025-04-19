#!/usr/bin/env pwsh
# Enhanced UI Trading Script - Multi-Timeframe Day Trading Algorithm
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

# Change to the trading algorithm directory
Set-Location $tradingAlgoPath

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
if ($testMode) {
    Write-Host "Running Enhanced UI in TEST mode with timeframe: $timeframe" -ForegroundColor Magenta
    & "$tradingAlgoPath\scripts\run_enhanced_ui.ps1" -Test -timeframe $timeframe
} else {
    Write-Host "Running Enhanced UI with timeframe: $timeframe" -ForegroundColor Green
    & "$tradingAlgoPath\scripts\run_enhanced_ui.ps1" -timeframe $timeframe
}
