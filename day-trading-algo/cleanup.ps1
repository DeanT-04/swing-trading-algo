#!/usr/bin/env pwsh
# Cleanup script to remove unnecessary files

# Set error action preference
$ErrorActionPreference = "Stop"

# Display banner
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "          Cleaning up unnecessary files                " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Define files to keep
$filesToKeep = @(
    "auto_trade.ps1",
    "simple_trade.ps1",
    "trade.bat",
    "day_trading_main.py",
    "README.md",
    "requirements.txt",
    "config\best_config_95plus_20250418_223546.yaml",
    "cleanup.ps1"
)

# Define directories to keep
$dirsToKeep = @(
    "src",
    "config",
    "models",
    "data",
    "reports"
)

# Get all PowerShell scripts in the day-trading-algo directory
$psScripts = Get-ChildItem -Path "day-trading-algo\*.ps1" -File

# Remove unnecessary PowerShell scripts
foreach ($script in $psScripts) {
    $relativePath = $script.FullName.Substring($script.FullName.IndexOf("day-trading-algo\"))
    if ($filesToKeep -notcontains $script.Name -and $script.Name -ne "cleanup.ps1") {
        Write-Host "Removing $relativePath" -ForegroundColor Yellow
        Remove-Item -Path $script.FullName -Force
    }
    else {
        Write-Host "Keeping $relativePath" -ForegroundColor Green
    }
}

# Get all batch files in the day-trading-algo directory
$batFiles = Get-ChildItem -Path "day-trading-algo\*.bat" -File

# Remove unnecessary batch files
foreach ($file in $batFiles) {
    $relativePath = $file.FullName.Substring($file.FullName.IndexOf("day-trading-algo\"))
    if ($filesToKeep -notcontains $file.Name -and $file.Name -ne "trade.bat") {
        Write-Host "Removing $relativePath" -ForegroundColor Yellow
        Remove-Item -Path $file.FullName -Force
    }
    else {
        Write-Host "Keeping $relativePath" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Cleanup completed!" -ForegroundColor Green
