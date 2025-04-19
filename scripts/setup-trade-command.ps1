#!/usr/bin/env pwsh
# Setup script to create a global 'trade' command

# Set error action preference
$ErrorActionPreference = "Stop"

# Display banner
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "          Setting up global 'trade' command            " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Define the path to the trading algorithm
$tradingAlgoPath = "C:\Users\deana\Documents\Coding projects\trading-algo\swing-trading-algo"
$scriptPath = Join-Path $tradingAlgoPath "trade-anywhere.ps1"

# Check if the trading algorithm exists
if (-not (Test-Path $tradingAlgoPath)) {
    Write-Host "Trading algorithm not found at $tradingAlgoPath" -ForegroundColor Red
    Write-Host "Please update the script with the correct path" -ForegroundColor Red
    exit 1
}

# Check if PowerShell profile exists, create if it doesn't
if (-not (Test-Path $PROFILE)) {
    Write-Host "Creating PowerShell profile..." -ForegroundColor Yellow
    New-Item -Path $PROFILE -ItemType File -Force | Out-Null
}

# Check if the function already exists in the profile
$profileContent = Get-Content $PROFILE -ErrorAction SilentlyContinue
$functionExists = $profileContent -match "function Start-Trading"

if (-not $functionExists) {
    Write-Host "Adding 'trade' command to PowerShell profile..." -ForegroundColor Yellow
    
    # Add the function to the profile
    $functionDefinition = @"

# Trading Algorithm Command
function Start-Trading {
    param(
        [Parameter(ValueFromRemainingArguments=`$true)]
        [string[]]`$Arguments
    )
    
    & "$scriptPath" `$Arguments
}

# Create an alias for easier use
Set-Alias -Name trade -Value Start-Trading
"@
    
    Add-Content -Path $PROFILE -Value $functionDefinition
    
    Write-Host "Command added successfully!" -ForegroundColor Green
    Write-Host "You can now use 'trade' from anywhere in PowerShell" -ForegroundColor Green
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  trade" -ForegroundColor Yellow
    Write-Host "  trade -Test" -ForegroundColor Yellow
    Write-Host "  trade -5m" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "NOTE: You need to restart PowerShell or run 'Import-Module $PROFILE' for changes to take effect" -ForegroundColor Magenta
} else {
    Write-Host "The 'trade' command is already set up in your PowerShell profile" -ForegroundColor Green
}

# Offer to reload the profile
$reload = Read-Host "Do you want to reload the PowerShell profile now? (y/n)"
if ($reload -eq "y" -or $reload -eq "Y") {
    Write-Host "Reloading PowerShell profile..." -ForegroundColor Yellow
    . $PROFILE
    Write-Host "Profile reloaded. You can now use the 'trade' command from anywhere." -ForegroundColor Green
} else {
    Write-Host "Please restart PowerShell or run 'Import-Module $PROFILE' to use the 'trade' command" -ForegroundColor Yellow
}
