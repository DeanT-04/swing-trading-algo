#!/usr/bin/env pwsh
# Fix for trade-ui command

# Get the current directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = $scriptDir

# Add the function to the profile
$functionDefinition = @"

# Enhanced Trading UI Command
function Start-EnhancedTrading {
    param(
        [Parameter(ValueFromRemainingArguments=`$true)]
        [string[]]`$Arguments
    )
    
    & "$projectDir\run_enhanced_ui.ps1" `$Arguments
}

# Create alias for enhanced UI
Set-Alias -Name trade-ui -Value Start-EnhancedTrading
"@

# Check if PowerShell profile exists, create if it doesn't
if (-not (Test-Path $PROFILE)) {
    Write-Host "Creating PowerShell profile..." -ForegroundColor Yellow
    New-Item -Path $PROFILE -ItemType File -Force | Out-Null
}

# Add the function to the profile
Add-Content -Path $PROFILE -Value $functionDefinition
    
Write-Host "The 'trade-ui' command has been added to your PowerShell profile" -ForegroundColor Green
Write-Host "Please restart PowerShell or run '. $PROFILE' to use the command" -ForegroundColor Yellow

# Offer to reload the profile
$reload = Read-Host "Do you want to reload the PowerShell profile now? (y/n)"
if ($reload -eq "y" -or $reload -eq "Y") {
    Write-Host "Reloading PowerShell profile..." -ForegroundColor Yellow
    . $PROFILE
    Write-Host "Profile reloaded. You can now use the 'trade-ui' command." -ForegroundColor Green
} else {
    Write-Host "Please restart PowerShell or run '. $PROFILE' to use the 'trade-ui' command" -ForegroundColor Yellow
}
