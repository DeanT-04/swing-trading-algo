# This script will set up your PowerShell profile with the Start-DayTrading command

# Install required packages
Write-Host "Installing required packages..." -ForegroundColor Cyan
try {
    $projectDir = "C:\Users\deana\Documents\Coding projects\trading-algo\swing-trading-algo\day-trading-algo"
    Set-Location -Path $projectDir
    python -m pip install -r requirements.txt
    Write-Host "All required packages installed successfully." -ForegroundColor Green
}
catch {
    Write-Host "Error installing packages: $_" -ForegroundColor Red
    Write-Host "Press any key to continue anyway..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Check if profile exists, create if it doesn't
if (!(Test-Path -Path $PROFILE)) {
    Write-Host "Creating PowerShell profile..." -ForegroundColor Cyan
    New-Item -ItemType File -Path $PROFILE -Force | Out-Null
}

# Check if the function already exists in the profile
$profileContent = Get-Content -Path $PROFILE -ErrorAction SilentlyContinue
$functionExists = $profileContent -match "function Start-DayTrading"

if (!$functionExists) {
    # Add the function to the profile
    Write-Host "Adding Start-DayTrading function to your PowerShell profile..." -ForegroundColor Cyan

    $functionDefinition = @"

# Day Trading Algorithm shortcut
function Start-DayTrading {
    # Navigate to the script directory and run it
    & "C:\Users\deana\Documents\Coding projects\trading-algo\swing-trading-algo\day-trading-algo\start-trading.ps1"
}
"@

    Add-Content -Path $PROFILE -Value $functionDefinition

    Write-Host "Function added successfully!" -ForegroundColor Green
    Write-Host "You can now use the 'Start-DayTrading' command from any PowerShell window." -ForegroundColor Green
}
else {
    Write-Host "The Start-DayTrading function already exists in your profile." -ForegroundColor Yellow
}

# Set execution policy to allow running scripts
try {
    $currentPolicy = Get-ExecutionPolicy -Scope CurrentUser
    if ($currentPolicy -eq "Restricted" -or $currentPolicy -eq "AllSigned") {
        Write-Host "Setting execution policy to RemoteSigned for CurrentUser..." -ForegroundColor Cyan
        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
        Write-Host "Execution policy updated." -ForegroundColor Green
    }
}
catch {
    Write-Host "Warning: Could not update execution policy. You may need to run PowerShell as Administrator." -ForegroundColor Yellow
}

# Reload the profile
Write-Host "Reloading PowerShell profile..." -ForegroundColor Cyan
try {
    . $PROFILE
    Write-Host "Profile reloaded successfully!" -ForegroundColor Green
    Write-Host "You can now use the 'Start-DayTrading' command." -ForegroundColor Green
}
catch {
    Write-Host "Error reloading profile: $_" -ForegroundColor Red
    Write-Host "Please restart PowerShell or manually reload your profile with: . `$PROFILE" -ForegroundColor Yellow
}

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "You can now start the day trading algorithm by typing 'Start-DayTrading' in any PowerShell window." -ForegroundColor Cyan
