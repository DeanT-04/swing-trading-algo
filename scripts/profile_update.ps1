# Enhanced Trading UI Command
function Start-EnhancedTrading {
    param(
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$Arguments
    )

    & "C:\Users\deana\Documents\Coding projects\trading-algo\swing-trading-algo\scripts\trade-ui-anywhere.ps1" $Arguments
}

# Create alias for enhanced UI
Set-Alias -Name trade-ui -Value Start-EnhancedTrading
