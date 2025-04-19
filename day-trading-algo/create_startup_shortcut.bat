@echo off
echo Creating startup shortcut for Day Trading Algorithm

REM Get the full path to the auto_start_trader.bat file
set "SCRIPT_PATH=%~dp0auto_start_trader.bat"

REM Create a shortcut in the Windows Startup folder
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_PATH=%STARTUP_FOLDER%\Day Trading Algorithm.lnk"

REM Create the shortcut using PowerShell
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%SCRIPT_PATH%'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.Description = 'Start Day Trading Algorithm automatically at login'; $Shortcut.Save()"

if exist "%SHORTCUT_PATH%" (
    echo Shortcut created successfully at:
    echo %SHORTCUT_PATH%
    echo The Day Trading Algorithm will now start automatically when you log in.
) else (
    echo Failed to create shortcut.
)

pause
