@echo off
REM Direct launcher for Enhanced UI
REM This script provides a simple way to launch the enhanced UI directly

echo ╔═══════════════════════════════════════════════════════════════════════╗
echo ║                                                                       ║
echo ║   Enhanced UI for Day Trading Algorithm - Direct Launcher             ║
echo ║                                                                       ║
echo ╚═══════════════════════════════════════════════════════════════════════╝

powershell -ExecutionPolicy Bypass -File "%~dp0run_enhanced_ui.ps1" %*
