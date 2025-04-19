#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Day Trading Algorithm Runner

This script is a wrapper for day_trading_main.py for backward compatibility.
Please use day_trading_main.py directly for new development.
"""

import os
import sys
import subprocess

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def main():
    """Main function that redirects to day_trading_main.py."""
    print("Note: run_day_trading.py is deprecated. Please use day_trading_main.py instead.")
    print("Redirecting to day_trading_main.py...\n")

    # Get command line arguments
    args = sys.argv[1:]

    # Construct command
    cmd = [sys.executable, "day_trading_main.py"] + args

    # Execute command
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
