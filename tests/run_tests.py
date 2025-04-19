#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run tests for the Swing Trading Algorithm.
"""

import os
import sys
import pytest


def main():
    """Run tests for the Swing Trading Algorithm."""
    print("Running tests for Swing Trading Algorithm...")
    
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Run pytest with coverage
    args = [
        "--verbose",
        "--cov=src",
        "--cov-report=term-missing",
        "tests"
    ]
    
    # Run the tests
    result = pytest.main(args)
    
    # Return the exit code
    return result


if __name__ == "__main__":
    sys.exit(main())
