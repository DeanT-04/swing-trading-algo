#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utilities for the Swing Trading Algorithm.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def setup_logging(config: Dict, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        config: Logging configuration dictionary
        log_dir: Directory to store log files (optional)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Extract configuration
    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Create logger
    logger = logging.getLogger("swing_trading")
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        log_path = Path(log_dir)
        os.makedirs(log_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"swing_trading_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
