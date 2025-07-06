"""
Logging utility for the application
"""

import logging
import sys
from const import LOG_FORMAT, LOG_LEVEL

def setup_logger():
    """Setup application logger"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )

def get_logger(name):
    """Get logger instance"""
    return logging.getLogger(name)