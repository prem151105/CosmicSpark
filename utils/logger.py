"""
Logging configuration for MOSDAC AI Help Bot
"""

import sys
import os
from loguru import logger
from .config import get_settings

def setup_logger():
    """Setup application logging"""
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler
    log_dir = os.path.dirname(settings.LOG_FILE)
    if log_dir:  # Only try to create directory if a path is specified
        os.makedirs(log_dir, exist_ok=True)
    
    if settings.LOG_FILE:  # Only add file handler if LOG_FILE is not empty
        logger.add(
            settings.LOG_FILE,
            level=settings.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    return logger