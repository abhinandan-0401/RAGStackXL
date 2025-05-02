"""
Logging configuration for RAGStackXL.
Uses loguru for enhanced logging capabilities.
"""
import sys
import os
from pathlib import Path
from loguru import logger

from app.config.settings import settings

# Configure logger
def configure_logging():
    """Configure loguru logger with settings from config."""
    # Clear existing handlers
    logger.remove()
    
    # Setup console logging
    logger.add(
        sys.stderr,
        level=settings.LOGGING.LEVEL,
        format=settings.LOGGING.FORMAT,
        colorize=True,
    )
    
    # Create log directory if it doesn't exist
    log_dir = Path(settings.BASE_DIR) / "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file logging
    logger.add(
        log_dir / "app.log",
        level=settings.LOGGING.LEVEL,
        format=settings.LOGGING.FORMAT,
        rotation=settings.LOGGING.ROTATION, 
        retention=settings.LOGGING.RETENTION,
        compression="zip",
        serialize=settings.LOGGING.JSON,
        backtrace=True,
        diagnose=True,
    )
    
    # Log system info at startup
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Log level: {settings.LOGGING.LEVEL}")
    
    return logger

# Export configured logger
log = configure_logging() 