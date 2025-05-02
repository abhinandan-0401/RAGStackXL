"""
Logging utilities for the application.
"""
from loguru import logger

# Configure logger
logger.add("logs/app.log", rotation="10 MB", level="INFO")

# Create a log object that can be imported elsewhere
log = logger 