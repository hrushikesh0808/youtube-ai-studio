"""
Application Logger

Provides a reusable logger for the entire project.
"""

import logging
from pathlib import Path

# Create application logger
logger = logging.getLogger("youtube_ai_studio")

# Set minimum log level
logger.setLevel(logging.INFO)

# Create log formatter
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# Create console handler
console_handler = logging.StreamHandler()

# Apply formatter to console handler
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

# Create logs directory if it doesn't exist
log_directory = Path("logs")
log_directory.mkdir(exist_ok=True)

# Log file path
log_file = log_directory / "youtube_ai_studio.log"

# Create file handler
file_handler = logging.FileHandler(
    log_file,
    encoding="utf-8"
)

# Apply formatter to file handler
file_handler.setFormatter(formatter)

# Add file handler to logger
logger.addHandler(file_handler)