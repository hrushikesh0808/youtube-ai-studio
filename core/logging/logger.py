"""
Application Logger

Provides a reusable logger for the entire project.
"""

import logging

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