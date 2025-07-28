"""Coloured logging configuration for the benchmarking system.

This module provides a pre-configured logger with coloured output formatting
for improved readability during benchmark execution and debugging.
"""

from __future__ import annotations

import logging
import re
from typing import ClassVar


class LogMessageFilter(logging.Filter):
    """A logging filter to remove problematic control characters from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records, removing only problematic control characters.

        Returns:
            True if the log record should be processed, False otherwise.
        """
        if isinstance(record.msg, str):
            # Remove only problematic control characters, preserve emoji and printable Unicode
            record.msg = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", record.msg)
        if record.exc_text:
            record.exc_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", record.exc_text)
        return True


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colour codes to different log levels."""

    # ANSI colour codes
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[90m",  # Grey
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET: ClassVar[str] = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with appropriate colours.

        Returns:
            The formatted log record with appropriate colours.
        """
        # Get the colour for this log level
        colour = self.COLORS.get(record.levelname, "")

        # Format the message
        formatted = super().format(record)

        # Apply colour to the entire line
        if colour:
            formatted = f"{colour}{formatted}{self.RESET}"

        return formatted


# Create and configure logger with colour formatting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with colour formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))

# Add the custom filter to the console handler
console_handler.addFilter(LogMessageFilter())

# Add handler to logger
logger.addHandler(console_handler)

# Prevent duplicate logs from root logger
logger.propagate = False
