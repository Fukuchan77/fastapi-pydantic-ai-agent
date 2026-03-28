"""Python logging configuration module.

This module configures Python's built-in logging system for the application.
It should be called early in the application startup sequence.
"""

import logging
import sys

from app.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure Python's built-in logging system.

    Sets up the root logger with:
    - Appropriate log level based on environment (DEBUG for development, INFO otherwise)
    - Console handler for output to stdout
    - Formatted log messages with timestamp, level, logger name, and message

    This function is idempotent - it's safe to call multiple times.
    The first call configures logging, subsequent calls have no effect.
    This prevents duplicate handlers when settings are reloaded.

    Log levels:
        - development: DEBUG (verbose logging for troubleshooting)
        - staging/production: INFO (cleaner logs, only important messages)

    Format:
        YYYY-MM-DD HH:MM:SS - LEVEL - logger.name - message

    Args:
        settings: Application settings instance containing app_env

    Example:
        >>> from app.config import get_settings
        >>> configure_logging(get_settings())
        # Python logging is now configured
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Check if already configured (idempotent behavior)
    # If handlers already exist, don't add more
    if root_logger.handlers:
        return

    # Set log level based on environment
    # Development: DEBUG level for detailed logging
    # Production/Staging: INFO level for cleaner logs
    log_level = logging.DEBUG if settings.app_env == "development" else logging.INFO

    root_logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Create formatter with required fields:
    # - asctime: timestamp
    # - levelname: log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    # - name: logger name
    # - message: log message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add formatter to handler
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)
