"""Python logging configuration module.

This module configures Python's built-in logging system for the application.
It should be called early in the application startup sequence.
"""

import json
import logging
import sys
from typing import Any

from app.config import Settings
from app.middleware.request_id import request_id_var


class RequestIDFilter(logging.Filter):
    """Logging filter that attaches the current request id to every record.

    Reads `app.middleware.request_id.request_id_var` (populated per-request
    by `RequestIDMiddleware`) so every log record carries `request_id`
    without each call site inserting it manually (Req 13.4).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach the current request id to `record` and allow it through.

        Args:
            record: The log record being emitted.

        Returns:
            Always True (this filter never suppresses records).
        """
        record.request_id = request_id_var.get()
        return True


class JSONFormatter(logging.Formatter):
    """Formatter that renders each log record as a single-line JSON object.

    Emits `timestamp`, `level`, `logger`, `message`, and `request_id`
    (Req 13.3/13.4) plus `exc_info` when the record carries exception info.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render `record` as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON-encoded string representing the record.
        """
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", ""),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(settings: Settings) -> None:
    """Configure Python's built-in logging system.

    Sets up the root logger with:
    - Appropriate log level based on environment (DEBUG for development, INFO otherwise)
    - A console handler emitting structured JSON records (Req 13.3)
    - A `RequestIDFilter` attaching the current request id to every record (Req 13.4)

    This function is idempotent - it's safe to call multiple times.
    The first call configures logging, subsequent calls have no effect.
    This prevents duplicate handlers when settings are reloaded.

    Log levels:
        - development: DEBUG (verbose logging for troubleshooting)
        - staging/production: INFO (cleaner logs, only important messages)

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

    # Create console handler emitting structured JSON, with request_id attached
    # via a Filter rather than manual per-call insertion (Req 13.3, 13.4)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(JSONFormatter())
    console_handler.addFilter(RequestIDFilter())

    # Add handler to root logger
    root_logger.addHandler(console_handler)
