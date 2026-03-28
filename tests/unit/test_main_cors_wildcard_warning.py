"""Unit tests for Task 19.1: CORS wildcard warning at startup.

This test ensures that a warning is logged when CORS_ORIGINS contains "*"
to prevent accidental production misconfiguration.
"""

import logging
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def test_cors_wildcard_logs_warning_at_startup(caplog: pytest.LogCaptureFixture) -> None:
    """Test that a warning is logged when CORS_ORIGINS contains wildcard.

    Task 19.1: Prevent accidental production misconfiguration by warning
    when CORS allows all origins.
    """
    # Clear settings cache to ensure fresh settings
    from app.config import get_settings

    get_settings.cache_clear()

    # Set environment variable for CORS_ORIGINS
    with patch.dict(os.environ, {"CORS_ORIGINS": "*"}, clear=False):
        # Clear cache again after setting env var
        get_settings.cache_clear()

        with caplog.at_level(logging.WARNING):
            # Import app fresh to trigger lifespan
            from app.main import app

            # Create TestClient to trigger lifespan startup
            with TestClient(app):
                pass  # Lifespan runs on context manager entry

    # Check that warning was logged
    warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]

    # Verify warning mentions CORS and wildcard
    found_cors_warning = False
    for record in warning_logs:
        message_lower = record.message.lower()
        if "cors" in message_lower and ("*" in record.message or "wildcard" in message_lower):
            found_cors_warning = True
            break

    assert found_cors_warning, (
        f"Expected CORS wildcard warning in logs. Found: {[r.message for r in warning_logs]}"
    )


def test_cors_specific_origins_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that NO warning is logged when CORS has specific origins.

    Task 19.1: Warning should only appear for wildcard "*", not for
    legitimate specific origins.
    """
    # Clear settings cache to ensure fresh settings
    from app.config import get_settings

    get_settings.cache_clear()

    # Set environment variable for specific CORS origins
    with patch.dict(
        os.environ,
        {"CORS_ORIGINS": "https://example.com,https://app.example.com"},
        clear=False,
    ):
        # Clear cache again after setting env var
        get_settings.cache_clear()

        with caplog.at_level(logging.WARNING):
            # Import app fresh to trigger lifespan
            from app.main import app

            # Create TestClient to trigger lifespan startup
            with TestClient(app):
                pass  # Lifespan runs on context manager entry

    # Check that NO CORS wildcard warning was logged
    warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]

    # Filter for CORS-related warnings
    cors_warnings = [
        record
        for record in warning_logs
        if "cors" in record.message.lower()
        and ("*" in record.message or "wildcard" in record.message.lower())
    ]

    assert len(cors_warnings) == 0, (
        f"Expected NO CORS wildcard warning for specific origins. "
        f"Found: {[r.message for r in cors_warnings]}"
    )


def test_cors_wildcard_in_list_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that warning is logged even when wildcard is in a list with other origins.

    Task 19.1: Having "*" anywhere in the origins list is a security risk.
    """
    # Clear settings cache to ensure fresh settings
    from app.config import get_settings

    get_settings.cache_clear()

    # Set environment variable with wildcard mixed with specific origins
    with patch.dict(
        os.environ,
        {"CORS_ORIGINS": "https://example.com,*,https://app.example.com"},
        clear=False,
    ):
        # Clear cache again after setting env var
        get_settings.cache_clear()

        with caplog.at_level(logging.WARNING):
            # Import app fresh to trigger lifespan
            from app.main import app

            # Create TestClient to trigger lifespan startup
            with TestClient(app):
                pass  # Lifespan runs on context manager entry

    # Check that warning was logged
    warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]

    # Verify warning mentions CORS and wildcard
    found_cors_warning = False
    for record in warning_logs:
        message_lower = record.message.lower()
        if "cors" in message_lower and ("*" in record.message or "wildcard" in message_lower):
            found_cors_warning = True
            break

    assert found_cors_warning, (
        f"Expected CORS wildcard warning in logs. Found: {[r.message for r in warning_logs]}"
    )
