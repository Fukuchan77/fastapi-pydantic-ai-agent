"""Unit tests for logging configuration module.

RED PHASE: Writing failing tests for logging_config.py module that doesn't exist yet.
These tests MUST fail initially, then pass after implementation.
"""

import logging

import pytest

from app.config import Settings


def test_configure_logging_function_exists() -> None:
    """Test that configure_logging function exists and is callable.

    RED PHASE: This test will fail because app.logging_config module doesn't exist.
    """
    from app.logging_config import configure_logging

    assert callable(configure_logging)


def test_configure_logging_with_development_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logging is configured with DEBUG level in development.

    RED PHASE: This test will fail because configure_logging doesn't exist.
    """
    from app.logging_config import configure_logging

    # Set development environment
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "development")

    settings = Settings()

    # Clear any existing handlers to start fresh
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure logging
    configure_logging(settings)

    # Assert DEBUG level is set for development
    assert root_logger.level == logging.DEBUG


def test_configure_logging_with_production_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logging is configured with INFO level in production.

    RED PHASE: This test will fail because configure_logging doesn't exist.
    """
    from app.logging_config import configure_logging

    # Set production environment
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "production")

    settings = Settings()

    # Clear any existing handlers to start fresh
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure logging
    configure_logging(settings)

    # Assert INFO level is set for production
    assert root_logger.level == logging.INFO


def test_configure_logging_format_includes_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that log format includes timestamp, level, logger name, and message.

    RED PHASE: This test will fail because configure_logging doesn't exist.
    """
    from app.logging_config import configure_logging

    # Set up test environment
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "development")

    settings = Settings()

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure logging
    configure_logging(settings)

    # Check that at least one handler exists
    assert len(root_logger.handlers) > 0

    # Get the first handler and check its formatter
    handler = root_logger.handlers[0]
    assert handler.formatter is not None

    # Get the format string from the formatter
    format_string = handler.formatter._fmt

    # Assert required fields are in the format
    assert "%(asctime)s" in format_string, "Format must include timestamp"
    assert "%(levelname)s" in format_string, "Format must include log level"
    assert "%(name)s" in format_string, "Format must include logger name"
    assert "%(message)s" in format_string, "Format must include message"


def test_configure_logging_adds_console_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a console handler is properly configured.

    RED PHASE: This test will fail because configure_logging doesn't exist.
    """
    from app.logging_config import configure_logging

    # Set up test environment
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "development")

    settings = Settings()

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure logging
    configure_logging(settings)

    # Check that at least one handler exists
    assert len(root_logger.handlers) > 0

    # Check that at least one handler is a StreamHandler (console)
    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers
    )
    assert has_stream_handler, "Should have at least one StreamHandler for console output"


def test_configure_logging_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that calling configure_logging multiple times is safe (idempotent).

    RED PHASE: This test will fail because configure_logging doesn't exist.
    """
    from app.logging_config import configure_logging

    # Set up test environment
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "development")

    settings = Settings()

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure logging first time
    configure_logging(settings)
    first_handler_count = len(root_logger.handlers)

    # Configure logging second time
    configure_logging(settings)
    second_handler_count = len(root_logger.handlers)

    # Configure logging third time
    configure_logging(settings)
    third_handler_count = len(root_logger.handlers)

    # Handler count should not increase with multiple calls
    assert first_handler_count == second_handler_count == third_handler_count
    assert first_handler_count > 0, "Should have at least one handler after configuration"


def test_configure_logging_with_staging_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logging is configured with INFO level in staging (non-development).

    RED PHASE: This test will fail because configure_logging doesn't exist.
    """
    from app.logging_config import configure_logging

    # Set staging environment
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "staging")

    settings = Settings()

    # Clear any existing handlers to start fresh
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure logging
    configure_logging(settings)

    # Assert INFO level is set for staging (non-development)
    assert root_logger.level == logging.INFO
