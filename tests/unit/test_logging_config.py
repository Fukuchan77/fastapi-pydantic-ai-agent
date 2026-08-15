"""Unit tests for logging configuration module.

Covers structured JSON log records (Req 13.3) and `request_id` attachment
via a logging `Filter` rather than manual per-call insertion (Req 13.4).
"""

import json
import logging

import pytest

from app.config import Settings
from app.middleware.request_id import request_id_var


def test_configure_logging_function_exists() -> None:
    """Test that configure_logging function exists and is callable."""
    from app.logging_config import configure_logging

    assert callable(configure_logging)


def test_configure_logging_with_development_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logging is configured with DEBUG level in development."""
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
    """Test that logging is configured with INFO level in production."""
    from app.logging_config import configure_logging

    # Set production environment
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "production")
    # Production forbids a wildcard host allow-list; unrelated to logging.
    monkeypatch.setenv("ALLOWED_HOSTS", "api.example.com")

    settings = Settings()

    # Clear any existing handlers to start fresh
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure logging
    configure_logging(settings)

    # Assert INFO level is set for production
    assert root_logger.level == logging.INFO


def test_configure_logging_with_staging_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logging is configured with INFO level in staging (non-development)."""
    from app.logging_config import configure_logging

    # Set staging environment
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "staging")
    # Non-development also forbids a wildcard host allow-list; unrelated to logging.
    monkeypatch.setenv("ALLOWED_HOSTS", "api.example.com")

    settings = Settings()

    # Clear any existing handlers to start fresh
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure logging
    configure_logging(settings)

    # Assert INFO level is set for staging (non-development)
    assert root_logger.level == logging.INFO


def test_configure_logging_adds_console_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a console handler is properly configured."""
    from app.logging_config import configure_logging

    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "development")

    settings = Settings()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    configure_logging(settings)

    assert len(root_logger.handlers) > 0
    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers
    )
    assert has_stream_handler, "Should have at least one StreamHandler for console output"


def test_configure_logging_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that calling configure_logging multiple times is safe (idempotent)."""
    from app.logging_config import configure_logging

    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "development")

    settings = Settings()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    configure_logging(settings)
    first_handler_count = len(root_logger.handlers)

    configure_logging(settings)
    second_handler_count = len(root_logger.handlers)

    configure_logging(settings)
    third_handler_count = len(root_logger.handlers)

    assert first_handler_count == second_handler_count == third_handler_count
    assert first_handler_count > 0, "Should have at least one handler after configuration"


# ---------------------------------------------------------------------------
# Structured JSON log records (Req 13.3)
# ---------------------------------------------------------------------------


def test_configure_logging_emits_valid_json_with_required_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every emitted log record is a single line of JSON with the required fields."""
    from app.logging_config import configure_logging

    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "development")

    settings = Settings()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    configure_logging(settings)

    logger = logging.getLogger("test.json.logger")
    logger.info("hello world")

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.json.logger"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload
    assert "request_id" in payload


def test_configure_logging_json_records_include_exception_info(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A logged exception's traceback is included in the JSON record."""
    from app.logging_config import configure_logging

    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "development")

    settings = Settings()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    configure_logging(settings)

    logger = logging.getLogger("test.json.exception")
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("something failed")

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["message"] == "something failed"
    assert "exc_info" in payload
    assert "ValueError: boom" in payload["exc_info"]


# ---------------------------------------------------------------------------
# request_id attached via a logging Filter (Req 13.4)
# ---------------------------------------------------------------------------


def test_configure_logging_attaches_request_id_from_context_var(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every log record carries the current request_id without manual insertion."""
    from app.logging_config import configure_logging

    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "development")

    settings = Settings()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    configure_logging(settings)

    token = request_id_var.set("test-request-id-abc123")
    try:
        logger = logging.getLogger("test.json.request_id")
        logger.info("request handled")
    finally:
        request_id_var.reset(token)

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["request_id"] == "test-request-id-abc123"


def test_configure_logging_request_id_defaults_to_empty_outside_a_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Log records emitted with no active request_id context still carry the field."""
    from app.logging_config import configure_logging

    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
    monkeypatch.setenv("APP_ENV", "development")

    settings = Settings()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    configure_logging(settings)

    logger = logging.getLogger("test.json.no_request_id")
    logger.info("background task log")

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["request_id"] == request_id_var.get()


def test_request_id_filter_sets_record_attribute_from_context_var() -> None:
    """`RequestIDFilter` attaches the contextvar's current value to the record."""
    from app.logging_config import RequestIDFilter

    token = request_id_var.set("filter-test-id")
    try:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="msg",
            args=None,
            exc_info=None,
        )
        result = RequestIDFilter().filter(record)
    finally:
        request_id_var.reset(token)

    assert result is True
    assert record.request_id == "filter-test-id"


def test_json_formatter_produces_valid_json() -> None:
    """`JSONFormatter.format()` returns a single-line JSON object."""
    from app.logging_config import JSONFormatter
    from app.logging_config import RequestIDFilter

    record = logging.LogRecord(
        name="test.formatter",
        level=logging.WARNING,
        pathname=__file__,
        lineno=42,
        msg="a warning: %s",
        args=("detail",),
        exc_info=None,
    )
    RequestIDFilter().filter(record)

    formatted = JSONFormatter().format(record)
    payload = json.loads(formatted)

    assert payload["level"] == "WARNING"
    assert payload["logger"] == "test.formatter"
    assert payload["message"] == "a warning: detail"
    assert "request_id" in payload
