"""Tests for authentication failure logging.

This module tests that authentication failures are logged for security monitoring.
"""

import logging

import pytest
from fastapi import HTTPException

from app.config import Settings
from app.deps.auth import verify_api_key
from app.middleware.request_id import request_id_var


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock Settings instance with a test API key."""
    return Settings(
        api_key="test-secret-key",
        llm_model="openai:gpt-4o",
        llm_api_key="sk-test",
    )


@pytest.mark.asyncio
async def test_missing_api_key_logs_failure(
    mock_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that missing API key logs a security event."""
    # Arrange: Set request ID for correlation
    request_id_var.set("test-request-123")

    # Act: Call verify_api_key with missing key
    with caplog.at_level(logging.WARNING), pytest.raises(HTTPException):
        await verify_api_key(api_key=None, settings=mock_settings)

    # Assert: Security event was logged
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "authentication failed" in record.message.lower()
    assert "missing" in record.message.lower() or "none" in record.message.lower()
    assert "test-request-123" in record.message


@pytest.mark.asyncio
async def test_invalid_api_key_logs_failure(
    mock_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that invalid API key logs a security event."""
    # Arrange: Set request ID for correlation
    request_id_var.set("test-request-456")

    # Act: Call verify_api_key with wrong key
    with caplog.at_level(logging.WARNING), pytest.raises(HTTPException):
        await verify_api_key(api_key="wrong-key", settings=mock_settings)

    # Assert: Security event was logged
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "authentication failed" in record.message.lower()
    assert "invalid" in record.message.lower()
    assert "test-request-456" in record.message


@pytest.mark.asyncio
async def test_valid_api_key_does_not_log(
    mock_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that successful authentication does not log a warning."""
    # Arrange: Set request ID
    request_id_var.set("test-request-789")

    # Act: Call verify_api_key with correct key
    with caplog.at_level(logging.WARNING):
        await verify_api_key(api_key="test-secret-key", settings=mock_settings)

    # Assert: No warning was logged
    assert len(caplog.records) == 0


@pytest.mark.asyncio
async def test_auth_failure_does_not_log_api_key(
    mock_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that failed authentication does not log the API key (security risk)."""
    # Arrange
    request_id_var.set("test-request-security")

    # Act: Try with wrong key
    with caplog.at_level(logging.WARNING), pytest.raises(HTTPException):
        await verify_api_key(api_key="wrong-key", settings=mock_settings)

    # Assert: The actual API key value is not in the log message
    log_message = caplog.records[0].message
    assert "wrong-key" not in log_message
    assert "test-secret-key" not in log_message
    assert "sk-test" not in log_message


@pytest.mark.asyncio
async def test_auth_failure_includes_request_id_in_extra(
    mock_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that authentication failure includes request_id in log extra dict."""
    # Arrange
    request_id_var.set("test-request-extra")

    # Act
    with caplog.at_level(logging.WARNING), pytest.raises(HTTPException):
        await verify_api_key(api_key=None, settings=mock_settings)

    # Assert: request_id is in the log record's extra/context
    record = caplog.records[0]
    # Check if request_id was passed in extra dict (common logging pattern)
    # The exact location depends on logging configuration
    assert "test-request-extra" in record.message or hasattr(record, "request_id")
