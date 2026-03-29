"""Verify RedisSessionStore.get_history() logs exceptions before swallowing them.

requires adding logger.warning() before silent exception swallow
in RedisSessionStore.get_history() to improve production debugging.
"""

import logging
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from app.stores.session_store import RedisSessionStore


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock = AsyncMock()
    mock.get = AsyncMock()
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def session_store(mock_redis):
    """Create RedisSessionStore with mocked Redis client."""
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        store = RedisSessionStore(redis_url="redis://localhost:6379/0")
        return store


@pytest.mark.asyncio
async def test_get_history_logs_warning_on_deserialization_error(
    session_store, mock_redis, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify get_history() logs warning when deserialization fails.

    Silent exception swallow makes production debugging difficult.
    This test ensures warnings are logged with exc_info=True for traceability.
    """
    # Mock Redis to return invalid JSON data that will fail deserialization
    mock_redis.get.return_value = b"invalid json data {{{"

    # Capture log output at WARNING level
    with caplog.at_level(logging.WARNING):
        result = await session_store.get_history("test-session")

    # Should return empty list (graceful degradation)
    assert result == []

    # Should log warning with session_id and exception details
    assert len(caplog.records) > 0
    assert any("test-session" in record.message for record in caplog.records)
    assert any(record.levelname == "WARNING" for record in caplog.records)

    # Should log exception info (exc_info=True)
    assert any(record.exc_info is not None for record in caplog.records)


@pytest.mark.asyncio
async def test_get_history_logs_warning_with_exc_info_true(
    session_store, mock_redis, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify get_history() logs exception with exc_info=True for full traceback.

    exc_info=True is critical for production debugging - it includes
    the full exception traceback in logs, not just the message.
    """
    # Mock Redis to return data that triggers ValidationError
    mock_redis.get.return_value = b'{"invalid": "structure"}'

    with caplog.at_level(logging.WARNING):
        result = await session_store.get_history("session-123")

    assert result == []

    # Verify exc_info=True was used (provides full traceback)
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warning_records) > 0
    assert any(record.exc_info is not None for record in warning_records)


@pytest.mark.asyncio
async def test_get_history_returns_empty_list_after_logging_error(
    session_store, mock_redis
) -> None:
    """Verify get_history() still returns empty list after logging (graceful degradation).

    After adding logging, the method should still return empty list
    for backward compatibility and graceful error handling.
    """
    # Mock Redis to return invalid data
    mock_redis.get.return_value = b"completely invalid"

    result = await session_store.get_history("test-session")

    # Should still return empty list for graceful degradation
    assert result == []
    assert isinstance(result, list)
