"""Unit test for Fix double redis.close() call bug.

This test verifies that RedisSessionStore.close() calls the underlying
Redis client's close() method exactly once, not twice.

Bug location: app/stores/session_store.py lines 492-493
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from app.stores.session_store import RedisSessionStore


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for testing."""
    mock = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def redis_store(mock_redis):
    """Create RedisSessionStore with mocked Redis client."""
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        store = RedisSessionStore(redis_url="redis://localhost:6379/0")
        return store


@pytest.mark.asyncio
async def test_redis_store_close_called_once(redis_store, mock_redis):
    """Test that close() calls redis.close() exactly once, not twice.

    P1-HIGH: The bug at lines 492-493 calls await self._redis.close()
    twice consecutively, which can cause ConnectionError or connection pool
    corruption. This test verifies the fix by asserting close() is called
    exactly once.
    """
    # Call the close method
    await redis_store.close()

    # Assert that the underlying Redis close() was called exactly once
    # This will FAIL with the current buggy implementation that calls close() twice
    mock_redis.close.assert_called_once()

    # Additional assertion: verify call count explicitly
    assert mock_redis.close.call_count == 1, (
        f"Expected redis.close() to be called once, but it was called "
        f"{mock_redis.close.call_count} times. This indicates the double close bug."
    )
