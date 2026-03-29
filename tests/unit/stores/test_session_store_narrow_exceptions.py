"""Test for narrowing broad Exception handlers in session_store ().

The issue: Line 406 in session_store.py uses `except Exception:` which is too broad
and can mask legitimate errors like ValidationError, ValueError, TypeError during
deserialization in RedisSessionStore.get_history().

This test verifies that after narrowing the exception handler, only specific
exceptions are caught, while unexpected errors propagate properly.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from app.stores.session_store import RedisSessionStore


@pytest.mark.asyncio
async def test_validation_error_caught_during_deserialization():
    """Test that ValidationError during deserialization is caught and handled gracefully.

    After narrowing the exception handler to catch only
    (ValidationError, ValueError, TypeError), this should still work - returning []
    instead of raising.
    """
    store = RedisSessionStore(redis_url="redis://localhost:6379/0")

    # Mock Redis to return invalid JSON that will cause ValidationError
    with patch.object(store, "_redis") as mock_redis:
        # Return data that will fail Pydantic validation
        mock_redis.get = AsyncMock(return_value=b'{"invalid": "data"}')

        # Should return empty list instead of raising
        result = await store.get_history("test-session")
        assert result == []


@pytest.mark.asyncio
async def test_value_error_caught_during_deserialization():
    """Test that ValueError during deserialization is caught and handled gracefully.

    After narrowing, ValueError should still be caught.
    """
    store = RedisSessionStore(redis_url="redis://localhost:6379/0")

    with patch.object(store, "_redis") as mock_redis:
        # Return malformed JSON that will cause ValueError
        mock_redis.get = AsyncMock(return_value=b"{malformed json}")

        # Should return empty list instead of raising
        result = await store.get_history("test-session")
        assert result == []


@pytest.mark.asyncio
async def test_unexpected_error_not_caught():
    """Test that unexpected errors (not in the narrow exception list) propagate.

    RED phase: This test should FAIL before fix because broad
    `except Exception:` catches RuntimeError during validation. After narrowing
    to specific exceptions, RuntimeError should propagate and this test will PASS.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    store = RedisSessionStore(redis_url="redis://localhost:6379/0")

    with (
        patch.object(store, "_redis") as mock_redis,
        patch.object(
            ModelMessagesTypeAdapter,
            "validate_json",
            side_effect=RuntimeError("Unexpected validation error"),
        ),
    ):
        # Return valid data from Redis
        mock_redis.get = AsyncMock(return_value=b"[]")

        # Before fix: RuntimeError is caught by broad `except Exception:`
        # and empty list is returned
        # After fix: RuntimeError should propagate
        with pytest.raises(RuntimeError, match="Unexpected validation error"):
            await store.get_history("test-session")
