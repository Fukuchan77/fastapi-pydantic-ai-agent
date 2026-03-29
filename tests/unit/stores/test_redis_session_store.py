"""Unit tests for RedisSessionStore implementation.

Redis-backed SessionStore for multi-instance deployments.
These are unit tests with mocked Redis client - no actual Redis required.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import RedisSessionStore


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for testing."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    mock.keys = AsyncMock(return_value=[])
    mock.ttl = AsyncMock(return_value=-1)
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def redis_store(mock_redis):
    """Create RedisSessionStore with mocked Redis client."""
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        store = RedisSessionStore(
            redis_url="redis://localhost:6379/0", session_ttl=3600, key_prefix="test_session:"
        )
        return store


@pytest.mark.asyncio
async def test_redis_store_get_history_empty(redis_store, mock_redis):
    """Test get_history returns empty list for non-existent session."""
    # RED: This test will fail because RedisSessionStore doesn't exist yet
    mock_redis.get.return_value = None

    history = await redis_store.get_history("test-session-id")

    assert history == []
    mock_redis.get.assert_called_once_with("test_session:test-session-id")


@pytest.mark.asyncio
async def test_redis_store_save_and_get_history(redis_store, mock_redis):
    """Test save_history stores messages and get_history retrieves them."""
    session_id = "test-session-123"
    messages = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[TextPart(content="Hi there!")]),
    ]

    # Mock serialized data using JSON (matching new implementation)
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    serialized = ModelMessagesTypeAdapter.dump_json(messages)
    mock_redis.get.return_value = serialized

    # Save history
    await redis_store.save_history(session_id, messages)

    # Verify set was called with correct key and TTL
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args
    assert call_args[0][0] == f"test_session:{session_id}"
    assert call_args[1]["ex"] == 3600  # TTL

    # Get history
    history = await redis_store.get_history(session_id)

    assert len(history) == 2
    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[1], ModelResponse)


@pytest.mark.asyncio
async def test_redis_store_clear(redis_store, mock_redis):
    """Test clear removes session data."""
    # RED: This test will fail because RedisSessionStore doesn't exist yet
    session_id = "test-session-456"

    await redis_store.clear(session_id)

    mock_redis.delete.assert_called_once_with("test_session:test-session-456")


@pytest.mark.asyncio
async def test_redis_store_cleanup_expired_sessions(redis_store, mock_redis):
    """Test cleanup_expired_sessions returns 0 since Redis handles TTL automatically."""
    removed_count = await redis_store.cleanup_expired_sessions()

    # Redis handles expiry automatically via TTL, so cleanup returns 0
    # and doesn't need to call any Redis operations
    assert removed_count == 0
    # Verify no Redis operations were called (Redis handles TTL automatically)
    mock_redis.keys.assert_not_called()
    mock_redis.delete.assert_not_called()


@pytest.mark.asyncio
async def test_redis_store_validate_session_id(redis_store):
    """Test session_id validation raises ValueError for invalid IDs."""
    # RED: This test will fail because RedisSessionStore doesn't exist yet
    with pytest.raises(ValueError, match="session_id cannot be empty"):
        await redis_store.get_history("")

    with pytest.raises(ValueError, match="session_id too long"):
        await redis_store.get_history("x" * 300)

    with pytest.raises(ValueError, match="invalid characters"):
        await redis_store.get_history("session@invalid!")


def test_redis_store_generate_session_id(redis_store):
    """Test generate_session_id returns valid UUID."""
    # RED: This test will fail because RedisSessionStore doesn't exist yet
    session_id = redis_store.generate_session_id()

    assert isinstance(session_id, str)
    assert len(session_id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    assert session_id.count("-") == 4
