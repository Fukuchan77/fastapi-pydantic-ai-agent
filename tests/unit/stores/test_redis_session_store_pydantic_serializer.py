"""Tests for Task 21.2: Replace pickle with Pydantic serializer in RedisSessionStore.

Security: pickle.loads() is an RCE vector if Redis is compromised.
Solution: Use Pydantic AI's type-safe JSON serialization instead.
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
async def test_save_history_uses_json_serialization_not_pickle(session_store, mock_redis):
    """Task 21.2: Verify save_history uses JSON serialization instead of pickle.

    Security requirement: pickle.dumps() must not be used for serialization.
    """
    messages = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[TextPart(content="Hi there")]),
    ]

    await session_store.save_history("test-session", messages)

    # Verify set() was called
    assert mock_redis.set.called
    call_args = mock_redis.set.call_args

    # Extract the serialized data (second argument)
    serialized_data = call_args[0][1]

    # Task 21.2: Data should be JSON bytes, not pickle bytes
    # JSON starts with '[' or '{', pickle starts with '\x80' or other binary markers
    assert isinstance(serialized_data, bytes), "Serialized data should be bytes"
    assert serialized_data[0:1] in (b"[", b"{"), "Data should be JSON, not pickle"


@pytest.mark.asyncio
async def test_get_history_uses_json_deserialization_not_pickle(session_store, mock_redis):
    """Task 21.2: Verify get_history uses JSON deserialization instead of pickle.

    Security requirement: pickle.loads() must not be used for deserialization.
    RCE risk: If Redis is compromised, attacker can inject malicious pickle data.
    """
    # Create valid JSON-serialized messages
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    messages = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[TextPart(content="Hi there")]),
    ]

    # Serialize using Pydantic's TypeAdapter (expected behavior)
    json_data = ModelMessagesTypeAdapter.dump_json(messages)

    # Mock Redis to return JSON data
    mock_redis.get.return_value = json_data

    # Get history should deserialize JSON successfully
    result = await session_store.get_history("test-session")

    # Verify correct deserialization
    assert len(result) == 2
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[1], ModelResponse)
    assert result[0].parts[0].content == "Hello"
    assert result[1].parts[0].content == "Hi there"


@pytest.mark.asyncio
async def test_get_history_returns_empty_list_on_invalid_json(session_store, mock_redis):
    """Task 21.2: Verify graceful handling of corrupted JSON data.

    When Redis data is corrupted, should return empty list instead of raising exception.
    """
    # Mock Redis to return invalid JSON
    mock_redis.get.return_value = b"invalid json data {"

    # Should return empty list on parse error
    result = await session_store.get_history("test-session")

    assert result == []


@pytest.mark.asyncio
async def test_get_history_rejects_pickle_data(session_store, mock_redis):
    """Task 21.2 SECURITY: Verify that pickle data is NOT accepted.

    If Redis contains old pickle-serialized data, it should be rejected
    (return empty list) rather than unsafely deserialized.
    """
    import pickle

    messages = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[TextPart(content="Hi there")]),
    ]

    # Serialize using pickle (old, unsafe method)
    pickle_data = pickle.dumps(messages)

    # Mock Redis to return pickle data
    mock_redis.get.return_value = pickle_data

    # Should return empty list (pickle data is not valid JSON)
    result = await session_store.get_history("test-session")

    assert result == []
