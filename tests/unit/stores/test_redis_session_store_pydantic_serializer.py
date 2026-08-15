"""Tests for Replace pickle with Pydantic serializer in RedisSessionStore.

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
    """Verify save_history uses JSON serialization instead of pickle.

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

    # Data should be JSON bytes, not pickle bytes
    # JSON starts with '[' or '{', pickle starts with '\x80' or other binary markers
    assert isinstance(serialized_data, bytes), "Serialized data should be bytes"
    assert serialized_data[0:1] in (b"[", b"{"), "Data should be JSON, not pickle"


@pytest.mark.asyncio
async def test_get_history_uses_json_deserialization_not_pickle(session_store, mock_redis):
    """Verify get_history uses JSON deserialization instead of pickle.

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
async def test_get_history_deserializes_v1_shaped_payload(session_store, mock_redis):
    """Verify get_history deserializes history written by pydantic-ai v1.

    Req 5.7: the v2-adapter migration's Redis key-prefix cutover (Req 7.1)
    means a v2 store never again reads a key a pre-cutover ("v1") instance
    wrote, so no production read path can exercise this once the cutover
    lands — a fixture is the only way left to demonstrate the claim.

    The payload below is not round-tripped through the current (v2) adapter;
    it is the literal bytes `ModelMessagesTypeAdapter.dump_json()` produced
    under the pinned pre-migration `pydantic-ai-slim==1.107.2` for two
    messages equivalent to this module's other fixtures. Diffing it against
    the v2 shape shows exactly what v2 added: a `state` key on
    `ModelRequest` and a `cost` key inside `usage`, both optional with
    defaults, so a v1 payload omitting them must still validate cleanly.
    """
    v1_shaped_json = (
        b'[{"parts":[{"content":"Hello","timestamp":"2026-08-11T00:00:00Z",'
        b'"part_kind":"user-prompt"}],"timestamp":null,"instructions":null,'
        b'"kind":"request","run_id":null,"conversation_id":null,"metadata":null},'
        b'{"parts":[{"content":"Hi there","id":null,"provider_name":null,'
        b'"provider_details":null,"part_kind":"text"}],"usage":{"input_tokens":0,'
        b'"cache_write_tokens":0,"cache_read_tokens":0,"output_tokens":0,'
        b'"input_audio_tokens":0,"cache_audio_read_tokens":0,"output_audio_tokens":0,'
        b'"details":{}},"model_name":null,"timestamp":"2026-08-11T00:00:01Z",'
        b'"kind":"response","provider_name":null,"provider_url":null,'
        b'"provider_details":null,"provider_response_id":null,"finish_reason":null,'
        b'"run_id":null,"conversation_id":null,"metadata":null,"state":"complete"}]'
    )
    mock_redis.get.return_value = v1_shaped_json

    result = await session_store.get_history("test-session")

    assert len(result) == 2
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[1], ModelResponse)
    assert result[0].parts[0].content == "Hello"
    assert result[1].parts[0].content == "Hi there"


@pytest.mark.asyncio
async def test_get_history_returns_empty_list_on_invalid_json(session_store, mock_redis):
    """Verify graceful handling of corrupted JSON data.

    When Redis data is corrupted, should return empty list instead of raising exception.
    """
    # Mock Redis to return invalid JSON
    mock_redis.get.return_value = b"invalid json data {"

    # Should return empty list on parse error
    result = await session_store.get_history("test-session")

    assert result == []


@pytest.mark.asyncio
async def test_get_history_rejects_pickle_data(session_store, mock_redis):
    """SECURITY: Verify that pickle data is NOT accepted.

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
