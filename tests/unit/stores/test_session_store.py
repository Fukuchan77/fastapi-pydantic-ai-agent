"""Unit tests for SessionStore Protocol and InMemorySessionStore implementation.

Tests cover:
- SessionStore Protocol interface validation
- InMemorySessionStore CRUD operations
- History ordering and append behavior
- Session ID validation
- Message validation
- TTL-based session cleanup
- LRU eviction when max_sessions exceeded
- Per-session locking for concurrency
- Edge cases (empty sessions, special characters, etc.)
"""

import asyncio
import time
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore
from app.stores.session_store import SessionStore


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol Interface Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionStoreProtocol:
    """Verify SessionStore Protocol defines the expected interface."""

    def test_protocol_has_get_history_method(self) -> None:
        """Protocol declares get_history() method."""
        assert hasattr(SessionStore, "get_history")

    def test_protocol_has_save_history_method(self) -> None:
        """Protocol declares save_history() method."""
        assert hasattr(SessionStore, "save_history")

    def test_protocol_has_clear_method(self) -> None:
        """Protocol declares clear() method."""
        assert hasattr(SessionStore, "clear")

    def test_protocol_has_cleanup_expired_sessions_method(self) -> None:
        """Protocol declares cleanup_expired_sessions() method (Task 3.15)."""
        assert hasattr(SessionStore, "cleanup_expired_sessions")

    def test_in_memory_store_satisfies_protocol(self) -> None:
        """InMemorySessionStore satisfies SessionStore Protocol."""
        store = InMemorySessionStore()
        # Runtime check: store should have all required methods
        assert hasattr(store, "get_history")
        assert hasattr(store, "save_history")
        assert hasattr(store, "clear")
        assert hasattr(store, "cleanup_expired_sessions")


# ═══════════════════════════════════════════════════════════════════════════════
# Construction Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestInMemorySessionStoreConstruction:
    """Test InMemorySessionStore initialization and configuration."""

    def test_default_construction(self) -> None:
        """Default constructor creates empty store with default limits."""
        store = InMemorySessionStore()
        assert store.max_messages == 1000
        assert store.session_ttl == 3600
        assert store.max_sessions == 10_000

    def test_custom_max_messages(self) -> None:
        """Constructor accepts custom max_messages limit."""
        store = InMemorySessionStore(max_messages=500)
        assert store.max_messages == 500

    def test_custom_session_ttl(self) -> None:
        """Constructor accepts custom session_ttl."""
        store = InMemorySessionStore(session_ttl=1800)
        assert store.session_ttl == 1800

    def test_custom_max_sessions(self) -> None:
        """Constructor accepts custom max_sessions limit (Task 3.16)."""
        store = InMemorySessionStore(max_sessions=5000)
        assert store.max_sessions == 5000

    def test_default_max_sessions_constant(self) -> None:
        """DEFAULT_MAX_SESSIONS constant is defined."""
        assert InMemorySessionStore.DEFAULT_MAX_SESSIONS == 10_000


# ═══════════════════════════════════════════════════════════════════════════════
# get_history() Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetHistory:
    """Test get_history() method behavior."""

    @pytest.mark.asyncio
    async def test_unknown_session_returns_empty_list(self) -> None:
        """get_history() returns empty list for unknown session_id."""
        store = InMemorySessionStore()
        history = await store.get_history("unknown-session")
        assert history == []

    @pytest.mark.asyncio
    async def test_retrieve_saved_history(self) -> None:
        """get_history() returns previously saved messages."""
        store = InMemorySessionStore()
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there")]),
        ]
        await store.save_history("session-1", messages)
        history = await store.get_history("session-1")
        assert len(history) == 2
        assert history[0] == messages[0]
        assert history[1] == messages[1]

    @pytest.mark.asyncio
    async def test_history_ordering_preserved(self) -> None:
        """get_history() returns messages in chronological order."""
        store = InMemorySessionStore()
        messages = [
            ModelRequest(parts=[UserPromptPart(content="First")]),
            ModelResponse(parts=[TextPart(content="Second")]),
            ModelRequest(parts=[UserPromptPart(content="Third")]),
            ModelResponse(parts=[TextPart(content="Fourth")]),
        ]
        await store.save_history("session-1", messages)
        history = await store.get_history("session-1")
        assert len(history) == 4
        assert history[0].parts[0].content == "First"  # type: ignore
        assert history[1].parts[0].content == "Second"  # type: ignore
        assert history[2].parts[0].content == "Third"  # type: ignore
        assert history[3].parts[0].content == "Fourth"  # type: ignore

    @pytest.mark.asyncio
    async def test_get_history_updates_last_access(self) -> None:
        """get_history() updates last access time for TTL tracking."""
        store = InMemorySessionStore(session_ttl=1)
        await store.save_history("session-1", [])
        time.sleep(0.5)
        # Access should update last_access time
        await store.get_history("session-1")
        # Session should not be expired after access
        expired = await store.cleanup_expired_sessions()
        assert expired == 0

    @pytest.mark.asyncio
    async def test_get_history_empty_session_id_raises(self) -> None:
        """get_history() raises ValueError for empty session_id."""
        store = InMemorySessionStore()
        with pytest.raises(ValueError, match="session_id cannot be empty"):
            await store.get_history("")

    @pytest.mark.asyncio
    async def test_get_history_long_session_id_raises(self) -> None:
        """get_history() raises ValueError for session_id > 256 chars."""
        store = InMemorySessionStore()
        long_id = "a" * 257
        with pytest.raises(ValueError, match="session_id too long"):
            await store.get_history(long_id)

    @pytest.mark.asyncio
    async def test_get_history_invalid_characters_raises(self) -> None:
        """get_history() raises ValueError for invalid characters in session_id."""
        store = InMemorySessionStore()
        with pytest.raises(ValueError, match="invalid characters"):
            await store.get_history("session@123")


# ═══════════════════════════════════════════════════════════════════════════════
# save_history() Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveHistory:
    """Test save_history() method behavior."""

    @pytest.mark.asyncio
    async def test_save_empty_history(self) -> None:
        """save_history() accepts empty message list."""
        store = InMemorySessionStore()
        await store.save_history("session-1", [])
        history = await store.get_history("session-1")
        assert history == []

    @pytest.mark.asyncio
    async def test_save_replaces_existing_history(self) -> None:
        """save_history() replaces any existing history for the session."""
        store = InMemorySessionStore()
        # Save initial history
        messages1 = [ModelRequest(parts=[UserPromptPart(content="First")])]
        await store.save_history("session-1", messages1)
        # Replace with new history
        messages2 = [
            ModelRequest(parts=[UserPromptPart(content="New first")]),
            ModelResponse(parts=[TextPart(content="New second")]),
        ]
        await store.save_history("session-1", messages2)
        # Should have new history only
        history = await store.get_history("session-1")
        assert len(history) == 2
        assert history[0].parts[0].content == "New first"  # type: ignore

    @pytest.mark.asyncio
    async def test_save_history_updates_last_access(self) -> None:
        """save_history() updates last access time for TTL tracking."""
        store = InMemorySessionStore(session_ttl=1)
        await store.save_history("session-1", [])
        time.sleep(0.5)
        # Save should update last_access time
        await store.save_history("session-1", [])
        # Session should not be expired after save
        expired = await store.cleanup_expired_sessions()
        assert expired == 0

    @pytest.mark.asyncio
    async def test_save_history_max_messages_limit(self) -> None:
        """save_history() raises ValueError when messages exceed max_messages."""
        store = InMemorySessionStore(max_messages=5)
        messages = [ModelRequest(parts=[UserPromptPart(content=f"Msg {i}")]) for i in range(6)]
        with pytest.raises(ValueError, match="Too many messages"):
            await store.save_history("session-1", messages)

    @pytest.mark.asyncio
    async def test_save_history_non_model_message_raises(self) -> None:
        """save_history() raises TypeError for non-ModelMessage instances (Task 3.14)."""
        store = InMemorySessionStore()
        invalid_messages: list[Any] = [{"role": "user", "content": "Hello"}]
        with pytest.raises(TypeError, match="must be ModelMessage instances"):
            await store.save_history("session-1", invalid_messages)

    @pytest.mark.asyncio
    async def test_save_history_empty_session_id_raises(self) -> None:
        """save_history() raises ValueError for empty session_id."""
        store = InMemorySessionStore()
        with pytest.raises(ValueError, match="session_id cannot be empty"):
            await store.save_history("", [])

    @pytest.mark.asyncio
    async def test_save_history_long_session_id_raises(self) -> None:
        """save_history() raises ValueError for session_id > 256 chars."""
        store = InMemorySessionStore()
        long_id = "a" * 257
        with pytest.raises(ValueError, match="session_id too long"):
            await store.save_history(long_id, [])

    @pytest.mark.asyncio
    async def test_save_history_invalid_characters_raises(self) -> None:
        """save_history() raises ValueError for invalid characters in session_id."""
        store = InMemorySessionStore()
        with pytest.raises(ValueError, match="invalid characters"):
            await store.save_history("session/123", [])


# ═══════════════════════════════════════════════════════════════════════════════
# clear() Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestClear:
    """Test clear() method behavior."""

    @pytest.mark.asyncio
    async def test_clear_removes_history(self) -> None:
        """clear() removes all message history for a session."""
        store = InMemorySessionStore()
        messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        await store.save_history("session-1", messages)
        await store.clear("session-1")
        history = await store.get_history("session-1")
        assert history == []

    @pytest.mark.asyncio
    async def test_clear_removes_last_access(self) -> None:
        """clear() removes last_access entry to prevent memory leak (Task 3.12)."""
        store = InMemorySessionStore()
        await store.save_history("session-1", [])
        # Verify last_access exists
        assert "session-1" in store._last_access
        await store.clear("session-1")
        # Verify last_access removed
        assert "session-1" not in store._last_access

    @pytest.mark.asyncio
    async def test_clear_removes_lock(self) -> None:
        """clear() removes lock entry to prevent memory leak."""
        store = InMemorySessionStore()
        await store.save_history("session-1", [])
        # Lock should exist after save
        assert "session-1" in store._locks
        await store.clear("session-1")
        # Lock should be removed after clear
        assert "session-1" not in store._locks

    @pytest.mark.asyncio
    async def test_clear_unknown_session_no_error(self) -> None:
        """clear() does not raise error for non-existent session."""
        store = InMemorySessionStore()
        await store.clear("unknown-session")  # Should not raise

    @pytest.mark.asyncio
    async def test_clear_empty_session_id_raises(self) -> None:
        """clear() raises ValueError for empty session_id."""
        store = InMemorySessionStore()
        with pytest.raises(ValueError, match="session_id cannot be empty"):
            await store.clear("")

    @pytest.mark.asyncio
    async def test_clear_long_session_id_raises(self) -> None:
        """clear() raises ValueError for session_id > 256 chars."""
        store = InMemorySessionStore()
        long_id = "a" * 257
        with pytest.raises(ValueError, match="session_id too long"):
            await store.clear(long_id)

    @pytest.mark.asyncio
    async def test_clear_invalid_characters_raises(self) -> None:
        """clear() raises ValueError for invalid characters in session_id."""
        store = InMemorySessionStore()
        with pytest.raises(ValueError, match="invalid characters"):
            await store.clear("session!123")


# ═══════════════════════════════════════════════════════════════════════════════
# Session ID Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionIdValidation:
    """Test session_id validation across all methods."""

    @pytest.mark.asyncio
    async def test_valid_alphanumeric_session_id(self) -> None:
        """Valid alphanumeric session_id is accepted."""
        store = InMemorySessionStore()
        await store.save_history("session123", [])
        history = await store.get_history("session123")
        assert history == []

    @pytest.mark.asyncio
    async def test_valid_session_id_with_underscore(self) -> None:
        """Valid session_id with underscore is accepted."""
        store = InMemorySessionStore()
        await store.save_history("session_123", [])
        history = await store.get_history("session_123")
        assert history == []

    @pytest.mark.asyncio
    async def test_valid_session_id_with_hyphen(self) -> None:
        """Valid session_id with hyphen is accepted."""
        store = InMemorySessionStore()
        await store.save_history("session-123", [])
        history = await store.get_history("session-123")
        assert history == []

    @pytest.mark.asyncio
    async def test_valid_mixed_case_session_id(self) -> None:
        """Valid mixed-case session_id is accepted."""
        store = InMemorySessionStore()
        await store.save_history("SessionABC123", [])
        history = await store.get_history("SessionABC123")
        assert history == []

    @pytest.mark.asyncio
    async def test_session_id_max_length_accepted(self) -> None:
        """session_id with exactly 256 chars is accepted."""
        store = InMemorySessionStore()
        valid_id = "a" * 256
        await store.save_history(valid_id, [])
        history = await store.get_history(valid_id)
        assert history == []


# Test message validation, TTL, LRU, concurrency, and edge cases
class TestMessageValidation:
    """Test message validation in save_history()."""

    @pytest.mark.asyncio
    async def test_valid_model_request(self) -> None:
        """ModelRequest instances are accepted."""
        store = InMemorySessionStore()
        messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        await store.save_history("session-1", messages)
        assert len(await store.get_history("session-1")) == 1

    @pytest.mark.asyncio
    async def test_strict_isinstance_check(self) -> None:
        """save_history() uses strict isinstance check (Task 3.14)."""
        store = InMemorySessionStore()

        class FakeMessage:
            def __init__(self) -> None:
                self.parts = []

        invalid_messages: list[Any] = [FakeMessage()]
        with pytest.raises(TypeError, match="must be ModelMessage instances"):
            await store.save_history("session-1", invalid_messages)


class TestTTLAndCleanup:
    """Test TTL-based session lifecycle management (Task 3.11)."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired_sessions(self) -> None:
        """cleanup_expired_sessions() removes sessions older than TTL."""
        store = InMemorySessionStore(session_ttl=1)
        await store.save_history("session-1", [])
        time.sleep(1.1)
        expired = await store.cleanup_expired_sessions()
        assert expired == 1
        assert await store.get_history("session-1") == []

    @pytest.mark.asyncio
    async def test_cleanup_is_public_method(self) -> None:
        """cleanup_expired_sessions() is public for external calling (Task 3.15)."""
        store = InMemorySessionStore()
        assert hasattr(store, "cleanup_expired_sessions")
        assert not store.cleanup_expired_sessions.__name__.startswith("_")


class TestLRUEviction:
    """Test LRU eviction when max_sessions limit is exceeded (Task 3.16)."""

    @pytest.mark.asyncio
    async def test_lru_eviction_when_limit_exceeded(self) -> None:
        """LRU session is evicted when max_sessions limit is exceeded."""
        store = InMemorySessionStore(max_sessions=2)
        await store.save_history("session-1", [])
        time.sleep(0.1)
        await store.save_history("session-2", [])
        time.sleep(0.1)
        await store.save_history("session-3", [])
        # session-1 should be evicted
        assert await store.get_history("session-1") == []


class TestConcurrency:
    """Test per-session locking for concurrent operations (Task 3.5)."""

    @pytest.mark.asyncio
    async def test_concurrent_access_different_sessions(self) -> None:
        """Concurrent operations on different sessions do not block each other."""
        store = InMemorySessionStore()

        async def save(sid: str) -> None:
            await store.save_history(sid, [ModelRequest(parts=[UserPromptPart(content="Test")])])

        await asyncio.gather(save("s1"), save("s2"), save("s3"))
        assert len(await store.get_history("s1")) == 1
        assert len(await store.get_history("s2")) == 1
        assert len(await store.get_history("s3")) == 1

    @pytest.mark.asyncio
    async def test_lock_created_on_first_access(self) -> None:
        """Lock is created lazily on first access via setdefault."""
        store = InMemorySessionStore()
        assert "session-1" not in store._locks
        await store.save_history("session-1", [])
        assert "session-1" in store._locks
