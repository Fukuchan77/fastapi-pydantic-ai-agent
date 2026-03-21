"""Unit tests for InMemorySessionStore TTL (Time-To-Live) functionality."""

import asyncio

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore


class TestInMemorySessionStoreTTL:
    """Test TTL-based session lifecycle management in InMemorySessionStore."""

    @pytest.fixture
    def store_with_short_ttl(self) -> InMemorySessionStore:
        """Provide an InMemorySessionStore instance with a short TTL for testing."""
        # Use 1 second TTL for fast tests
        return InMemorySessionStore(session_ttl=1)

    @pytest.fixture
    def sample_messages(self) -> list[ModelMessage]:
        """Provide sample ModelMessage instances for testing."""
        return [
            ModelRequest(parts=[UserPromptPart(content="Test message")]),
        ]

    @pytest.mark.asyncio
    async def test_last_access_updated_on_get_history(
        self, store_with_short_ttl: InMemorySessionStore, sample_messages: list[ModelMessage]
    ) -> None:
        """_last_access must be updated when get_history is called."""
        session_id = "test-session-get"

        # Save initial history
        await store_with_short_ttl.save_history(session_id, sample_messages)

        # Get the initial last_access time
        initial_time = store_with_short_ttl._last_access.get(session_id)
        assert initial_time is not None, "_last_access should be set after save_history"

        # Wait a bit and access again
        await asyncio.sleep(0.1)
        await store_with_short_ttl.get_history(session_id)

        # Verify _last_access was updated
        updated_time = store_with_short_ttl._last_access.get(session_id)
        assert updated_time is not None
        assert updated_time > initial_time, "_last_access should be updated on get_history"

    @pytest.mark.asyncio
    async def test_last_access_updated_on_save_history(
        self, store_with_short_ttl: InMemorySessionStore, sample_messages: list[ModelMessage]
    ) -> None:
        """_last_access must be updated when save_history is called."""
        session_id = "test-session-save"

        # Save initial history
        await store_with_short_ttl.save_history(session_id, sample_messages)
        initial_time = store_with_short_ttl._last_access.get(session_id)
        assert initial_time is not None

        # Wait a bit and save again
        await asyncio.sleep(0.1)
        await store_with_short_ttl.save_history(session_id, sample_messages)

        # Verify _last_access was updated
        updated_time = store_with_short_ttl._last_access.get(session_id)
        assert updated_time is not None
        assert updated_time > initial_time, "_last_access should be updated on save_history"

    @pytest.mark.asyncio
    async def test_expired_session_removed_after_cleanup(
        self, store_with_short_ttl: InMemorySessionStore, sample_messages: list[ModelMessage]
    ) -> None:
        """Expired sessions must be removed when cleanup is called."""
        session_id = "expired-session"

        # Save a session
        await store_with_short_ttl.save_history(session_id, sample_messages)

        # Verify session exists
        history = await store_with_short_ttl.get_history(session_id)
        assert len(history) == 1

        # Wait for session to expire (TTL is 1 second)
        await asyncio.sleep(1.2)

        # Run cleanup
        removed_count = await store_with_short_ttl.cleanup_expired_sessions()

        # Verify expired session was removed
        assert removed_count == 1, "One expired session should have been removed"
        history_after = await store_with_short_ttl.get_history(session_id)
        assert history_after == [], "Expired session should return empty history"

    @pytest.mark.asyncio
    async def test_non_expired_session_kept_after_cleanup(
        self, store_with_short_ttl: InMemorySessionStore, sample_messages: list[ModelMessage]
    ) -> None:
        """Non-expired sessions must be kept when cleanup is called."""
        session_id = "active-session"

        # Save a session
        await store_with_short_ttl.save_history(session_id, sample_messages)

        # Wait less than TTL
        await asyncio.sleep(0.3)

        # Run cleanup
        removed_count = await store_with_short_ttl.cleanup_expired_sessions()

        # Verify active session was NOT removed
        assert removed_count == 0, "No sessions should have been removed"
        history = await store_with_short_ttl.get_history(session_id)
        assert len(history) == 1, "Active session should still exist"

    @pytest.mark.asyncio
    async def test_cleanup_returns_correct_count(
        self, store_with_short_ttl: InMemorySessionStore, sample_messages: list[ModelMessage]
    ) -> None:
        """Cleanup must return the correct count of removed sessions."""
        # Create multiple sessions
        session_ids = ["session-1", "session-2", "session-3"]

        for session_id in session_ids:
            await store_with_short_ttl.save_history(session_id, sample_messages)

        # Wait for all sessions to expire
        await asyncio.sleep(1.2)

        # Run cleanup
        removed_count = await store_with_short_ttl.cleanup_expired_sessions()

        # Verify all sessions were removed
        assert removed_count == 3, "All three expired sessions should have been removed"

        # Verify all sessions are gone
        for session_id in session_ids:
            history = await store_with_short_ttl.get_history(session_id)
            assert history == [], f"Session {session_id} should be gone"

    @pytest.mark.asyncio
    async def test_mixed_expired_and_active_sessions(
        self, store_with_short_ttl: InMemorySessionStore, sample_messages: list[ModelMessage]
    ) -> None:
        """Cleanup must remove only expired sessions, keeping active ones."""
        expired_session = "expired-session"
        active_session = "active-session"

        # Create expired session
        await store_with_short_ttl.save_history(expired_session, sample_messages)

        # Wait for first session to expire
        await asyncio.sleep(1.2)

        # Create active session (after first one expires)
        await store_with_short_ttl.save_history(active_session, sample_messages)

        # Run cleanup
        removed_count = await store_with_short_ttl.cleanup_expired_sessions()

        # Verify only expired session was removed
        assert removed_count == 1, "Only one expired session should have been removed"

        expired_history = await store_with_short_ttl.get_history(expired_session)
        assert expired_history == [], "Expired session should be gone"

        active_history = await store_with_short_ttl.get_history(active_session)
        assert len(active_history) == 1, "Active session should still exist"

    @pytest.mark.asyncio
    async def test_cleanup_with_no_expired_sessions(
        self, store_with_short_ttl: InMemorySessionStore
    ) -> None:
        """Cleanup should return 0 when no sessions are expired."""
        # Run cleanup on empty store
        removed_count = await store_with_short_ttl.cleanup_expired_sessions()

        # Verify no sessions were removed
        assert removed_count == 0, "No sessions should have been removed from empty store"

    @pytest.mark.asyncio
    async def test_session_ttl_default_value(self) -> None:
        """InMemorySessionStore should have a default TTL of 3600 seconds."""
        store = InMemorySessionStore()

        # Verify default TTL
        assert store.session_ttl == 3600, "Default session_ttl should be 3600 seconds"

    @pytest.mark.asyncio
    async def test_session_ttl_custom_value(self) -> None:
        """InMemorySessionStore should accept custom TTL value."""
        custom_ttl = 7200
        store = InMemorySessionStore(session_ttl=custom_ttl)

        # Verify custom TTL
        assert store.session_ttl == custom_ttl, f"session_ttl should be {custom_ttl}"
