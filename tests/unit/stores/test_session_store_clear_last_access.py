"""Unit tests for _last_access memory leak fix in InMemorySessionStore.clear()."""

import asyncio

import pytest
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore


class TestSessionStoreClearLastAccessFix:
    """Test suite for Fix _last_access memory leak in clear()."""

    @pytest.fixture
    def store(self) -> InMemorySessionStore:
        """Create an InMemorySessionStore instance with short TTL for testing."""
        return InMemorySessionStore(session_ttl=2)

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelRequest(parts=[UserPromptPart(content="World")]),
        ]

    @pytest.mark.asyncio
    async def test_last_access_removed_after_clear(self, store, sample_messages) -> None:
        """_last_access must be removed when clear() is called on an active session.

        This test verifies that calling clear() removes the entry from _last_access
        dict to prevent memory leak in long-running services.
        """
        session_id = "test-session-clear"

        # Save history - this should populate _last_access
        await store.save_history(session_id, sample_messages)

        # Verify _last_access contains the session
        assert session_id in store._last_access, (
            "_last_access should contain session before clear()"
        )

        # Clear the session
        await store.clear(session_id)

        # Verify _last_access no longer contains the session
        assert session_id not in store._last_access, (
            "_last_access should be removed after clear() to prevent memory leak"
        )

    @pytest.mark.asyncio
    async def test_last_access_removed_after_cleanup_expired_sessions(
        self, store, sample_messages
    ) -> None:
        """_last_access must be removed when _cleanup_expired_sessions() removes an expired session.

        This test verifies that the cleanup process correctly delegates to clear()
        which should remove _last_access entries.
        """
        session_id = "test-session-expire"

        # Save history - this should populate _last_access
        await store.save_history(session_id, sample_messages)

        # Verify _last_access contains the session
        assert session_id in store._last_access, (
            "_last_access should contain session before cleanup"
        )

        # Wait for session to expire (TTL is 2 seconds)
        await asyncio.sleep(2.5)

        # Run cleanup
        removed_count = await store.cleanup_expired_sessions()

        # Verify cleanup removed the session
        assert removed_count == 1, "Cleanup should have removed one expired session"

        # Verify _last_access no longer contains the session
        assert session_id not in store._last_access, (
            "_last_access should be removed after cleanup removes expired session"
        )

    @pytest.mark.asyncio
    async def test_clear_non_existent_session_does_not_raise(self, store) -> None:
        """Clearing a non-existent session should not raise an error.

        This test verifies that clear() handles non-existent sessions gracefully
        and does not cause KeyError when trying to pop from _last_access.
        """
        session_id = "non-existent-session"

        # Clear a session that doesn't exist - should not raise
        await store.clear(session_id)

        # Verify _last_access does not contain the session
        assert session_id not in store._last_access
