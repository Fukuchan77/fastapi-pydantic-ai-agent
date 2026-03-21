"""Unit tests for InMemorySessionStore clear() race condition fix and lock cleanup."""

import asyncio

import pytest
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore


class TestSessionStoreClearRaceCondition:
    """Test that clear() is properly synchronized and cleans up locks."""

    @pytest.fixture
    def store(self) -> InMemorySessionStore:
        """Provide a fresh InMemorySessionStore instance."""
        return InMemorySessionStore()

    @pytest.mark.asyncio
    async def test_clear_concurrent_with_save_no_corruption(
        self, store: InMemorySessionStore
    ) -> None:
        """Concurrent clear() and save_history() must not corrupt data.

        This test verifies that locking in clear() prevents race conditions
        when clear and save operations happen concurrently on the same session.
        """
        session_id = "race-test-session"
        num_operations = 30

        # Create sample messages for saving
        sample_messages = [
            ModelRequest(parts=[UserPromptPart(content="Test message")]),
            ModelResponse(parts=[TextPart(content="Test response")]),
        ]

        async def save_operation() -> None:
            """Save history with a small delay to force concurrent execution."""
            await asyncio.sleep(0.001)
            await store.save_history(session_id, sample_messages)

        async def clear_operation() -> None:
            """Clear history with a small delay to force concurrent execution."""
            await asyncio.sleep(0.001)
            await store.clear(session_id)

        # Interleave save and clear operations
        tasks = []
        for i in range(num_operations):
            if i % 2 == 0:
                tasks.append(save_operation())
            else:
                tasks.append(clear_operation())

        # Execute all operations concurrently
        await asyncio.gather(*tasks)

        # After all operations, the store should be in a valid state
        # (either empty or containing valid messages, but not corrupted)
        final_history = await store.get_history(session_id)

        # Valid states: either empty (clear won) or contains valid messages (save won)
        assert final_history == [] or len(final_history) == 2
        if len(final_history) == 2:
            # If not empty, must be valid complete messages
            assert isinstance(final_history[0], ModelRequest)
            assert isinstance(final_history[1], ModelResponse)

    @pytest.mark.asyncio
    async def test_clear_removes_lock_entry(self, store: InMemorySessionStore) -> None:
        """clear() must remove the lock entry from _locks dict to prevent memory leak.

        This test verifies that after clearing a session, the lock for that session
        is also removed from the _locks dictionary, preventing unbounded memory growth.
        """
        session_id = "memory-leak-test"

        # Save some history (this will create a lock entry)
        messages = [ModelRequest(parts=[UserPromptPart(content="Test")])]
        await store.save_history(session_id, messages)

        # Verify lock was created (internal implementation detail but necessary to test)
        assert session_id in store._locks, "Lock should exist after save_history"

        # Clear the session
        await store.clear(session_id)

        # Verify lock was removed
        assert session_id not in store._locks, (
            "Lock should be removed after clear() to prevent memory leak"
        )

    @pytest.mark.asyncio
    async def test_clear_removes_lock_even_for_nonexistent_session(
        self, store: InMemorySessionStore
    ) -> None:
        """clear() must clean up lock even when clearing a non-existent session.

        This ensures that even if a session never had data, calling clear()
        won't leave a dangling lock entry.
        """
        session_id = "never-existed"

        # Clear a session that was never created
        await store.clear(session_id)

        # Verify no lock entry was left behind
        assert session_id not in store._locks, (
            "No lock should remain after clearing non-existent session"
        )

    @pytest.mark.asyncio
    async def test_multiple_clears_no_lock_accumulation(self, store: InMemorySessionStore) -> None:
        """Multiple clear() calls must not accumulate locks in memory.

        This test verifies that repeatedly clearing the same session doesn't
        cause _locks dict to grow unboundedly.
        """
        session_id = "repeated-clear-test"

        # Save, clear, save, clear multiple times
        messages = [ModelRequest(parts=[UserPromptPart(content="Test")])]

        for _ in range(10):
            await store.save_history(session_id, messages)
            await store.clear(session_id)

        # After all operations, no lock should remain
        assert session_id not in store._locks, (
            "No lock should remain after repeated clear operations"
        )

        # Verify _locks dict hasn't grown
        assert len(store._locks) == 0, "_locks dict should be empty after all clears"
