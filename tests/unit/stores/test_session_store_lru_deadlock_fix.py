"""Tests for LRU eviction deadlock bug fix.

HIGH FIX: LRU eviction logic can cause deadlock when two concurrent
save_history() calls try to evict each other as LRU victims.

The deadlock scenario:
- Thread A: save_history("session_1") holds lock for session_1,
  finds session_2 as LRU victim, tries to acquire lock for session_2
- Thread B: save_history("session_2") holds lock for session_2,
  finds session_1 as LRU victim, tries to acquire lock for session_1
- DEADLOCK: A waits for B's lock, B waits for A's lock

The fix: Don't hold current session's lock while trying to evict victim.
Perform eviction AFTER releasing current session lock.
"""

import asyncio

import pytest
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore


class TestLRUEvictionDeadlock:
    """Test that LRU eviction doesn't cause deadlocks."""

    @pytest.mark.asyncio
    async def test_concurrent_saves_dont_deadlock_on_eviction(self) -> None:
        """Concurrent save_history calls should not deadlock when triggering LRU eviction.

        This test creates a scenario where:
        1. Store has max_sessions=2
        2. Two sessions already exist (session_0, session_1)
        3. Two new concurrent saves (session_2, session_3) both trigger eviction
        4. Each might try to evict the other
        5. Without proper locking, this causes deadlock
        6. With the fix, eviction happens after releasing session lock, preventing deadlock
        """
        # Create store with very low max_sessions to easily trigger eviction
        store = InMemorySessionStore(
            max_messages=10,
            session_ttl=3600,
            max_sessions=2,  # Very low limit to force eviction
        )

        # Pre-populate with 2 sessions (at max capacity)
        msg1 = ModelRequest(parts=[UserPromptPart(content="test 1")])
        msg2 = ModelRequest(parts=[UserPromptPart(content="test 2")])

        await store.save_history("session_0", [msg1])
        await asyncio.sleep(0.01)  # Small delay to ensure different timestamps
        await store.save_history("session_1", [msg2])

        # Now store is at capacity (2 sessions)
        assert len(store._store) == 2

        # Create messages for new sessions
        msg3 = ModelRequest(parts=[UserPromptPart(content="test 3")])
        msg4 = ModelRequest(parts=[UserPromptPart(content="test 4")])

        # Simulate concurrent saves that will both trigger eviction
        # Each will try to evict something, potentially causing deadlock
        # if the implementation holds session lock while acquiring victim lock

        async def save_with_delay(session_id: str, msg: ModelRequest) -> None:
            """Save with small delay to increase chance of race condition."""
            await asyncio.sleep(0.001)
            await store.save_history(session_id, [msg])

        # Run two concurrent saves - both will exceed max_sessions
        # Without the fix, this can deadlock if they try to evict each other
        tasks = [
            save_with_delay("session_2", msg3),
            save_with_delay("session_3", msg4),
        ]

        # Use wait_for with timeout to detect deadlock
        # If deadlock occurs, this will raise asyncio.TimeoutError
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=2.0,  # Generous timeout - should complete quickly if no deadlock
            )
        except TimeoutError:
            pytest.fail("Deadlock detected: concurrent save_history calls timed out")

        # After concurrent saves, store should still be at max_sessions
        # (eviction should have happened)
        assert len(store._store) <= store.max_sessions

        # At least one of the new sessions should exist
        assert "session_2" in store._store or "session_3" in store._store

    @pytest.mark.asyncio
    async def test_rapid_concurrent_saves_with_eviction(self) -> None:
        """Rapid concurrent saves with eviction should not deadlock.

        This is a stress test with more concurrent operations.
        """
        store = InMemorySessionStore(
            max_messages=10,
            session_ttl=3600,
            max_sessions=5,  # Low limit
        )

        # Pre-fill to capacity
        msg = ModelRequest(parts=[UserPromptPart(content="test")])
        for i in range(5):
            await store.save_history(f"initial_{i}", [msg])
            await asyncio.sleep(0.001)  # Stagger timestamps

        # Now launch many concurrent saves, all will trigger eviction
        async def rapid_save(session_id: str) -> None:
            await store.save_history(session_id, [msg])

        tasks = [rapid_save(f"new_{i}") for i in range(10)]

        # This should complete without deadlock
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=3.0,
            )
        except TimeoutError:
            pytest.fail("Deadlock detected in rapid concurrent operations")

        # Store should be at max capacity
        assert len(store._store) <= store.max_sessions
