"""Test for LRU eviction lock acquisition during concurrent operations.

This test verifies that when LRU eviction occurs, the evicted session's lock
is properly acquired to prevent race conditions where a concurrent save_history
operation could resurrect the evicted session.
"""

import asyncio

import pytest
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore


@pytest.mark.asyncio
async def test_lru_eviction_acquires_victim_session_lock() -> None:
    """Test that LRU eviction acquires the victim session's lock.

    This test simulates a race condition where:
    1. save_history(session_A) triggers LRU eviction of session_B
    2. Concurrent save_history(session_B) is trying to save
    3. Without proper locking, session_B could be resurrected after eviction

    Expected behavior: Session B should not reappear after eviction.
    """
    # Create store with max_sessions=2 to trigger quick eviction
    store = InMemorySessionStore(max_sessions=2, session_ttl=3600)

    # Create initial sessions to fill capacity
    msg1 = ModelRequest(parts=[UserPromptPart(content="test message 1")])
    msg2 = ModelRequest(parts=[UserPromptPart(content="test message 2")])

    await store.save_history("session_1", [msg1])
    await store.save_history("session_2", [msg2])

    # Verify both sessions exist
    assert len(store._store) == 2
    assert "session_1" in store._store
    assert "session_2" in store._store

    # Now we'll create a scenario where:
    # - session_3 save triggers eviction of session_1 (LRU)
    # - But session_1 has a concurrent save operation

    # Simulate concurrent operations
    msg3 = ModelRequest(parts=[UserPromptPart(content="test message 3")])
    msg1_updated = ModelRequest(parts=[UserPromptPart(content="updated message 1")])

    # Create a flag to coordinate timing
    eviction_started = asyncio.Event()

    async def save_session1_with_delay() -> None:
        """Simulate slow save_history for session_1."""
        # Acquire session_1 lock
        async with store._locks.setdefault("session_1", asyncio.Lock()):
            # Wait for eviction to be triggered by session_3
            await eviction_started.wait()
            # Small delay to let eviction attempt to proceed
            await asyncio.sleep(0.1)
            # Now try to save (should be blocked if eviction holds the lock)
            store._last_access["session_1"] = asyncio.get_event_loop().time()
            store._store["session_1"] = [msg1_updated]

    async def save_session3_triggers_eviction() -> None:
        """Save session_3 which should evict session_1 (LRU)."""
        # Give session_1 time to acquire its lock
        await asyncio.sleep(0.05)
        eviction_started.set()
        # This should trigger eviction of session_1
        await store.save_history("session_3", [msg3])

    # Run both operations concurrently
    await asyncio.gather(
        save_session1_with_delay(),
        save_session3_triggers_eviction(),
    )

    # After eviction, session_1 should NOT be in the store
    # because the fix ensures we acquire session_1's lock before evicting it
    assert len(store._store) == 2
    assert "session_1" not in store._store, "Session 1 should be evicted and not resurrected"
    assert "session_2" in store._store
    assert "session_3" in store._store


@pytest.mark.asyncio
async def test_lru_eviction_order_respects_last_access() -> None:
    """Test that LRU eviction evicts the least recently accessed session."""
    store = InMemorySessionStore(max_sessions=2, session_ttl=3600)

    msg = ModelRequest(parts=[UserPromptPart(content="test")])

    # Create sessions with known access order
    await store.save_history("session_1", [msg])
    await asyncio.sleep(0.01)
    await store.save_history("session_2", [msg])

    # Access session_1 again to make it more recent
    await asyncio.sleep(0.01)
    await store.get_history("session_1")

    # Now session_2 should be LRU
    # Adding session_3 should evict session_2
    await store.save_history("session_3", [msg])

    assert len(store._store) == 2
    assert "session_1" in store._store, "Session 1 should remain (more recent)"
    assert "session_2" not in store._store, "Session 2 should be evicted (LRU)"
    assert "session_3" in store._store
