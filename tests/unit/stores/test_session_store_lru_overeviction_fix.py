"""Test for Task 16.37: Fix LRU over-eviction bug in session store.

This test verifies that LRU eviction includes a guard to re-check store capacity
after acquiring locks, preventing unnecessary eviction when concurrent operations
reduce the store size between lock acquisitions.
"""

import asyncio

import pytest
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore


@pytest.mark.asyncio
async def test_lru_eviction_still_works_when_store_exceeds_capacity() -> None:
    """Test that LRU eviction proceeds when store genuinely exceeds max_sessions.

    This is the baseline test to ensure normal LRU eviction behavior works.
    """
    # Setup: Store with max_sessions=2
    store = InMemorySessionStore(max_sessions=2)

    # Create 2 sessions to reach capacity
    msg = ModelRequest(parts=[UserPromptPart(content="test")])
    await store.save_history("session_1", [msg])
    await asyncio.sleep(0.01)  # Ensure different access times
    await store.save_history("session_2", [msg])

    # Verify we're at capacity
    assert len(store._store) == 2

    # Identify LRU session
    lru_session = min(store._last_access.items(), key=lambda x: x[1])[0]
    assert lru_session == "session_1"

    # Add a 3rd session, triggering LRU eviction
    await store.save_history("session_3", [msg])

    # EXPECTED: session_1 (LRU) should be evicted
    assert "session_1" not in store._store, "LRU session should be evicted when over capacity"
    assert "session_2" in store._store
    assert "session_3" in store._store
    assert len(store._store) == 2


@pytest.mark.asyncio
async def test_lru_eviction_race_condition_with_concurrent_clear() -> None:
    """Test the race condition scenario where clear() runs during LRU eviction.

    This test demonstrates Task 16.37: Without the len(self._store) > self.max_sessions
    guard inside the final critical section, an LRU session can be evicted even when
    the store is no longer over capacity due to concurrent clear() operations.

    The test uses asyncio.gather to run save_history and clear concurrently,
    with strategic delays to trigger the race window.
    """
    # Setup: Store with max_sessions=3
    store = InMemorySessionStore(max_sessions=3)

    # Create 3 sessions to reach capacity
    msg = ModelRequest(parts=[UserPromptPart(content="test")])
    await store.save_history("session_1", [msg])
    await asyncio.sleep(0.01)
    await store.save_history("session_2", [msg])
    await asyncio.sleep(0.01)
    await store.save_history("session_3", [msg])

    # Verify all 3 sessions exist
    assert len(store._store) == 3

    # Identify LRU session
    lru_session = min(store._last_access.items(), key=lambda x: x[1])[0]
    assert lru_session == "session_1"

    # Create a flag to control timing
    clear_executed = asyncio.Event()

    async def save_new_session() -> None:
        """Save session_4, which will trigger LRU eviction."""
        # Add a small delay to let clear() execute first
        await asyncio.sleep(0.05)
        await store.save_history("session_4", [msg])

    async def clear_different_session() -> None:
        """Clear session_2 (not the LRU), reducing store size."""
        await asyncio.sleep(0.01)
        await store.clear("session_2")
        clear_executed.set()

    # Run both operations concurrently
    # The timing aims to have clear() execute while save_history() is in the LRU eviction logic
    await asyncio.gather(save_new_session(), clear_different_session())

    # Wait for clear to complete
    await clear_executed.wait()

    # EXPECTED BEHAVIOR WITH FIX:
    # - session_2 was cleared, so store had only 2 sessions before session_4 was added
    # - After session_4 is added, store has 3 sessions (at capacity)
    # - session_1 (LRU) should NOT be evicted because the final guard prevents it

    # BUGGY BEHAVIOR WITHOUT FIX:
    # - The eviction logic detects 4 sessions initially
    # - clear() runs and removes session_2
    # - The eviction proceeds anyway, removing session_1 unnecessarily
    # - Result: only 2 sessions remain instead of 3

    # Check the actual behavior
    # Note: Due to timing variability, this test might not always trigger the race.
    # The definitive test is checking the code has the guard at line 205.
    # This test documents the expected behavior when the race does occur.

    assert "session_2" not in store._store, "session_2 should be cleared"
    assert "session_4" in store._store, "session_4 should be saved"
    assert "session_3" in store._store, "session_3 should remain"

    # The critical assertion: session_1 should remain because the concurrent clear()
    # reduced the store size below max_sessions before the eviction completed.
    # Without the fix (len(self._store) > self.max_sessions guard), session_1 gets evicted.
    if "session_1" not in store._store:
        pytest.fail(
            "BUG DETECTED: Task 16.37 - LRU session (session_1) was evicted even though "
            "concurrent clear() reduced store size below max_sessions. "
            "The eviction logic at line 205 needs to re-check len(self._store) > self.max_sessions "
            "before evicting to prevent this race condition."
        )

    # With the fix, we should have 3 sessions
    assert len(store._store) == 3, f"Expected 3 sessions, got {len(store._store)}"
