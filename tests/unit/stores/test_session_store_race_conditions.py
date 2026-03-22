"""Tests for critical race conditions in session_store.py.

This module tests the three critical issues identified in the code review:
1. LRU eviction race condition (line 157)
2. Lock cleanup race condition (line 187)
3. SSE  prefix consistency
"""

import asyncio

import pytest
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore


class TestLRUEvictionRaceCondition:
    """Test LRU eviction atomicity (issue at line 157)."""

    @pytest.mark.asyncio
    async def test_lru_eviction_with_concurrent_access(self):
        """Test that LRU eviction doesn't cause race conditions with concurrent operations.

        Issue: Lines 157-161 perform LRU eviction AFTER releasing the lock from save_history.
        Between the lock release and the eviction check, another operation could modify
        _store or _last_access, causing inconsistent state.

        Expected behavior: The eviction should be atomic with the save operation.
        """
        # Create store with capacity for only 2 sessions
        store = InMemorySessionStore(max_sessions=2)

        # Create messages for testing
        msg1 = ModelRequest(parts=[UserPromptPart(content="test1")])
        msg2 = ModelRequest(parts=[UserPromptPart(content="test2")])
        msg3 = ModelRequest(parts=[UserPromptPart(content="test3")])

        # Fill up to capacity
        await store.save_history("session1", [msg1])
        await store.save_history("session2", [msg2])

        # Verify both sessions exist
        assert len(store._store) == 2
        assert len(store._last_access) == 2

        # Now save a third session, which should trigger LRU eviction
        # Race condition: Between saving and checking max_sessions,
        # another operation could interfere
        async def concurrent_save():
            await asyncio.sleep(0.001)  # Small delay to interleave
            await store.save_history("session4", [msg1])

        # Start concurrent operation
        task = asyncio.create_task(concurrent_save())

        # Save third session (should evict session1 as LRU)
        await store.save_history("session3", [msg3])

        # Wait for concurrent operation
        await task

        # After both operations:
        # - Total sessions should not exceed max_sessions (2)
        # - _store and _last_access should have consistent keys
        assert len(store._store) <= 2, f"Store has {len(store._store)} sessions, expected <= 2"
        assert len(store._last_access) <= 2, (
            f"Last access has {len(store._last_access)} entries, expected <= 2"
        )
        assert set(store._store.keys()) == set(store._last_access.keys()), (
            "Store and last_access keys are inconsistent"
        )


class TestLockCleanupRaceCondition:
    """Test lock cleanup atomicity (issue at line 187)."""

    @pytest.mark.asyncio
    async def test_lock_cleanup_with_concurrent_operations(self):
        """Test that lock cleanup doesn't interfere with concurrent operations.

        Issue: Line 187 calls _locks.pop(session_id, None) AFTER releasing the lock.
        Between lines 182 (lock release) and 187 (lock removal), another operation
        could call setdefault to create a new lock, which then gets removed while
        still in use.

        Expected behavior: Lock removal should be safe and not cause deadlock.
        FIX: After fix, clear() uses store_lock to prevent removing a lock that's
        in use, but it's OK if get_history() creates a new lock after clear() completes.
        """
        store = InMemorySessionStore()
        msg = ModelRequest(parts=[UserPromptPart(content="test")])

        # Save initial history
        await store.save_history("session1", [msg])

        # Verify lock exists and session has data
        assert "session1" in store._locks
        assert "session1" in store._store

        # Clear with concurrent get_history on same session
        async def concurrent_get():
            # This should acquire a lock via setdefault during/after the clear operation
            await asyncio.sleep(0.001)  # Small delay to interleave with clear
            history = await store.get_history("session1")
            return history

        task = asyncio.create_task(concurrent_get())

        # Clear the session (which removes data and lock under store_lock)
        await store.clear("session1")

        # The concurrent get should complete without error or deadlock
        result = await task
        assert isinstance(result, list)
        assert len(result) == 0, "Session should be cleared"

        # After clear, session data should be removed
        assert "session1" not in store._store, "Session data should be removed"

        # Note: Lock may or may not exist depending on timing - get_history()
        # might create a new lock after clear() completes. This is OK and doesn't
        # indicate a bug. The important thing is no deadlock or corruption occurred.


class TestSSEDataPrefixConsistency:
    """Test SSE  prefix consistency (issue at line 99)."""

    def test_format_event_with_serialization_error_has_data_prefix(self):
        r"""Test that error events from serialization failures have 'data: ' prefix.

        Issue: Line 99 in app/api/v1/agent.py returns formatted string without 'data: ' prefix
        when JSON serialization fails. It returns:
            f"data: {json.dumps(error_payload)}\\n\\n"

        But should return:
            f"data: {json.dumps(error_payload)}\\n\\n"

        Expected behavior: All SSE events must have 'data: ' prefix for proper parsing.
        """
        import json

        from app.api.v1.agent import DefaultSSEAdapter

        adapter = DefaultSSEAdapter()

        # Create an object that json.dumps cannot serialize (set is not JSON serializable)
        unserializable_content = {"data": set([1, 2, 3])}  # type: ignore

        # Try to format an event with unserializable content
        # This will trigger the exception handler at line 96-99
        result = adapter.format_event("delta", unserializable_content)  # type: ignore

        # The result should have ' ' prefix even in error case
        assert result.startswith("data: "), f"SSE event missing 'data: ' prefix. Got: {result[:50]}"

        # Verify it's properly formatted SSE
        assert result.endswith("\n\n"), "SSE event should end with \\n\\n"

        # Verify the error payload is valid JSON
        data_line = result.removeprefix("data: ").removesuffix("\n\n")
        error_payload = json.loads(data_line)
        assert error_payload["type"] == "error"
        assert "content" in error_payload
