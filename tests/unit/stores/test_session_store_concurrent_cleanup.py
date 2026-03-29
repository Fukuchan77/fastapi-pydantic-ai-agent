"""Tests for concurrent iteration issue in cleanup_expired_sessions.

Problem: RuntimeError can occur when cleanup_expired_sessions iterates over
_last_access.items() while another coroutine modifies the dictionary via clear().
"""

import asyncio

import pytest

from app.stores.session_store import InMemorySessionStore


@pytest.mark.asyncio
async def test_cleanup_with_concurrent_clear_no_runtime_error() -> None:
    """Test cleanup_expired_sessions with concurrent modifications.

    RED PHASE: This test should fail because the current implementation
    iterates directly over self._last_access.items() without creating a snapshot,
    allowing RuntimeError when another coroutine calls clear() during iteration.
    """
    # Create store with very short TTL to force immediate expiration
    store = InMemorySessionStore(session_ttl=0)

    # Add multiple sessions
    session_ids = [f"session_{i}" for i in range(10)]
    for session_id in session_ids:
        await store.save_history(session_id, [])

    # Create a task that runs cleanup while another task clears sessions
    async def cleanup_loop() -> None:
        """Run cleanup multiple times to increase chance of concurrent modification."""
        for _ in range(10):
            await store.cleanup_expired_sessions()
            await asyncio.sleep(0.001)  # Small delay to allow interleaving

    async def clear_loop() -> None:
        """Clear sessions concurrently to trigger dictionary modification during iteration."""
        for session_id in session_ids:
            await store.clear(session_id)
            await asyncio.sleep(0.001)  # Small delay to allow interleaving

    # Run both tasks concurrently - should not raise RuntimeError
    try:
        await asyncio.gather(
            cleanup_loop(),
            clear_loop(),
        )
    except RuntimeError as e:
        if "dictionary changed size during iteration" in str(e):
            pytest.fail(f"RuntimeError during concurrent cleanup and clear: {e}")
        raise


@pytest.mark.asyncio
async def test_cleanup_with_concurrent_save_no_runtime_error() -> None:
    """Test that cleanup_expired_sessions doesn't raise RuntimeError during concurrent saves.

    This tests the scenario where new sessions are being added (via save_history
    which updates _last_access) while cleanup is iterating.
    """
    store = InMemorySessionStore(session_ttl=0)

    # Add initial expired sessions
    for i in range(5):
        await store.save_history(f"expired_{i}", [])

    async def cleanup_loop() -> None:
        """Run cleanup multiple times."""
        for _ in range(10):
            await store.cleanup_expired_sessions()
            await asyncio.sleep(0.001)

    async def save_loop() -> None:
        """Add new sessions concurrently."""
        for i in range(10):
            await store.save_history(f"new_{i}", [])
            await asyncio.sleep(0.001)

    # Should not raise RuntimeError
    try:
        await asyncio.gather(
            cleanup_loop(),
            save_loop(),
        )
    except RuntimeError as e:
        if "dictionary changed size during iteration" in str(e):
            pytest.fail(f"RuntimeError during concurrent cleanup and save: {e}")
        raise
