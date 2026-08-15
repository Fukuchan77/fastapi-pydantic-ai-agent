"""Regression tests: phantom `_last_access` entries must not defeat `max_sessions`.

`get_history()` stamps `_last_access` for a session id even when that id is
never saved (Requirement 2, ADR-4 — this write is deliberate: it is the only
path that lets TTL cleanup reclaim the per-session lock of a read-but-never-
saved id). The bug this file guards against is in *victim selection*, not in
that write: scanning `_last_access` for the oldest timestamp can pick a
phantom id that eviction's `lru_session_id in self._store` guard then refuses
to remove, so nothing is evicted and `max_sessions` is inert until TTL expiry.
"""

import asyncio
import time

import pytest
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore


@pytest.mark.asyncio
async def test_phantom_entry_never_selected_as_eviction_victim() -> None:
    """A read-but-never-saved id must not block eviction of the genuine LRU.

    Requirements: 2.1, 2.2, 2.3, 2.7.
    """
    store = InMemorySessionStore(max_sessions=2)
    msg = ModelRequest(parts=[UserPromptPart(content="test")])

    # Create a phantom `_last_access` entry, older than every saved session.
    await store.get_history("phantom")
    assert "phantom" not in store._store
    assert "phantom" in store._last_access

    await asyncio.sleep(0.01)
    await store.save_history("session_1", [msg])
    await asyncio.sleep(0.01)
    await store.save_history("session_2", [msg])

    # Over capacity: this save must evict exactly one *saved* session.
    await store.save_history("session_3", [msg])

    assert len(store._store) <= store.max_sessions, (
        "stored session count must never exceed max_sessions, even when a "
        "phantom last-access entry is older than every saved session"
    )
    assert "session_1" not in store._store, (
        "session_1 is the genuine least-recently-accessed saved session and "
        "must be the eviction victim"
    )
    assert "session_2" in store._store
    assert "session_3" in store._store


@pytest.mark.asyncio
async def test_phantom_entry_lock_still_reclaimed_by_ttl_cleanup() -> None:
    """TTL cleanup must still reclaim the lock of a read-but-never-saved id.

    This is the invariant whose loss would turn the eviction fix into an
    unbounded per-session lock leak (ADR-4), so it is asserted directly
    rather than left to code review.

    Requirements: 2.5.
    """
    store = InMemorySessionStore(session_ttl=1)

    await store.get_history("phantom")
    assert "phantom" not in store._store
    assert "phantom" in store._last_access
    assert "phantom" in store._locks

    store._last_access["phantom"] = time.time() - 2

    expired_count = await store.cleanup_expired_sessions()

    assert expired_count == 1
    assert "phantom" not in store._last_access
    assert "phantom" not in store._locks
