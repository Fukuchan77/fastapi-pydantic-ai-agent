"""Pluggable session history store interface and implementations.

Split by backend per `.sdd/steering/file-size-policy.md`
(`protocol.py`, `in_memory.py`, `redis.py`); this package re-exports the
public surface so `from app.stores.session_store import SessionStore,
InMemorySessionStore, RedisSessionStore` keeps working unchanged.
"""

from app.stores.session_store.in_memory import InMemorySessionStore
from app.stores.session_store.protocol import SessionStore
from app.stores.session_store.redis import RedisSessionStore


__all__ = ["InMemorySessionStore", "RedisSessionStore", "SessionStore"]
