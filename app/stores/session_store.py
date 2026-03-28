"""SessionStore Protocol and InMemorySessionStore implementation.

This module provides a pluggable session history store interface for
maintaining conversation state across agent interactions.
"""

import asyncio
import re
import time
import uuid
from collections.abc import Sequence
from typing import Protocol

from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse


class SessionStore(Protocol):
    """Protocol defining the session history store interface.

    Implementations must provide message history persistence, retrieval,
    and deletion capabilities keyed by session identifier.
    """

    async def get_history(self, session_id: str) -> list[ModelMessage]:
        """Retrieve message history for a session.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            List of messages in chronological order. Returns empty list
            if session_id is unknown.
        """
        ...

    async def save_history(self, session_id: str, messages: Sequence[ModelMessage]) -> None:
        """Save message history for a session.

        This operation replaces any existing history for the session.

        Args:
            session_id: Unique identifier for the conversation session.
            messages: Complete message history to store, in chronological order.
        """
        ...

    async def clear(self, session_id: str) -> None:
        """Remove all message history for a session.

        Args:
            session_id: Unique identifier for the conversation session.
                Clearing a non-existent session does not raise an error.
        """
        ...

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions based on TTL.

        Task 3.15: This method is public (not private) so it can be called
        from external code like the lifespan manager.

        Returns:
            Number of sessions removed.
        """
        ...

    def generate_session_id(self) -> str:
        """Generate a new UUID v4 session identifier.

        Task 16.20: Server-side session ID generation for security.
        UUIDs are cryptographically strong and prevent session hijacking
        via guessable or enumerable session IDs.

        Returns:
            A string containing a UUID v4 in standard hyphenated format
            (e.g., "550e8400-e29b-41d4-a716-446655440000").
        """
        ...


class InMemorySessionStore:
    """In-memory session history store with per-session locking.

    This implementation stores conversation histories in a dictionary,
    suitable for development and single-instance deployments. For
    production use with multiple replicas or persistence requirements,
    consider implementing a Redis-backed or database-backed store.

    Thread safety: Uses per-session asyncio.Lock to prevent race conditions
    when multiple concurrent operations access the same session. Each session
    has its own lock to allow concurrent access to different sessions.
    """

    # Task 16.14: Extract magic numbers to class constants for maintainability
    DEFAULT_MAX_MESSAGES: int = 1000
    DEFAULT_SESSION_TTL: int = 3600
    DEFAULT_MAX_SESSIONS: int = 10_000
    MAX_SESSION_ID_LENGTH: int = 256

    def __init__(
        self,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        session_ttl: int = DEFAULT_SESSION_TTL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        """Initialize an empty in-memory session store with per-session locks and TTL.

        Args:
            max_messages: Maximum number of messages allowed per session (default: 1000).
                Used to prevent unbounded memory growth.
            session_ttl: Time-to-live for inactive sessions in seconds (default: 3600).
                Sessions not accessed for this duration will be eligible for cleanup.
            max_sessions: Maximum number of sessions to store (default: 10,000).
                Task 3.16: When exceeded, the least-recently-used session is evicted.
        """
        self._store: dict[str, list[ModelMessage]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_access: dict[str, float] = {}
        # Store-level lock to protect metadata operations (LRU eviction, lock cleanup)
        self._store_lock: asyncio.Lock = asyncio.Lock()
        # Compile regex pattern for session_id validation (alphanumeric, underscore, hyphen only)
        self._session_id_pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
        self.max_messages = max_messages
        self.session_ttl = session_ttl
        self.max_sessions = max_sessions

    async def get_history(self, session_id: str) -> list[ModelMessage]:
        """Retrieve message history for a session.

        Args:
            session_id: Unique identifier for the conversation session.
                Must be 1-256 characters, containing only alphanumeric
                characters, underscores, and hyphens.

        Returns:
            List of messages in chronological order. Returns empty list
            if session_id is unknown.

        Raises:
            ValueError: If session_id is invalid (empty, too long, or contains
                invalid characters).
        """
        self._validate_session_id(session_id)
        # Update last access time
        self._last_access[session_id] = time.time()
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            return self._store.get(session_id, [])

    async def save_history(self, session_id: str, messages: Sequence[ModelMessage]) -> None:
        """Save message history for a session.

        This operation replaces any existing history for the session.

        Args:
            session_id: Unique identifier for the conversation session.
                Must be 1-256 characters, containing only alphanumeric
                characters, underscores, and hyphens.
            messages: Complete message history to store, in chronological order.
                Must not exceed max_messages limit. All elements must be
                ModelMessage instances.

        Raises:
            ValueError: If session_id is invalid (empty, too long, or contains
                invalid characters), or if messages list exceeds max_messages limit.
            TypeError: If messages list contains non-ModelMessage instances.
        """
        self._validate_session_id(session_id)
        self._validate_messages(messages)
        # Update last access time
        self._last_access[session_id] = time.time()
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            self._store[session_id] = list(messages)

        # HIGH FIX: Perform LRU eviction AFTER releasing current session lock
        # This prevents deadlock when two concurrent save_history() calls try to evict each other:
        # - Thread A: holds lock for session_1, tries to evict session_2
        # - Thread B: holds lock for session_2, tries to evict session_1
        # - DEADLOCK: circular wait for each other's locks
        # Solution: Never hold one session lock while trying to acquire another

        # Task 3.16: Evict LRU session if max_sessions limit exceeded
        if len(self._store) > self.max_sessions:
            # Determine which session to evict
            lru_session_id: str | None = None
            lru_lock: asyncio.Lock | None = None

            async with self._store_lock:
                # Re-check after acquiring store lock (double-checked locking pattern)
                if len(self._store) > self.max_sessions:
                    # Find the session with the oldest _last_access time (LRU)
                    lru_session_id = min(self._last_access.items(), key=lambda x: x[1])[0]

                    # Get victim session's lock reference
                    # This prevents concurrent save_history(lru_session) from resurrecting
                    # the session after eviction completes
                    lru_lock = self._locks.setdefault(lru_session_id, asyncio.Lock())

            # Acquire LRU session lock then perform eviction
            # (session lock should be acquired before store lock in the locking hierarchy)
            if lru_lock is not None and lru_session_id is not None:
                async with lru_lock, self._store_lock:
                    # Task 16.37: Re-check capacity inside critical section to prevent
                    # over-eviction when concurrent clear() or TTL cleanup reduced store size
                    # between the initial capacity check and final eviction
                    if len(self._store) > self.max_sessions and lru_session_id in self._store:
                        # Cleanup all session data atomically
                        self._store.pop(lru_session_id, None)
                        self._last_access.pop(lru_session_id, None)
                        self._locks.pop(lru_session_id, None)

    async def clear(self, session_id: str) -> None:
        """Remove all message history for a session.

        This operation acquires the session lock to prevent race conditions
        with concurrent get_history or save_history operations. After clearing,
        the lock entry is removed from the _locks dict to prevent memory leaks.

        Clearing a non-existent session does not raise an error.

        Args:
            session_id: Unique identifier for the conversation session.
                Must be 1-256 characters, containing only alphanumeric
                characters, underscores, and hyphens.

        Raises:
            ValueError: If session_id is invalid (empty, too long, or contains
                invalid characters).
        """
        self._validate_session_id(session_id)
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            self._store.pop(session_id, None)
            # Task 3.12: Remove _last_access entry to prevent memory leak
            self._last_access.pop(session_id, None)
        # FIX: Protect lock cleanup with store lock to prevent race condition
        # where concurrent operation creates new lock via setdefault between
        # releasing session lock and popping lock entry
        async with self._store_lock:
            # Clean up lock entry to prevent memory leak in long-running services
            self._locks.pop(session_id, None)

    def _validate_session_id(self, session_id: str) -> None:
        """Validate session_id input.

        Args:
            session_id: The session identifier to validate.

        Raises:
            ValueError: If session_id is empty, exceeds 256 characters,
                or contains characters outside [a-zA-Z0-9_-].
        """
        if not session_id:
            raise ValueError("session_id cannot be empty")
        if len(session_id) > self.MAX_SESSION_ID_LENGTH:
            raise ValueError(f"session_id too long (max {self.MAX_SESSION_ID_LENGTH} chars)")
        if not self._session_id_pattern.match(session_id):
            raise ValueError("session_id contains invalid characters")

    def _validate_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Validate messages parameter.

        Args:
            messages: The messages list to validate.

        Raises:
            ValueError: If messages list exceeds max_messages limit.
            TypeError: If messages list contains non-ModelMessage instances.
        """
        # Check message count limit
        if len(messages) > self.max_messages:
            raise ValueError(f"Too many messages (max {self.max_messages})")

        # Task 3.14: Use strict isinstance check instead of structural validation
        # This prevents duck-typed objects from bypassing validation
        for msg in messages:
            if not isinstance(msg, (ModelRequest, ModelResponse)):
                raise TypeError("All messages must be ModelMessage instances")

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions based on TTL.

        Task 3.15: This method is public (not private) so it can be called
        from external code like the lifespan manager.

        Iterates through all sessions and removes those that haven't been
        accessed for longer than session_ttl seconds.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        expired_sessions = []

        # Find expired sessions
        # FIX: Create snapshot of items to prevent RuntimeError if dict is modified during iteration
        for session_id, last_access in list(self._last_access.items()):
            if now - last_access > self.session_ttl:
                expired_sessions.append(session_id)

        # Remove expired sessions
        for session_id in expired_sessions:
            await self.clear(session_id)

        return len(expired_sessions)

    def generate_session_id(self) -> str:
        """Generate a new UUID v4 session identifier.

        Task 16.20: Server-side session ID generation for security.
        UUIDs are cryptographically strong and prevent session hijacking
        via guessable or enumerable session IDs.

        Returns:
            A string containing a UUID v4 in standard hyphenated format
            (e.g., "550e8400-e29b-41d4-a716-446655440000").
        """
        return str(uuid.uuid4())


class RedisSessionStore:
    """Redis-backed session history store for multi-instance deployments.

    Task 17.2: Production-ready session store using Redis for persistence.
    This implementation enables session sharing across multiple application
    instances, making it suitable for horizontally scaled deployments.

    Features:
        - Automatic TTL-based expiration via Redis
        - JSON serialization of Pydantic AI ModelMessage objects
        - Session ID validation (same rules as InMemorySessionStore)
        - Connection pooling via redis-py
        - Async operations via redis.asyncio

    Thread safety: Redis operations are atomic by default. Multiple concurrent
    requests can safely access different sessions, and Redis ensures data
    consistency for operations on the same session.

    Args:
        redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
        session_ttl: Time-to-live for sessions in seconds (default: 3600)
        key_prefix: Prefix for all Redis keys (default: "session:")
    """

    DEFAULT_SESSION_TTL: int = 3600
    DEFAULT_KEY_PREFIX: str = "session:"
    MAX_SESSION_ID_LENGTH: int = 256

    def __init__(
        self,
        redis_url: str,
        session_ttl: int = DEFAULT_SESSION_TTL,
        key_prefix: str = DEFAULT_KEY_PREFIX,
    ) -> None:
        """Initialize Redis session store.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
            session_ttl: Time-to-live for inactive sessions in seconds (default: 3600)
            key_prefix: Prefix for all Redis keys (default: "session:")
        """
        import redis.asyncio as redis

        self._redis = redis.from_url(redis_url, decode_responses=False)
        self.session_ttl = session_ttl
        self.key_prefix = key_prefix
        # Compile regex pattern for session_id validation
        self._session_id_pattern = re.compile(r"^[a-zA-Z0-9_-]+$")

    async def get_history(self, session_id: str) -> list[ModelMessage]:
        """Retrieve message history for a session from Redis.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            List of messages in chronological order. Returns empty list
            if session_id is unknown or data is invalid.

        Raises:
            ValueError: If session_id is invalid.
        """
        self._validate_session_id(session_id)

        key = f"{self.key_prefix}{session_id}"
        data = await self._redis.get(key)

        if data is None:
            return []

        # Deserialize pickle bytes to ModelMessage objects
        import pickle

        try:
            # Task 17.2: Using pickle for ModelMessage serialization
            # Security note: pickle.loads() can be unsafe with untrusted data.
            # This is acceptable here because:
            # 1. Data comes from Redis (internal trusted store, not user input)
            # 2. Redis connection requires authentication
            # 3. Used only for session storage in authenticated API
            messages: list[ModelMessage] = pickle.loads(data)  # noqa: S301
            return messages
        except (pickle.UnpicklingError, TypeError, AttributeError):
            # If data is corrupted, return empty list
            return []

    async def save_history(self, session_id: str, messages: Sequence[ModelMessage]) -> None:
        """Save message history for a session to Redis.

        Args:
            session_id: Unique identifier for the conversation session.
            messages: Complete message history to store.

        Raises:
            ValueError: If session_id is invalid.
        """
        self._validate_session_id(session_id)

        # Serialize messages to pickle bytes
        import pickle

        serialized = pickle.dumps(list(messages))

        key = f"{self.key_prefix}{session_id}"
        # Store with TTL - Redis will automatically expire the key
        await self._redis.set(key, serialized, ex=self.session_ttl)

    async def clear(self, session_id: str) -> None:
        """Remove all message history for a session from Redis.

        Args:
            session_id: Unique identifier for the conversation session.

        Raises:
            ValueError: If session_id is invalid.
        """
        self._validate_session_id(session_id)
        key = f"{self.key_prefix}{session_id}"
        await self._redis.delete(key)

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions from Redis.

        Note: Redis automatically handles expiration via TTL, so this method
        performs no actual work. It exists to satisfy the SessionStore protocol
        and returns 0 to indicate Redis is handling expiration automatically.

        Returns:
            Always returns 0 (Redis handles expiration automatically).
        """
        # Redis handles TTL expiration automatically, no cleanup needed
        # This method exists to satisfy the SessionStore protocol
        return 0

    def generate_session_id(self) -> str:
        """Generate a new UUID v4 session identifier.

        Returns:
            A string containing a UUID v4 in standard hyphenated format.
        """
        return str(uuid.uuid4())

    def _validate_session_id(self, session_id: str) -> None:
        """Validate session_id input.

        Args:
            session_id: The session identifier to validate.

        Raises:
            ValueError: If session_id is empty, exceeds 256 characters,
                or contains characters outside [a-zA-Z0-9_-].
        """
        if not session_id:
            raise ValueError("session_id cannot be empty")
        if len(session_id) > self.MAX_SESSION_ID_LENGTH:
            raise ValueError(f"session_id too long (max {self.MAX_SESSION_ID_LENGTH} chars)")
        if not self._session_id_pattern.match(session_id):
            raise ValueError("session_id contains invalid characters")

    async def close(self) -> None:
        """Close the Redis connection.

        Should be called during application shutdown to properly clean up resources.
        """
        await self._redis.close()
