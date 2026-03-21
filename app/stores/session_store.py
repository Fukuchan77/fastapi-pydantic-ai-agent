"""SessionStore Protocol and InMemorySessionStore implementation.

This module provides a pluggable session history store interface for
maintaining conversation state across agent interactions.
"""

import asyncio
import re
from collections.abc import Sequence
from typing import Protocol

from pydantic_ai.messages import ModelMessage


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

    def __init__(self, max_messages: int = 1000) -> None:
        """Initialize an empty in-memory session store with per-session locks.

        Args:
            max_messages: Maximum number of messages allowed per session (default: 1000).
                Used to prevent unbounded memory growth.
        """
        self._store: dict[str, list[ModelMessage]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Compile regex pattern for session_id validation (alphanumeric, underscore, hyphen only)
        self._session_id_pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
        self.max_messages = max_messages

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
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            self._store[session_id] = list(messages)

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
        if len(session_id) > 256:
            raise ValueError("session_id too long (max 256 chars)")
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

        # Check that all elements are ModelMessage instances using structural validation
        # ModelMessage types should have a 'parts' attribute
        for msg in messages:
            if not hasattr(msg, "parts"):
                raise TypeError("All messages must be ModelMessage instances")
