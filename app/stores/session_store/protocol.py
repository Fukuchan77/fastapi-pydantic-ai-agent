"""SessionStore Protocol defining the pluggable session history store interface."""

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

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions based on TTL.

        This method is public (not private) so it can be called
        from external code like the lifespan manager.

        Returns:
            Number of sessions removed.
        """
        ...

    def generate_session_id(self) -> str:
        """Generate a new UUID v4 session identifier.

        Server-side session ID generation for security.
        UUIDs are cryptographically strong and prevent session hijacking
        via guessable or enumerable session IDs.

        Returns:
            A string containing a UUID v4 in standard hyphenated format
            (e.g., "550e8400-e29b-41d4-a716-446655440000").
        """
        ...
