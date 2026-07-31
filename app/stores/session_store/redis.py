"""RedisSessionStore: multi-instance-safe session history store backed by Redis."""

import logging
import re
import uuid
from collections.abc import Sequence

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelMessagesTypeAdapter


# Logger for exception logging in RedisSessionStore
logger = logging.getLogger(__name__)


class RedisSessionStore:
    """Redis-backed session history store for multi-instance deployments.

    Production-ready session store using Redis for persistence.
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
        self._session_id_pattern = re.compile(r"^[a-zA-Z0-9_.-]+$")

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

        # Deserialize JSON bytes to ModelMessage objects using Pydantic TypeAdapter
        # Security: Replaced pickle.loads() with type-safe JSON deserialization
        # to eliminate RCE vector if Redis is compromised
        try:
            messages: list[ModelMessage] = ModelMessagesTypeAdapter.validate_json(data)
            return messages
        except (ValidationError, ValueError, TypeError):
            # Narrow exception handler to only catch expected deserialization errors
            # ValidationError: Pydantic validation fails (invalid schema)
            # ValueError: JSON parsing fails (malformed JSON)
            # TypeError: Data is wrong type (e.g., not bytes/str)
            # Log exception details for production debugging
            # exc_info=True includes full traceback, critical for diagnosing
            # deserialization errors in production
            logger.warning(
                "Failed to deserialize session %s",
                session_id,
                exc_info=True,
            )
            # If data is corrupted or invalid JSON, return empty list
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

        # Serialize messages to JSON bytes using Pydantic TypeAdapter
        # Security: Replaced pickle.dumps() with type-safe JSON serialization
        serialized = ModelMessagesTypeAdapter.dump_json(list(messages))

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
                or contains characters outside [a-zA-Z0-9_.-].
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
