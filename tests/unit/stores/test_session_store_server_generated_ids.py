"""Unit tests for server-side session ID generation (Task 16.20).

Tests verify that:
- SessionStore can generate new UUIDs for sessions
- Generated IDs follow UUID v4 format
- Duplicate ID generation is extremely unlikely
"""

import re
import uuid

from app.stores.session_store import InMemorySessionStore


class TestServerGeneratedSessionIds:
    """Test server-side UUID generation for session IDs (Task 16.20)."""

    def test_session_store_has_generate_session_id_method(self) -> None:
        """SessionStore should have a generate_session_id() method."""
        store = InMemorySessionStore()
        assert hasattr(store, "generate_session_id")

    def test_generate_session_id_returns_string(self) -> None:
        """generate_session_id() should return a string."""
        store = InMemorySessionStore()
        session_id = store.generate_session_id()
        assert isinstance(session_id, str)

    def test_generate_session_id_returns_valid_uuid(self) -> None:
        """generate_session_id() should return a valid UUID v4 string."""
        store = InMemorySessionStore()
        session_id = store.generate_session_id()
        # Should be parseable as UUID
        parsed = uuid.UUID(session_id)
        # Should be version 4
        assert parsed.version == 4

    def test_generate_session_id_uses_hyphenated_format(self) -> None:
        """generate_session_id() should return UUID in standard hyphenated format."""
        store = InMemorySessionStore()
        session_id = store.generate_session_id()
        # Standard UUID format: 8-4-4-4-12 hexadecimal characters with hyphens
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        assert uuid_pattern.match(session_id), f"Invalid UUID format: {session_id}"

    def test_generate_session_id_produces_unique_ids(self) -> None:
        """generate_session_id() should produce unique IDs on each call."""
        store = InMemorySessionStore()
        ids = {store.generate_session_id() for _ in range(100)}
        # All 100 IDs should be unique
        assert len(ids) == 100

    def test_generated_id_is_valid_for_save_history(self) -> None:
        """Generated session IDs should pass validation in save_history()."""
        store = InMemorySessionStore()
        session_id = store.generate_session_id()
        # Should not raise validation error
        import asyncio

        asyncio.run(store.save_history(session_id, []))

    def test_generated_id_length_within_limits(self) -> None:
        """Generated session IDs should be within the 256 character limit."""
        store = InMemorySessionStore()
        session_id = store.generate_session_id()
        # UUID v4 in hyphenated format is exactly 36 characters
        assert len(session_id) == 36
        assert len(session_id) <= store.MAX_SESSION_ID_LENGTH
