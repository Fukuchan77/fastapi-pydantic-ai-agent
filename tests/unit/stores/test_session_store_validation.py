"""Unit tests for SessionStore session_id input validation."""

import pytest
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore


class TestSessionIdValidation:
    """Test that session_id input is properly validated."""

    @pytest.fixture
    def store(self) -> InMemorySessionStore:
        """Provide a fresh InMemorySessionStore instance."""
        return InMemorySessionStore()

    @pytest.fixture
    def sample_messages(self) -> list[ModelRequest]:
        """Provide sample messages for testing."""
        return [ModelRequest(parts=[UserPromptPart(content="Test message")])]

    # ===== Tests for empty session_id =====

    @pytest.mark.asyncio
    async def test_get_history_rejects_empty_session_id(self, store: InMemorySessionStore) -> None:
        """get_history must reject empty session_id."""
        with pytest.raises(ValueError, match="session_id cannot be empty"):
            await store.get_history("")

    @pytest.mark.asyncio
    async def test_save_history_rejects_empty_session_id(
        self, store: InMemorySessionStore, sample_messages: list[ModelRequest]
    ) -> None:
        """save_history must reject empty session_id."""
        with pytest.raises(ValueError, match="session_id cannot be empty"):
            await store.save_history("", sample_messages)

    @pytest.mark.asyncio
    async def test_clear_rejects_empty_session_id(self, store: InMemorySessionStore) -> None:
        """Clear must reject empty session_id."""
        with pytest.raises(ValueError, match="session_id cannot be empty"):
            await store.clear("")

    # ===== Tests for session_id too long =====

    @pytest.mark.asyncio
    async def test_get_history_rejects_too_long_session_id(
        self, store: InMemorySessionStore
    ) -> None:
        """get_history must reject session_id exceeding 256 characters."""
        long_id = "a" * 257
        with pytest.raises(ValueError, match="session_id too long \\(max 256 chars\\)"):
            await store.get_history(long_id)

    @pytest.mark.asyncio
    async def test_save_history_rejects_too_long_session_id(
        self, store: InMemorySessionStore, sample_messages: list[ModelRequest]
    ) -> None:
        """save_history must reject session_id exceeding 256 characters."""
        long_id = "a" * 257
        with pytest.raises(ValueError, match="session_id too long \\(max 256 chars\\)"):
            await store.save_history(long_id, sample_messages)

    @pytest.mark.asyncio
    async def test_clear_rejects_too_long_session_id(self, store: InMemorySessionStore) -> None:
        """Clear must reject session_id exceeding 256 characters."""
        long_id = "a" * 257
        with pytest.raises(ValueError, match="session_id too long \\(max 256 chars\\)"):
            await store.clear(long_id)

    # ===== Tests for invalid characters =====

    @pytest.mark.asyncio
    async def test_get_history_rejects_invalid_characters(
        self, store: InMemorySessionStore
    ) -> None:
        """get_history must reject session_id with invalid characters."""
        invalid_ids = [
            "session@id",  # @ not allowed
            "session id",  # space not allowed
            "session/id",  # / not allowed
            "session.id",  # . not allowed
            "session#id",  # # not allowed
            "session$id",  # $ not allowed
        ]
        for invalid_id in invalid_ids:
            with pytest.raises(ValueError, match="session_id contains invalid characters"):
                await store.get_history(invalid_id)

    @pytest.mark.asyncio
    async def test_save_history_rejects_invalid_characters(
        self, store: InMemorySessionStore, sample_messages: list[ModelRequest]
    ) -> None:
        """save_history must reject session_id with invalid characters."""
        invalid_ids = [
            "session@id",
            "session id",
            "session/id",
            "session.id",
            "session#id",
            "session$id",
        ]
        for invalid_id in invalid_ids:
            with pytest.raises(ValueError, match="session_id contains invalid characters"):
                await store.save_history(invalid_id, sample_messages)

    @pytest.mark.asyncio
    async def test_clear_rejects_invalid_characters(self, store: InMemorySessionStore) -> None:
        """Clear must reject session_id with invalid characters."""
        invalid_ids = [
            "session@id",
            "session id",
            "session/id",
            "session.id",
            "session#id",
            "session$id",
        ]
        for invalid_id in invalid_ids:
            with pytest.raises(ValueError, match="session_id contains invalid characters"):
                await store.clear(invalid_id)

    # ===== Tests for valid session_id (boundary cases) =====

    @pytest.mark.asyncio
    async def test_valid_session_id_with_allowed_characters(
        self, store: InMemorySessionStore, sample_messages: list[ModelRequest]
    ) -> None:
        """Valid session_id with alphanumeric, underscore, and hyphen should work."""
        valid_ids = [
            "session123",
            "session_id",
            "session-id",
            "session_123-abc",
            "a",  # Single character
            "123",  # Numeric only
            "_",  # Underscore only
            "-",  # Hyphen only
        ]
        for valid_id in valid_ids:
            # Should not raise any exception
            await store.save_history(valid_id, sample_messages)
            history = await store.get_history(valid_id)
            assert len(history) == 1
            await store.clear(valid_id)

    @pytest.mark.asyncio
    async def test_session_id_at_max_length(
        self, store: InMemorySessionStore, sample_messages: list[ModelRequest]
    ) -> None:
        """session_id exactly at 256 chars should be accepted."""
        max_length_id = "a" * 256
        # Should not raise any exception
        await store.save_history(max_length_id, sample_messages)
        history = await store.get_history(max_length_id)
        assert len(history) == 1
        await store.clear(max_length_id)
