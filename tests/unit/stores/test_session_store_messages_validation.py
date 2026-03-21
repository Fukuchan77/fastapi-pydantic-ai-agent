"""Unit tests for InMemorySessionStore.save_history() messages validation."""

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore


class TestSessionStoreMessagesValidation:
    """Test messages parameter validation in save_history()."""

    @pytest.fixture
    def store(self) -> InMemorySessionStore:
        """Provide a fresh InMemorySessionStore instance."""
        return InMemorySessionStore()

    @pytest.mark.asyncio
    async def test_save_history_too_many_messages_raises_error(
        self, store: InMemorySessionStore
    ) -> None:
        """save_history must raise ValueError when messages exceed max_messages limit."""
        session_id = "test-session"

        # Create 1001 messages (default max is 1000)
        too_many_messages = [
            ModelRequest(parts=[UserPromptPart(content=f"Message {i}")]) for i in range(1001)
        ]

        with pytest.raises(ValueError, match=r"Too many messages \(max \d+\)") as exc_info:
            await store.save_history(session_id, too_many_messages)

        error_msg = str(exc_info.value)
        assert "too many messages" in error_msg.lower()
        assert "max" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_save_history_invalid_type_raises_error(
        self, store: InMemorySessionStore
    ) -> None:
        """save_history must raise TypeError when messages contain non-ModelMessage items."""
        session_id = "test-session"

        # Create a list with invalid types mixed in
        invalid_messages = [
            ModelRequest(parts=[UserPromptPart(content="Valid message")]),
            "This is not a ModelMessage",  # type: ignore
            {"content": "This is also not a ModelMessage"},  # type: ignore
        ]

        with pytest.raises(
            TypeError, match="All messages must be ModelMessage instances"
        ) as exc_info:
            await store.save_history(session_id, invalid_messages)

        error_msg = str(exc_info.value)
        assert "modelmessage" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_save_history_exactly_at_limit_passes(self, store: InMemorySessionStore) -> None:
        """save_history must accept exactly max_messages count."""
        session_id = "test-session"

        # Create exactly 1000 messages (the default limit)
        exactly_max_messages = [
            ModelRequest(parts=[UserPromptPart(content=f"Message {i}")]) for i in range(1000)
        ]

        # Should not raise any exception
        await store.save_history(session_id, exactly_max_messages)

        # Verify they were saved
        history = await store.get_history(session_id)
        assert len(history) == 1000

    @pytest.mark.asyncio
    async def test_save_history_empty_list_passes(self, store: InMemorySessionStore) -> None:
        """save_history must accept empty message list."""
        session_id = "test-session"

        # Should not raise any exception
        await store.save_history(session_id, [])

        # Verify empty list was saved
        history = await store.get_history(session_id)
        assert history == []

    @pytest.mark.asyncio
    async def test_save_history_all_valid_types_passes(self, store: InMemorySessionStore) -> None:
        """save_history must accept all valid ModelMessage types."""
        session_id = "test-session"

        # Create messages with valid types
        valid_messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="User message 1")]),
            ModelRequest(parts=[UserPromptPart(content="User message 2")]),
        ]

        # Should not raise any exception
        await store.save_history(session_id, valid_messages)

        # Verify they were saved
        history = await store.get_history(session_id)
        assert len(history) == 2
