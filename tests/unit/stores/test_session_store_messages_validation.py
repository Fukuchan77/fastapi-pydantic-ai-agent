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
    async def test_save_history_too_many_messages_trims_to_limit(
        self, store: InMemorySessionStore
    ) -> None:
        """save_history must trim to max_messages instead of raising when exceeded (3.3, 3.4)."""
        session_id = "test-session"

        # Create 1001 messages (default max is 1000); no tool calls, so the
        # cut lands exactly at the cap with no head-pin overshoot.
        too_many_messages = [
            ModelRequest(parts=[UserPromptPart(content=f"Message {i}")]) for i in range(1001)
        ]

        await store.save_history(session_id, too_many_messages)

        history = await store.get_history(session_id)
        assert len(history) == 1000
        # Capacity contract (protocol.py save_history docstring): the pinned
        # head survives and the *oldest* of the rest is what's discarded, not
        # an arbitrary suffix/prefix truncation — proving the store actually
        # delegates to the trimmer rather than e.g. keeping messages[:1000].
        assert history[0].parts[0].content == "Message 0"  # type: ignore[attr-defined]
        assert history[1].parts[0].content == "Message 2"  # type: ignore[attr-defined]
        assert history[-1].parts[0].content == "Message 1000"  # type: ignore[attr-defined]

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
