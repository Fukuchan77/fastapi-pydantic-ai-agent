"""Unit tests for SessionStore Protocol and InMemorySessionStore implementation."""

import asyncio

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore
from app.stores.session_store import SessionStore


class TestSessionStoreProtocol:
    """Test that SessionStore Protocol is correctly defined."""

    def test_session_store_protocol_has_required_methods(self) -> None:
        """SessionStore Protocol must define get_history, save_history, and clear methods."""
        assert hasattr(SessionStore, "get_history")
        assert hasattr(SessionStore, "save_history")
        assert hasattr(SessionStore, "clear")

    def test_session_store_protocol_has_cleanup_expired_sessions(self) -> None:
        """SessionStore Protocol must define cleanup_expired_sessions method.

        Task 3.15: The cleanup_expired_sessions method must be public (not private)
        so it can be called from external code like the lifespan manager.

        RED PHASE: This test will fail initially because cleanup_expired_sessions
        is not yet in the SessionStore Protocol.
        """
        assert hasattr(SessionStore, "cleanup_expired_sessions"), (
            "SessionStore Protocol must define cleanup_expired_sessions method"
        )


class TestInMemorySessionStore:
    """Test InMemorySessionStore implementation."""

    @pytest.fixture
    def store(self) -> InMemorySessionStore:
        """Provide a fresh InMemorySessionStore instance."""
        return InMemorySessionStore()

    @pytest.fixture
    def sample_messages(self) -> list[ModelMessage]:
        """Provide sample ModelMessage instances for testing."""
        return [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there!")]),
            ModelRequest(parts=[UserPromptPart(content="How are you?")]),
            ModelResponse(parts=[TextPart(content="I'm doing well, thanks!")]),
        ]

    @pytest.mark.asyncio
    async def test_get_history_unknown_session_returns_empty_list(
        self, store: InMemorySessionStore
    ) -> None:
        """Get history for unknown session_id must return empty list."""
        history = await store.get_history("unknown-session-id")
        assert history == []

    @pytest.mark.asyncio
    async def test_save_and_get_history(
        self, store: InMemorySessionStore, sample_messages: list[ModelMessage]
    ) -> None:
        """Messages can be saved and retrieved by session_id."""
        session_id = "test-session-123"

        await store.save_history(session_id, sample_messages)

        retrieved = await store.get_history(session_id)
        assert len(retrieved) == 4
        assert retrieved == sample_messages

    @pytest.mark.asyncio
    async def test_history_append_order_preserved(self, store: InMemorySessionStore) -> None:
        """History must preserve message order across multiple save operations."""
        session_id = "test-session-456"

        # First conversation turn
        first_messages = [
            ModelRequest(parts=[UserPromptPart(content="First message")]),
            ModelResponse(parts=[TextPart(content="First response")]),
        ]
        await store.save_history(session_id, first_messages)

        # Second conversation turn
        second_messages = [
            ModelRequest(parts=[UserPromptPart(content="Second message")]),
            ModelResponse(parts=[TextPart(content="Second response")]),
        ]
        await store.save_history(session_id, first_messages + second_messages)

        # Verify order is preserved
        history = await store.get_history(session_id)
        assert len(history) == 4
        assert history[0] == first_messages[0]
        assert history[1] == first_messages[1]
        assert history[2] == second_messages[0]
        assert history[3] == second_messages[1]

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self, store: InMemorySessionStore) -> None:
        """Different session IDs must have isolated message histories."""
        session1 = "session-1"
        session2 = "session-2"

        messages1 = [ModelRequest(parts=[UserPromptPart(content="Session 1 message")])]
        messages2 = [ModelRequest(parts=[UserPromptPart(content="Session 2 message")])]

        await store.save_history(session1, messages1)
        await store.save_history(session2, messages2)

        history1 = await store.get_history(session1)
        history2 = await store.get_history(session2)

        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0] != history2[0]

    @pytest.mark.asyncio
    async def test_clear_removes_session_history(
        self, store: InMemorySessionStore, sample_messages: list[ModelMessage]
    ) -> None:
        """Clear must remove all messages for the specified session."""
        session_id = "test-session-789"

        await store.save_history(session_id, sample_messages)

        # Verify messages were saved
        history_before = await store.get_history(session_id)
        assert len(history_before) == 4

        # Clear the session
        await store.clear(session_id)

        # Verify session is empty
        history_after = await store.get_history(session_id)
        assert history_after == []

    @pytest.mark.asyncio
    async def test_clear_only_affects_specified_session(self, store: InMemorySessionStore) -> None:
        """Clear must only remove history for the specified session_id."""
        session1 = "session-to-clear"
        session2 = "session-to-keep"

        messages1 = [ModelRequest(parts=[UserPromptPart(content="Clear me")])]
        messages2 = [ModelRequest(parts=[UserPromptPart(content="Keep me")])]

        await store.save_history(session1, messages1)
        await store.save_history(session2, messages2)

        # Clear only session1
        await store.clear(session1)

        # Verify session1 is cleared but session2 remains
        assert await store.get_history(session1) == []
        assert len(await store.get_history(session2)) == 1

    @pytest.mark.asyncio
    async def test_save_empty_messages_list(self, store: InMemorySessionStore) -> None:
        """Saving empty message list should not raise error."""
        session_id = "empty-session"

        await store.save_history(session_id, [])

        history = await store.get_history(session_id)
        assert history == []

    @pytest.mark.asyncio
    async def test_save_overwrites_previous_history(self, store: InMemorySessionStore) -> None:
        """Save operation must replace (not append to) existing history."""
        session_id = "overwrite-session"

        # Save initial messages
        initial_messages = [
            ModelRequest(parts=[UserPromptPart(content="Initial message")]),
        ]
        await store.save_history(session_id, initial_messages)

        # Save new messages (should replace, not append)
        new_messages = [
            ModelRequest(parts=[UserPromptPart(content="New message 1")]),
            ModelRequest(parts=[UserPromptPart(content="New message 2")]),
        ]
        await store.save_history(session_id, new_messages)

        history = await store.get_history(session_id)
        assert len(history) == 2
        assert history == new_messages

    @pytest.mark.asyncio
    async def test_clear_unknown_session_does_not_raise(self, store: InMemorySessionStore) -> None:
        """Clearing a non-existent session should not raise an error."""
        # Should not raise any exception
        await store.clear("non-existent-session")

    @pytest.mark.asyncio
    async def test_concurrent_save_history_no_data_loss(self, store: InMemorySessionStore) -> None:
        """Concurrent save_history calls for the same session must not corrupt data.

        This test verifies that per-session locking prevents data corruption when
        multiple concurrent operations attempt to save history for the same session.
        Each concurrent save writes a complete valid history. The locking ensures:
        1. No internal dict corruption occurs
        2. The final state is one complete valid history (not a mix/partial state)
        3. All write operations complete successfully without errors
        """
        session_id = "concurrent-session"
        num_concurrent_saves = 50

        # Create distinct complete histories for each concurrent save
        histories_to_save = []
        for i in range(num_concurrent_saves):
            history = [
                ModelRequest(parts=[UserPromptPart(content=f"Batch{i}-Msg1")]),
                ModelResponse(parts=[TextPart(content=f"Batch{i}-Reply1")]),
                ModelRequest(parts=[UserPromptPart(content=f"Batch{i}-Msg2")]),
            ]
            histories_to_save.append(history)

        async def save_complete_history(history: list[ModelMessage]) -> None:
            """Save a complete history for the session."""
            # Add small delay to force concurrent execution
            await asyncio.sleep(0.001)
            await store.save_history(session_id, history)

        # Fire off many concurrent save operations
        tasks = [save_complete_history(hist) for hist in histories_to_save]
        await asyncio.gather(*tasks)

        # Verify final state is one complete valid history (not corrupted)
        final_history = await store.get_history(session_id)
        assert len(final_history) == 3, (
            f"Expected final history to have 3 messages (one complete batch), "
            f"but got {len(final_history)}. This indicates data corruption."
        )

        # Verify the final history is internally consistent (all from same batch)
        if len(final_history) >= 2:
            # Extract batch number from first message
            first_msg = final_history[0]
            if isinstance(first_msg, ModelRequest):
                first_content = first_msg.parts[0].content
                if isinstance(first_content, str):
                    batch_num = first_content.split("-")[0]
                    # All messages should be from the same batch
                    for msg in final_history:
                        if isinstance(msg, (ModelRequest, ModelResponse)):
                            content = msg.parts[0].content
                            if isinstance(content, str):
                                assert content.startswith(batch_num), (
                                    "Data corruption detected: final history contains "
                                    "messages from multiple batches. This indicates "
                                    "incomplete locking protection."
                                )

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions_is_public_method(self) -> None:
        """Verify cleanup_expired_sessions is a public method, not private.

        Task 3.15: The cleanup_expired_sessions method must be public so it can
        be called from the lifespan manager and other external code.

        RED PHASE: This test will fail initially because the method is currently
        named _cleanup_expired_sessions (private).
        """
        store = InMemorySessionStore(session_ttl=1)  # 1 second TTL

        # Add a session
        await store.save_history(
            "test-session", [ModelRequest(parts=[UserPromptPart(content="Test message")])]
        )

        # Wait for session to expire
        import time

        time.sleep(1.1)

        # Call the public cleanup_expired_sessions method
        removed_count = await store.cleanup_expired_sessions()

        # Should have removed 1 expired session
        assert removed_count == 1

        # Session should be gone
        history = await store.get_history("test-session")
        assert history == []

    @pytest.mark.asyncio
    async def test_max_sessions_limit_evicts_lru(self) -> None:
        """Verify that max_sessions limit evicts least-recently-used session.

        Task 3.16: When the number of sessions exceeds max_sessions, the session
        with the oldest _last_access time should be evicted.

        RED PHASE: This test will fail initially because max_sessions limiting
        is not yet implemented.
        """
        # Create store with max_sessions=3
        store = InMemorySessionStore(max_sessions=3)

        # Add 3 sessions (at the limit)
        await store.save_history(
            "session-1", [ModelRequest(parts=[UserPromptPart(content="Message 1")])]
        )
        await store.save_history(
            "session-2", [ModelRequest(parts=[UserPromptPart(content="Message 2")])]
        )
        await store.save_history(
            "session-3", [ModelRequest(parts=[UserPromptPart(content="Message 3")])]
        )

        # All 3 sessions should exist
        assert len(await store.get_history("session-1")) == 1
        assert len(await store.get_history("session-2")) == 1
        assert len(await store.get_history("session-3")) == 1

        # Add a 4th session (exceeds limit)
        # This should evict session-1 (oldest _last_access)
        await store.save_history(
            "session-4", [ModelRequest(parts=[UserPromptPart(content="Message 4")])]
        )

        # session-1 should be evicted (LRU)
        assert await store.get_history("session-1") == []

        # Other sessions should still exist
        assert len(await store.get_history("session-2")) == 1
        assert len(await store.get_history("session-3")) == 1
        assert len(await store.get_history("session-4")) == 1

    @pytest.mark.asyncio
    async def test_max_sessions_limit_edge_case_one(self) -> None:
        """Verify max_sessions=1 edge case works correctly.

        Task 3.16: With max_sessions=1, only one session should exist at a time.

        RED PHASE: This test will fail initially because max_sessions limiting
        is not yet implemented.
        """
        store = InMemorySessionStore(max_sessions=1)

        # Add first session
        await store.save_history(
            "only-session", [ModelRequest(parts=[UserPromptPart(content="First")])]
        )
        assert len(await store.get_history("only-session")) == 1

        # Add second session (should evict first)
        await store.save_history(
            "new-session", [ModelRequest(parts=[UserPromptPart(content="Second")])]
        )

        # First session should be evicted
        assert await store.get_history("only-session") == []
        # Second session should exist
        assert len(await store.get_history("new-session")) == 1
