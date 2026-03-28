"""Integration tests for chat agent session persistence.

Tests multi-turn conversations with session history using real stores
and FunctionModel LLM. No HTTP server involved.
"""

import httpx
import pytest
from pydantic import SecretStr
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.agents.chat_agent import build_chat_agent
from app.agents.deps import AgentDeps
from app.config import Settings
from app.stores.session_store import InMemorySessionStore


@pytest.fixture
def settings() -> Settings:
    """Provide test settings with valid LLM configuration."""
    return Settings(
        api_key=SecretStr("test-api-key-12345"),
        llm_model="openai:gpt-4",
        llm_api_key=SecretStr("test-llm-key-12345"),
    )


@pytest.fixture
def session_store() -> InMemorySessionStore:
    """Provide a fresh in-memory session store for each test."""
    return InMemorySessionStore()


@pytest.fixture
def mock_llm() -> FunctionModel:
    """Provide FunctionModel that returns predictable responses."""

    def mock_response(messages: list, info: AgentInfo) -> ModelResponse:
        """Mock LLM that echoes the user message with context."""
        # Extract user message from the messages list
        user_messages = [
            msg.parts[0].content
            for msg in messages
            if hasattr(msg, "parts") and msg.parts and msg.parts[0].part_kind == "user-prompt"
        ]

        if user_messages:
            last_message = user_messages[-1]
            # Return a response that acknowledges the message
            return ModelResponse(parts=[TextPart(content=f"I received: {last_message}")])
        else:
            return ModelResponse(parts=[TextPart(content="Hello!")])

    return FunctionModel(mock_response)


class TestAgentSessionPersistence:
    """Tests for agent session history persistence across turns."""

    @pytest.mark.asyncio
    async def test_session_history_persists_across_turns(
        self,
        mock_llm: FunctionModel,
        settings: Settings,
        session_store: InMemorySessionStore,
        monkeypatch,
    ) -> None:
        """Session history should persist across multiple chat turns."""
        # Arrange: Set environment variables for Settings
        monkeypatch.setenv("API_KEY", settings.api_key.get_secret_value())
        monkeypatch.setenv("LLM_MODEL", settings.llm_model)
        llm_api_key_value = (
            settings.llm_api_key.get_secret_value()
            if settings.llm_api_key
            else "test-api-key-12345"
        )
        monkeypatch.setenv("LLM_API_KEY", llm_api_key_value)

        # Build agent with mock LLM
        agent = build_chat_agent(model=mock_llm)
        session_id = "test-session-1"

        async with httpx.AsyncClient() as client:
            agent_deps = AgentDeps(
                http_client=client,
                settings=settings,
                session_store=session_store,
            )

            # Act: First turn
            result1 = await agent.run(
                "Hello, my name is Alice",
                deps=agent_deps,
            )

            # Save history after first turn
            await session_store.save_history(session_id, result1.all_messages())

            # Load history for second turn
            history = await session_store.get_history(session_id)

            # Second turn with history
            result2 = await agent.run(
                "What is my name?",
                deps=agent_deps,
                message_history=history,
            )

            # Save history after second turn
            await session_store.save_history(session_id, result2.all_messages())

            # Assert: Second turn should have access to first turn's messages
            final_history = await session_store.get_history(session_id)

            # Verify history is not empty and contains messages
            assert len(final_history) > 0, "History should not be empty"

            # Verify that final history is larger than first turn history
            # (proving that messages accumulated across turns)
            assert len(final_history) > len(history), (
                "Final history should have more messages than first turn history"
            )

            # Verify message content from both turns is preserved
            history_str = str(final_history)
            assert "Alice" in history_str or "name" in history_str.lower(), (
                "History should contain content from the chat turns"
            )

    @pytest.mark.asyncio
    async def test_different_sessions_have_isolated_history(
        self,
        mock_llm: FunctionModel,
        settings: Settings,
        session_store: InMemorySessionStore,
        monkeypatch,
    ) -> None:
        """Different session IDs should have isolated conversation history."""
        # Arrange: Set environment variables
        monkeypatch.setenv("API_KEY", settings.api_key.get_secret_value())
        monkeypatch.setenv("LLM_MODEL", settings.llm_model)
        llm_api_key_value = (
            settings.llm_api_key.get_secret_value()
            if settings.llm_api_key
            else "test-api-key-12345"
        )
        monkeypatch.setenv("LLM_API_KEY", llm_api_key_value)

        # Build agent
        agent = build_chat_agent(model=mock_llm)
        session_id_1 = "session-alice"
        session_id_2 = "session-bob"

        async with httpx.AsyncClient() as client:
            agent_deps = AgentDeps(
                http_client=client,
                settings=settings,
                session_store=session_store,
            )

            # Act: Chat in first session
            result1 = await agent.run(
                "My name is Alice",
                deps=agent_deps,
            )
            await session_store.save_history(session_id_1, result1.all_messages())

            # Chat in second session
            result2 = await agent.run(
                "My name is Bob",
                deps=agent_deps,
            )
            await session_store.save_history(session_id_2, result2.all_messages())

            # Assert: Each session should have only its own messages
            history1 = await session_store.get_history(session_id_1)
            history2 = await session_store.get_history(session_id_2)

            # Verify histories are different
            assert history1 != history2, "Different sessions should have different histories"

            # Verify session 1 contains Alice but not Bob
            history1_str = str(history1)
            assert "Alice" in history1_str, "Session 1 should contain Alice"
            assert "Bob" not in history1_str, "Session 1 should NOT contain Bob"

            # Verify session 2 contains Bob but not Alice
            history2_str = str(history2)
            assert "Bob" in history2_str, "Session 2 should contain Bob"
            assert "Alice" not in history2_str, "Session 2 should NOT contain Alice"

    @pytest.mark.asyncio
    async def test_empty_session_returns_empty_history(
        self,
        session_store: InMemorySessionStore,
    ) -> None:
        """Non-existent session should return empty history."""
        # Act: Request history for non-existent session
        history = await session_store.get_history("non-existent-session")

        # Assert: Should return empty list
        assert history == [], "Non-existent session should return empty history"
        assert isinstance(history, list), "History should be a list"

    @pytest.mark.asyncio
    async def test_session_history_ordering(
        self,
        mock_llm: FunctionModel,
        settings: Settings,
        session_store: InMemorySessionStore,
        monkeypatch,
    ) -> None:
        """Session history should preserve message ordering."""
        # Arrange: Set environment variables
        monkeypatch.setenv("API_KEY", settings.api_key.get_secret_value())
        monkeypatch.setenv("LLM_MODEL", settings.llm_model)
        llm_api_key_value = (
            settings.llm_api_key.get_secret_value()
            if settings.llm_api_key
            else "test-api-key-12345"
        )
        monkeypatch.setenv("LLM_API_KEY", llm_api_key_value)

        # Build agent
        agent = build_chat_agent(model=mock_llm)
        session_id = "ordered-session"

        async with httpx.AsyncClient() as client:
            agent_deps = AgentDeps(
                http_client=client,
                settings=settings,
                session_store=session_store,
            )

            # Act: Multiple sequential turns
            messages = ["First message", "Second message", "Third message"]

            for i, msg in enumerate(messages):
                if i == 0:
                    # First turn - no history
                    result = await agent.run(msg, deps=agent_deps)
                else:
                    # Subsequent turns - load history
                    history = await session_store.get_history(session_id)
                    result = await agent.run(msg, deps=agent_deps, message_history=history)

                # Save after each turn
                await session_store.save_history(session_id, result.all_messages())

            # Assert: History should maintain order
            final_history = await session_store.get_history(session_id)

            # Extract user messages in order
            user_messages: list[str] = []
            for msg in final_history:
                if hasattr(msg, "parts") and msg.parts:
                    for part in msg.parts:
                        if part.part_kind == "user-prompt" and isinstance(part.content, str):
                            user_messages.append(part.content)

            # Verify all messages present and in order
            assert len(user_messages) == 3, "Should have all 3 user messages"
            assert user_messages[0] == "First message"
            assert user_messages[1] == "Second message"
            assert user_messages[2] == "Third message"

    @pytest.mark.asyncio
    async def test_session_clear_removes_history(
        self,
        mock_llm: FunctionModel,
        settings: Settings,
        session_store: InMemorySessionStore,
        monkeypatch,
    ) -> None:
        """Clearing a session should remove all history."""
        # Arrange: Set environment variables
        monkeypatch.setenv("API_KEY", settings.api_key.get_secret_value())
        monkeypatch.setenv("LLM_MODEL", settings.llm_model)
        llm_api_key_value = (
            settings.llm_api_key.get_secret_value()
            if settings.llm_api_key
            else "test-api-key-12345"
        )
        monkeypatch.setenv("LLM_API_KEY", llm_api_key_value)

        # Create session with history
        agent = build_chat_agent(model=mock_llm)
        session_id = "clear-test-session"

        async with httpx.AsyncClient() as client:
            agent_deps = AgentDeps(
                http_client=client,
                settings=settings,
                session_store=session_store,
            )

            result = await agent.run("Test message", deps=agent_deps)
            await session_store.save_history(session_id, result.all_messages())

            # Verify history exists
            history_before = await session_store.get_history(session_id)
            assert len(history_before) > 0, "History should exist before clear"

            # Act: Clear the session
            await session_store.clear(session_id)

            # Assert: History should be empty after clear
            history_after = await session_store.get_history(session_id)
            assert history_after == [], "History should be empty after clear"
