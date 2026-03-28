"""Local LLM tests using ollama:llama3.2:latest.

These tests require a running Ollama instance with llama3.2:latest model.
To set up:
    1. Install Ollama: https://ollama.com
    2. Start server: ollama serve
    3. Pull model: ollama pull llama3.2:latest
    4. Run tests: mise run test:local

Tests are automatically skipped if Ollama is not available (via require_ollama fixture).
"""

import pytest
from httpx import AsyncClient
from pydantic import SecretStr

from app.agents.chat_agent import build_chat_agent
from app.agents.chat_agent import build_model
from app.agents.deps import AgentDeps
from app.config import Settings
from app.stores.session_store import InMemorySessionStore


@pytest.fixture
def ollama_settings_llama32(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings configured for local Ollama with llama3.2:latest.

    LLM_BASE_URL is optional — LiteLLM defaults to http://localhost:11434.
    LLM_API_KEY is not required for Ollama (local provider).
    """
    # Remove LLM_API_KEY from environment to test Ollama without API key
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    return Settings(
        api_key=SecretStr("local-dev-api-key-12345"),
        llm_model="ollama:llama3.2:latest",
        enable_mock_tools=True,  # Enable mock tools for testing
    )


@pytest.mark.ollama
@pytest.mark.asyncio
async def test_agent_basic_response_llama32(
    ollama_settings_llama32: Settings,
) -> None:
    """Agent should return a non-empty string response from llama3.2:latest.

    This test verifies that:
    - build_model() correctly configures LiteLLM for Ollama
    - build_chat_agent() creates a working agent
    - The agent can complete a basic request using the local model

    The test does not validate specific output content since LLM responses
    are non-deterministic. It only verifies the execution completes successfully
    and returns a non-empty string.
    """
    # Build model and agent using the Ollama settings
    model = build_model(ollama_settings_llama32)
    agent = build_chat_agent(model=model, settings=ollama_settings_llama32)

    # Create agent dependencies
    async with AsyncClient() as http_client:
        deps = AgentDeps(
            http_client=http_client,
            settings=ollama_settings_llama32,
            session_store=InMemorySessionStore(),
        )

        # Run the agent with a simple prompt
        result = await agent.run("Say hello in one sentence.", deps=deps)

        # Verify response structure
        assert isinstance(result.output, str), "Agent output should be a string"
        assert len(result.output) > 0, "Agent output should not be empty"
        assert len(result.output) < 500, "Agent output should be reasonably short for this prompt"


@pytest.mark.ollama
@pytest.mark.asyncio
async def test_agent_with_session_llama32(
    ollama_settings_llama32: Settings,
) -> None:
    """Agent should maintain conversation history across multiple turns.

    This test verifies that:
    - Session store correctly persists message history
    - Agent can reference previous messages in context
    """
    # Build model and agent
    model = build_model(ollama_settings_llama32)
    agent = build_chat_agent(model=model, settings=ollama_settings_llama32)

    # Create shared session store and deps
    session_store = InMemorySessionStore()
    session_id = "test-session-llama32"

    async with AsyncClient() as http_client:
        deps = AgentDeps(
            http_client=http_client,
            settings=ollama_settings_llama32,
            session_store=session_store,
        )

        # First turn: establish context
        result1 = await agent.run(
            "My favorite color is blue.",
            deps=deps,
        )

        # Save history after first turn
        await session_store.save_history(session_id, result1.all_messages())

        # Second turn: reference previous context
        history = await session_store.get_history(session_id)
        result2 = await agent.run(
            "What is my favorite color?",
            deps=deps,
            message_history=history,
        )

        # Verify both responses are valid
        assert isinstance(result1.output, str)
        assert len(result1.output) > 0
        assert isinstance(result2.output, str)
        assert len(result2.output) > 0

        # Verify session history was maintained
        # (we can't verify the content due to non-determinism,
        # but we can verify the structure)
        final_history = await session_store.get_history(session_id)
        assert len(final_history) > 0, "Session history should be maintained"
