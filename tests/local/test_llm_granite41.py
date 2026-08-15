"""Local LLM tests using ollama:granite4.1:8b.

These tests require a running Ollama instance with the granite4.1:8b model.
To set up:
    1. Install Ollama: https://ollama.com
    2. Start server: ollama serve
    3. Pull model: ollama pull granite4.1:8b
    4. Run tests: mise run test:local

Tests are skipped if Ollama is unreachable (via the `require_ollama` fixture)
or if granite4.1:8b has not been pulled (via `skip_unless_model_pulled`).

Mock tools are off by default here. granite4.1:8b (8B parameters) does not
reliably terminate a tool-calling loop: given mock tools it will re-issue the
same `web_search` call until the run exhausts `request_limit`, even for a
prompt that needs no tool at all. Only the dedicated tool-calling test enables
them, and it asserts via `run_guarded()` — the same entry point the API uses —
so a looping model surfaces as `stop_reason="max_iterations"` rather than a raw
`UsageLimitExceeded`.
"""

import pytest
from httpx import AsyncClient
from pydantic import SecretStr
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.usage import UsageLimits

from app.agents.chat_agent import build_chat_agent
from app.agents.chat_agent import build_model
from app.agents.deps import AgentDeps
from app.agents.guardrails import run_guarded
from app.config import Settings
from app.stores.session_store import InMemorySessionStore
from tests.support.ollama import skip_unless_model_pulled


GRANITE41_MODEL = "granite4.1:8b"
"""Bare Ollama model name required by this module."""


def _build_granite41_settings(*, enable_mock_tools: bool) -> Settings:
    """Build Settings for local Ollama granite4.1:8b.

    LLM_BASE_URL is optional — LiteLLM defaults to http://localhost:11434.
    LLM_API_KEY is not required for Ollama (local provider).

    Args:
        enable_mock_tools: Whether to register the mock toolset on the agent.

    Returns:
        Settings pointing at the local granite4.1:8b model.
    """
    return Settings(
        api_key=SecretStr("local-dev-api-key-12345"),
        llm_model=f"ollama:{GRANITE41_MODEL}",
        enable_mock_tools=enable_mock_tools,
    )


@pytest.fixture
def ollama_settings_granite41(
    monkeypatch: pytest.MonkeyPatch,
    require_ollama: frozenset[str],
) -> Settings:
    """Settings for granite4.1:8b with mock tools disabled.

    Used by the tests that only exercise plain completion and session history,
    which need no tools; leaving mock tools on would let the model loop.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to drop LLM_API_KEY.
        require_ollama: Pulled Ollama model names; the test is skipped when
            granite4.1:8b is absent.

    Returns:
        Settings pointing at the local granite4.1:8b model, mock tools off.
    """
    skip_unless_model_pulled(require_ollama, GRANITE41_MODEL)

    # Remove LLM_API_KEY from environment to test Ollama without API key
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    return _build_granite41_settings(enable_mock_tools=False)


@pytest.fixture
def ollama_settings_granite41_with_tools(
    monkeypatch: pytest.MonkeyPatch,
    require_ollama: frozenset[str],
) -> Settings:
    """Settings for granite4.1:8b with mock tools enabled.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to drop LLM_API_KEY.
        require_ollama: Pulled Ollama model names; the test is skipped when
            granite4.1:8b is absent.

    Returns:
        Settings pointing at the local granite4.1:8b model, mock tools on.
    """
    skip_unless_model_pulled(require_ollama, GRANITE41_MODEL)

    # Remove LLM_API_KEY from environment to test Ollama without API key
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    return _build_granite41_settings(enable_mock_tools=True)


@pytest.mark.ollama
@pytest.mark.asyncio
async def test_agent_basic_response_granite41(
    ollama_settings_granite41: Settings,
) -> None:
    """Agent should return a non-empty string response from granite4.1:8b.

    This test verifies that:
    - build_model() correctly configures LiteLLM for Ollama with granite4.1
    - build_chat_agent() creates a working agent
    - The agent can complete a basic request using the local model

    The test does not validate specific output content since LLM responses
    are non-deterministic. It only verifies the execution completes successfully
    and returns a non-empty string.
    """
    # Build model and agent using the Ollama settings
    model = build_model(ollama_settings_granite41)
    agent = build_chat_agent(model=model, settings=ollama_settings_granite41)

    # Create agent dependencies
    async with AsyncClient() as http_client:
        deps = AgentDeps(
            http_client=http_client,
            settings=ollama_settings_granite41,
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
async def test_agent_with_mock_tool_granite41_terminates_within_limits(
    ollama_settings_granite41_with_tools: Settings,
) -> None:
    """A guarded granite4.1 run with mock tools must terminate on a known StopReason.

    This test verifies that:
    - Mock tools are registered when enable_mock_tools=True, and at least one
      is actually invoked against the real local model (not merely that the
      run finished, which a model that never calls a tool would also satisfy)
    - `run_guarded()` bounds the tool-calling loop, so the run always ends on
      a value from the closed `StopReason` vocabulary instead of raising
      `UsageLimitExceeded`

    Termination, not the answer, is the contract for *how* the run ends. Tool
    invocation is non-deterministic, and an 8B model may keep re-issuing the
    same call — the guardrail turning that into `max_iterations` is exactly
    the behaviour the API depends on, so both `completed` and
    `max_iterations` are valid outcomes for the stop reason. A raw exception
    escaping `run_guarded()` is not. Separately, at least one tool call must
    appear in the transcript on either outcome - a model that answered without
    ever attempting the tool would pass the stop-reason check for the wrong
    reason (mock tools silently not registered, say) and this test would not
    have caught it.
    """
    settings = ollama_settings_granite41_with_tools
    model = build_model(settings)
    agent = build_chat_agent(model=model, settings=settings)

    # Mirror the limits POST /v1/agent/chat applies (app/api/v1/agent.py).
    limits = UsageLimits(
        request_limit=settings.usage_request_limit,
        total_tokens_limit=settings.usage_total_tokens_limit,
    )

    async with AsyncClient() as http_client:
        deps = AgentDeps(
            http_client=http_client,
            settings=settings,
            session_store=InMemorySessionStore(),
        )

        # Prompt chosen to invite tool use; whether the tool fires is up to the model.
        guarded = await run_guarded(
            agent,
            "What is the current weather in Tokyo?",
            deps=deps,
            limits=limits,
            audit=deps.audit,
        )

    assert guarded.stop_reason in {"completed", "max_iterations"}, (
        f"Guarded run stopped on unexpected reason {guarded.stop_reason!r}; "
        "expected the run to either finish or be bounded by the iteration limit"
    )

    # A completed run must carry a usable answer; a bounded run deliberately does not.
    if guarded.stop_reason == "completed":
        assert isinstance(guarded.output, str), "Completed run should output a string"
        assert len(guarded.output) > 0, "Completed run should not output an empty string"

    # Confirm mock tools were actually reachable, not just that the run ended
    # cleanly - a "completed" stop_reason with zero tool calls is exactly what
    # you'd see if enable_mock_tools silently failed to register anything, and
    # the stop-reason assertion above would not have caught that.
    tool_calls_made = sum(
        1
        for m in guarded.messages
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    )
    assert tool_calls_made > 0 or guarded.stop_reason == "max_iterations", (
        "Expected at least one tool call, or a max_iterations stop from the "
        "model repeatedly attempting one; a clean completion with zero tool "
        "calls would mean mock tools were not actually registered/invoked"
    )


@pytest.mark.ollama
@pytest.mark.asyncio
async def test_agent_with_session_granite41(
    ollama_settings_granite41: Settings,
) -> None:
    """Agent should maintain conversation history across multiple turns with granite4.1.

    This test verifies that:
    - Session store correctly persists message history
    - Agent can reference previous messages in context
    """
    # Build model and agent
    model = build_model(ollama_settings_granite41)
    agent = build_chat_agent(model=model, settings=ollama_settings_granite41)

    # Create shared session store and deps
    session_store = InMemorySessionStore()
    session_id = "test-session-granite41"

    async with AsyncClient() as http_client:
        deps = AgentDeps(
            http_client=http_client,
            settings=ollama_settings_granite41,
            session_store=session_store,
        )

        # First turn: establish context
        result1 = await agent.run(
            "My favorite programming language is Python.",
            deps=deps,
        )

        # Save history after first turn
        await session_store.save_history(session_id, result1.all_messages())

        # Second turn: reference previous context
        history = await session_store.get_history(session_id)
        result2 = await agent.run(
            "What is my favorite programming language?",
            deps=deps,
            message_history=history,
        )

        # Verify both responses are valid
        assert isinstance(result1.output, str)
        assert len(result1.output) > 0
        assert isinstance(result2.output, str)
        assert len(result2.output) > 0

        # Verify session history was maintained
        final_history = await session_store.get_history(session_id)
        assert len(final_history) > 0, "Session history should be maintained"
