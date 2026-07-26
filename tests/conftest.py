"""Shared pytest fixtures for all tests."""

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import Request
from httpx import ASGITransport
from httpx import AsyncClient
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.config import Settings


# Some test modules still import the module-level `app.main.app` singleton
# directly at module scope (executed at collection time, before any fixture
# runs). Set minimal environment variables so that import keeps working; the
# `client` fixture below builds its own app via `create_app()` and does not
# depend on this.
if "API_KEY" not in os.environ:
    os.environ["API_KEY"] = "test-api-key-12345"
if "LLM_MODEL" not in os.environ:
    os.environ["LLM_MODEL"] = "openai:gpt-4"
if "LLM_API_KEY" not in os.environ:
    os.environ["LLM_API_KEY"] = "test-llm-key-12345"

from app.main import create_app


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear get_settings cache after each test to prevent pollution.

    The get_settings() function uses @cache, so settings are
    cached globally. Tests use monkeypatch to set different environment variables,
    but without clearing the cache, one test's settings could leak into another.

    This fixture runs automatically after every test (autouse=True) to ensure
    test isolation.
    """
    # Yield first to let the test run
    yield

    # Clear the cache after the test completes
    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clear_workflow_cache():
    """Clear workflow cache before and after each test to prevent pollution.

    The _workflow_cache dict in app/deps/workflow.py is a module-level
    global that persists between tests. Without clearing it, tests can see stale
    entries from previous tests, especially if Python reuses id() values after GC.

    Also clear the _get_cached_model LRU cache to prevent settings
    pollution. When tests change LLM settings via monkeypatch, the @lru_cache
    decorator on _get_cached_model() causes the old model to persist, leading
    to tests inadvertently sharing configuration.

    This fixture runs automatically (autouse=True) before and after every test
    to ensure test isolation.
    """
    from app.deps import workflow as wf

    # Clear before test runs
    wf._workflow_cache.clear()
    wf._get_cached_model.cache_clear()
    yield
    # Clear after test completes
    wf._workflow_cache.clear()
    wf._get_cached_model.cache_clear()


@pytest.fixture(autouse=True)
def test_env(monkeypatch):
    """Set up test environment variables for all tests.

    This fixture runs automatically (autouse=True) and provides minimal
    valid configuration to prevent startup failures.

    Note: LLM_API_KEY is set by default to support most tests. Individual tests
    that need to verify cloud provider validation should explicitly unset it
    using monkeypatch.delenv("LLM_API_KEY").

    Fixed misleading comment - the fixture DOES set LLM_API_KEY.
    """
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")  # Set for most tests
    # Disable Logfire in tests
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)


@pytest.fixture
def test_api_key() -> str:
    """Provide test API key for authenticated requests.

    Returns:
        Test API key matching the value in test_env fixture.
    """
    return "test-api-key-12345"


@pytest.fixture
def auth_headers(test_api_key: str) -> dict[str, str]:
    """Provide authentication headers for E2E tests.

    Args:
        test_api_key: Test API key from fixture.

    Returns:
        Headers dictionary with X-API-Key header.
    """
    return {"X-API-Key": test_api_key}


def simple_llm_function(messages: list, agent_info: AgentInfo) -> ModelResponse:
    """Simple LLM function for testing that returns predictable responses.

    Args:
        messages: List of ModelMessage objects.
        agent_info: Agent information.

    Returns:
        ModelResponse with canned response for testing.
    """
    # Extract the last user message content
    user_messages = [
        msg.parts[0].content
        for msg in messages
        if hasattr(msg, "parts") and msg.parts and msg.parts[0].part_kind == "user-prompt"
    ]

    if user_messages:
        last_message = user_messages[-1].lower()

        # Detect RAG evaluation prompts and return "relevant"
        is_evaluation = "respond with exactly one word" in last_message
        has_relevance = "relevant" in last_message or "insufficient" in last_message
        if is_evaluation and has_relevance:
            return ModelResponse(parts=[TextPart(content="relevant")])

        # Detect RAG synthesis prompts and return a contextual answer
        is_synthesis = (
            "using the following context" in last_message
            or "provide a clear and concise answer" in last_message
        )
        if is_synthesis:
            # Extract query from XML tags
            if "<query>" in last_message:
                query_start = last_message.find("<query>") + 7
                query_end = last_message.find("</query>", query_start)
                if query_end != -1:
                    query = last_message[query_start:query_end].strip()

                    # Extract context to provide more realistic answers
                    context = ""
                    if "<context>" in last_message:
                        context_start = last_message.find("<context>") + 9
                        context_end = last_message.find("</context>", context_start)
                        if context_end != -1:
                            context = last_message[context_start:context_end].strip()

                    # Generate contextual answer based on query content
                    if "fastapi" in query.lower():
                        content = (
                            "FastAPI is a modern, fast web framework for building APIs with Python."
                        )
                    elif context:
                        # Use context to generate relevant answer
                        content = f"Based on the provided context, {query}"
                    else:
                        content = f"Based on the available information, {query}"
                    return ModelResponse(parts=[TextPart(content=content)])
            content = "Based on the provided context, here is the answer."
            return ModelResponse(parts=[TextPart(content=content)])

        # Default response for other prompts
        return ModelResponse(parts=[TextPart(content=f"Test response to: {last_message[:50]}")])
    return ModelResponse(parts=[TextPart(content="Test response")])


async def simple_llm_stream_function(messages: list, agent_info: AgentInfo):
    """Simple LLM stream function for testing streaming responses.

    Args:
        messages: List of ModelMessage objects.
        agent_info: Agent information.

    Yields:
        Text chunks for streaming response.
    """
    # Extract the last user message content
    user_messages = [
        msg.parts[0].content
        for msg in messages
        if hasattr(msg, "parts") and msg.parts and msg.parts[0].part_kind == "user-prompt"
    ]

    if user_messages:
        last_message = user_messages[-1]
        # Stream response in chunks
        response = f"Test response to: {last_message[:50]}"
    else:
        response = "Test response"

    # Yield response in chunks (simulating streaming)
    chunk_size = 10
    for i in range(0, len(response), chunk_size):
        yield response[i : i + chunk_size]


@pytest.fixture
def test_model() -> FunctionModel:
    """Provide a FunctionModel for testing without real LLM calls.

    Returns:
        FunctionModel configured with simple_llm_function and stream support.
    """
    return FunctionModel(simple_llm_function, stream_function=simple_llm_stream_function)


def build_test_settings(**overrides: object) -> Settings:
    """Build a `Settings` instance for tests without relying on environment variables.

    Passing explicit field values to `Settings(...)` bypasses environment/`.env`
    lookups for those fields, so callers can construct isolated configurations
    (e.g. per-test CORS origins) without monkeypatching `os.environ`.

    Args:
        **overrides: Field overrides layered on top of the minimal valid defaults.

    Returns:
        A validated `Settings` instance.
    """
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "openai:gpt-4",
        "llm_api_key": "test-llm-key-12345",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest_asyncio.fixture
async def client(test_model: FunctionModel) -> AsyncIterator[AsyncClient]:
    """Provide an async HTTP client for E2E tests.

    Builds an isolated app via `create_app(settings=..., model=test_model)` so the
    chat agent is wired directly with the `FunctionModel`, without needing to
    monkeypatch environment variables or post-hoc override `app.state.chat_agent`.
    The RAG workflow dependency is separately overridden to also use `test_model`.

    Args:
        test_model: FunctionModel fixture for testing.

    Yields:
        AsyncClient configured for testing.
    """
    from app.deps.workflow import get_rag_workflow
    from app.workflows.corrective_rag import CorrectiveRAGWorkflow

    test_app = create_app(settings=build_test_settings(), model=test_model)

    def get_test_rag_workflow(req: Request) -> CorrectiveRAGWorkflow:
        """Test version of get_rag_workflow that uses FunctionModel."""
        return CorrectiveRAGWorkflow(
            vector_store=req.app.state.vector_store,
            llm_settings=req.app.state.settings,
            llm_model=test_model,  # Inject test model here
        )

    test_app.dependency_overrides[get_rag_workflow] = get_test_rag_workflow

    async with (
        test_app.router.lifespan_context(test_app),
        AsyncClient(
            transport=ASGITransport(app=test_app),
            base_url="http://test",
        ) as test_client,
    ):
        yield test_client
