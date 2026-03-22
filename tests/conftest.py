"""Shared pytest fixtures for all tests."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from fastapi import Request
from httpx import ASGITransport
from httpx import AsyncClient
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.main import app


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear get_settings cache after each test to prevent pollution.

    The get_settings() function uses @lru_cache(maxsize=1), so settings are
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
def test_env(monkeypatch):
    """Set up test environment variables for all tests.

    This fixture runs automatically (autouse=True) and provides minimal
    valid configuration to prevent startup failures.

    Note: LLM_API_KEY is intentionally NOT set here to allow tests to
    verify cloud provider validation. Individual tests should set it
    when testing valid configurations.
    """
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")  # Set for most tests
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
            # Extract query from the prompt
            if "query:" in last_message:
                query_start = last_message.find("query:") + 6
                query_end = last_message.find("\n", query_start)
                if query_end == -1:
                    query_end = last_message.find("context:", query_start)
                query = last_message[query_start:query_end].strip()
                content = f"Based on the provided context, {query}"
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


@asynccontextmanager
async def test_lifespan_override(test_model: FunctionModel):
    """Override lifespan to inject test model.

    This context manager wraps the real lifespan but replaces the
    chat agent and RAG workflow with ones using FunctionModel to avoid real LLM calls.

    Args:
        test_model: FunctionModel to use for testing.

    Yields:
        None: Control during test app lifetime.
    """
    from app.agents.chat_agent import build_chat_agent
    from app.deps.workflow import get_rag_workflow
    from app.main import lifespan as real_lifespan
    from app.workflows.corrective_rag import CorrectiveRAGWorkflow

    # Define test workflow factory that uses test_model
    def get_test_rag_workflow(req: Request):
        """Test version of get_rag_workflow that uses FunctionModel."""
        from app.config import get_settings

        return CorrectiveRAGWorkflow(
            vector_store=req.app.state.vector_store,
            llm_settings=get_settings(),
            llm_model=test_model,  # Inject test model here
        )

    # Run the real lifespan
    async with real_lifespan(app):
        # Override the chat agent with test model
        app.state.chat_agent = build_chat_agent(model=test_model)

        # Override RAG workflow dependency to use test model
        app.dependency_overrides[get_rag_workflow] = get_test_rag_workflow

        yield

        # Clean up overrides
        app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(test_model: FunctionModel) -> AsyncIterator[AsyncClient]:
    """Provide an async HTTP client for E2E tests.

    This fixture creates an AsyncClient with ASGI transport that communicates
    directly with the FastAPI app without starting a real HTTP server.
    The app's lifespan is executed, ensuring proper startup/shutdown.

    The chat agent is overridden with a FunctionModel to avoid real LLM API calls.

    Args:
        test_model: FunctionModel fixture for testing.

    Yields:
        AsyncClient configured for testing.
    """
    # Create a test app with overridden lifespan and async client with ASGI transport
    async with (
        test_lifespan_override(test_model),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as test_client,
    ):
        yield test_client
