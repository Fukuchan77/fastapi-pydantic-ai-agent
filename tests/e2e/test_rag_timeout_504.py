"""Tests for Task 20.7: RAG workflow timeout should return HTTP 504.

Task 20.7: Verify that when the RAG workflow times out (exceeds rag_workflow_timeout),
the API returns HTTP 504 Gateway Timeout instead of HTTP 500 Internal Server Error.
"""

import pytest
from httpx import ASGITransport
from httpx import AsyncClient
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.config import get_settings
from app.main import app
from app.stores.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_rag_query_timeout_returns_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that RAG workflow timeout returns HTTP 504 Gateway Timeout.

    Task 20.7: asyncio.TimeoutError from asyncio.timeout() should be caught
    and converted to HTTPException(status_code=504) instead of propagating
    to the global exception handler which returns 500.
    """
    # Configure very short workflow timeout (5 seconds minimum per Settings validation)
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("RAG_WORKFLOW_TIMEOUT", "5")  # 5 second timeout (minimum)
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "10")  # 10 second timeout (longer than workflow)
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "1")  # No retries
    get_settings.cache_clear()
    settings = get_settings()

    # Create a slow model that takes longer than the workflow timeout
    async def slow_model(
        messages: list,
        info: AgentInfo,
    ) -> ModelResponse:
        import asyncio

        await asyncio.sleep(15)  # Sleep 15 seconds (exceeds 5 second workflow timeout)
        return ModelResponse(parts=[TextPart(content="relevant")])

    model = FunctionModel(slow_model)

    # Replace vector store and model in app state
    app.state.vector_store = InMemoryVectorStore()
    await app.state.vector_store.add_documents(["Test document for timeout"])

    # Replace the workflow model (this is tricky - need to modify the dependency)
    # For now, we'll use monkeypatch to replace the llm_model in settings
    # Actually, we need to inject the model into the workflow - let me rethink this

    # Alternative: Use monkeypatch to set the model in the workflow factory
    # We need to patch the get_rag_workflow dependency to use our slow model
    from fastapi import Request

    from app.deps.workflow import get_rag_workflow
    from app.workflows.corrective_rag import CorrectiveRAGWorkflow

    def patched_get_rag_workflow(request: Request) -> CorrectiveRAGWorkflow:
        return CorrectiveRAGWorkflow(
            vector_store=request.app.state.vector_store,
            llm_settings=settings,
            llm_model=model,  # Use the slow model
        )

    app.dependency_overrides[get_rag_workflow] = patched_get_rag_workflow

    # Make request to RAG endpoint
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/rag/query",
            json={"query": "test query", "max_retries": 1},
            headers={"X-API-Key": "test-api-key-1234567890"},
        )

    # Clean up dependency overrides
    app.dependency_overrides.clear()

    # CRITICAL: Verify HTTP 504 Gateway Timeout is returned (not HTTP 500)
    # Currently FAILS: Returns HTTP 500 because TimeoutError propagates to global handler
    assert response.status_code == 504, (
        f"Expected HTTP 504 Gateway Timeout, got {response.status_code}. "
        f"Response: {response.json()}"
    )

    # Verify error message indicates timeout
    response_data = response.json()
    assert "timed out" in response_data.get("detail", "").lower()
