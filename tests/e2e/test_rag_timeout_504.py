"""Tests for RAG workflow timeout should return HTTP 504.

Verify that when the RAG workflow times out (exceeds rag_workflow_timeout),
the API returns HTTP 504 Gateway Timeout instead of HTTP 500 Internal Server Error.
"""

import asyncio

import pytest
from fastapi import Request
from httpx import ASGITransport
from httpx import AsyncClient
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.deps.workflow import get_rag_workflow
from app.main import create_app
from app.workflows.corrective_rag import CorrectiveRAGWorkflow
from tests.conftest import build_test_settings


@pytest.mark.asyncio
async def test_rag_query_timeout_returns_504(auth_headers: dict[str, str]) -> None:
    """Test that RAG workflow timeout returns HTTP 504 Gateway Timeout.

    asyncio.TimeoutError from asyncio.timeout() should be caught
    and converted to HTTPException(status_code=504) instead of propagating
    to the global exception handler which returns 500.

    Builds the app through `create_app(settings=..., model=...)` rather than
    importing the module-level singleton, so `app.state` is populated by a
    real lifespan (the singleton's state is otherwise never initialized when
    mutated directly - see plan.md L1.4).
    """
    # Configure very short workflow timeout (5 seconds minimum per Settings validation)
    settings = build_test_settings(
        rag_workflow_timeout=5,  # 5 second timeout (minimum)
        llm_agent_timeout=10,  # longer than the workflow timeout
        llm_retry_max_attempts=1,  # no retries
    )

    # Create a slow model that takes longer than the workflow timeout
    async def slow_model(
        messages: list,
        info: AgentInfo,
    ) -> ModelResponse:
        await asyncio.sleep(15)  # Sleep 15 seconds (exceeds 5 second workflow timeout)
        return ModelResponse(parts=[TextPart(content="relevant")])

    model = FunctionModel(slow_model)
    test_app = create_app(settings=settings, model=model)

    def patched_get_rag_workflow(request: Request) -> CorrectiveRAGWorkflow:
        return CorrectiveRAGWorkflow(
            vector_store=request.app.state.vector_store,
            llm_settings=settings,
            llm_model=model,  # Use the slow model
        )

    test_app.dependency_overrides[get_rag_workflow] = patched_get_rag_workflow

    async with (
        test_app.router.lifespan_context(test_app),
        AsyncClient(
            transport=ASGITransport(app=test_app),
            base_url="http://test",
        ) as client,
    ):
        await client.post(
            "/v1/rag/ingest",
            json={"chunks": ["Test document for timeout"]},
            headers=auth_headers,
        )

        # Make request to RAG endpoint
        response = await client.post(
            "/v1/rag/query",
            json={"query": "test query", "max_retries": 1},
            headers=auth_headers,
        )

    # CRITICAL: Verify HTTP 504 Gateway Timeout is returned (not HTTP 500)
    assert response.status_code == 504, (
        f"Expected HTTP 504 Gateway Timeout, got {response.status_code}. "
        f"Response: {response.json()}"
    )

    # Verify error message indicates timeout
    response_data = response.json()
    assert "timed out" in response_data.get("detail", "").lower()
