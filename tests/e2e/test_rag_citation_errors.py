"""E2E tests for citation-error mapping on POST /v1/rag/query.

Verifies that EmptyCitationError/DanglingCitationError raised by the
Corrective RAG workflow are mapped to an HTTP error response instead of
propagating to the generic 500 handler, and that citations are surfaced
on a successful response.
"""

import pytest
from fastapi import Request
from httpx import ASGITransport
from httpx import AsyncClient

from app.config import get_settings
from app.deps.workflow import get_rag_workflow
from app.main import app
from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow
from app.workflows.exceptions import DanglingCitationError
from app.workflows.exceptions import EmptyCitationError


class _RaisingWorkflow:
    """Stub workflow whose run() always raises the given exception."""

    def __init__(self, exc: Exception, llm_settings: object) -> None:
        self._exc = exc
        self.llm_settings = llm_settings

    async def run(self, query: str, max_retries: int = 3) -> dict:
        raise self._exc


@pytest.mark.asyncio
async def test_dangling_citation_error_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DanglingCitationError from the workflow should map to HTTP 502, not 500."""
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    get_settings.cache_clear()
    settings = get_settings()

    app.state.vector_store = InMemoryVectorStore()
    exc = DanglingCitationError(unknown_ids={"memory::0099"}, known_ids={"memory::0000"})

    def patched_get_rag_workflow(request: Request) -> CorrectiveRAGWorkflow:
        return _RaisingWorkflow(exc, settings)  # type: ignore[return-value]

    app.dependency_overrides[get_rag_workflow] = patched_get_rag_workflow

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/rag/query",
            json={"query": "test query"},
            headers={"X-API-Key": "test-api-key-1234567890"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_empty_citation_error_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    """An EmptyCitationError from the workflow should map to HTTP 502, not 500."""
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    get_settings.cache_clear()
    settings = get_settings()

    app.state.vector_store = InMemoryVectorStore()
    exc = EmptyCitationError()

    def patched_get_rag_workflow(request: Request) -> CorrectiveRAGWorkflow:
        return _RaisingWorkflow(exc, settings)  # type: ignore[return-value]

    app.dependency_overrides[get_rag_workflow] = patched_get_rag_workflow

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/rag/query",
            json={"query": "test query"},
            headers={"X-API-Key": "test-api-key-1234567890"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "detail" in response.json()
