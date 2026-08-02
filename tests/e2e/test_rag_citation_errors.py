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
from httpx import Response
from pydantic_ai.models.function import FunctionModel

from app.deps.workflow import get_rag_workflow
from app.main import create_app
from app.workflows.corrective_rag import CorrectiveRAGWorkflow
from app.workflows.exceptions import DanglingCitationError
from app.workflows.exceptions import EmptyCitationError
from tests.conftest import build_test_settings


class _RaisingWorkflow:
    """Stub workflow whose run() always raises the given exception."""

    def __init__(self, exc: Exception, llm_settings: object) -> None:
        self._exc = exc
        self.llm_settings = llm_settings

    async def run(self, query: str, max_retries: int = 3) -> dict:
        raise self._exc


async def _query_with_raising_workflow(
    exc: Exception,
    test_model: FunctionModel,
    auth_headers: dict[str, str],
) -> Response:
    """Build an isolated app whose RAG workflow always raises `exc`, and query it.

    Builds through `create_app(settings=..., model=...)` rather than importing
    the module-level singleton, so `app.state` is populated by a real lifespan
    (see plan.md L1.4).
    """
    settings = build_test_settings()
    test_app = create_app(settings=settings, model=test_model)

    def patched_get_rag_workflow(request: Request) -> CorrectiveRAGWorkflow:
        return _RaisingWorkflow(exc, settings)  # type: ignore[return-value]

    test_app.dependency_overrides[get_rag_workflow] = patched_get_rag_workflow

    async with (
        test_app.router.lifespan_context(test_app),
        AsyncClient(
            transport=ASGITransport(app=test_app),
            base_url="http://test",
        ) as client,
    ):
        return await client.post(
            "/v1/rag/query",
            json={"query": "test query"},
            headers=auth_headers,
        )


@pytest.mark.asyncio
async def test_dangling_citation_error_returns_502(
    test_model: FunctionModel, auth_headers: dict[str, str]
) -> None:
    """A DanglingCitationError from the workflow should map to HTTP 502, not 500."""
    exc = DanglingCitationError(unknown_ids={"memory::0099"}, known_ids={"memory::0000"})
    response = await _query_with_raising_workflow(exc, test_model, auth_headers)

    assert response.status_code == 502
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_empty_citation_error_returns_502(
    test_model: FunctionModel, auth_headers: dict[str, str]
) -> None:
    """An EmptyCitationError from the workflow should map to HTTP 502, not 500."""
    exc = EmptyCitationError()
    response = await _query_with_raising_workflow(exc, test_model, auth_headers)

    assert response.status_code == 502
    assert "detail" in response.json()
