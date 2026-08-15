"""Regression tests for oversized chunks on `POST /v1/rag/ingest`.

`IngestRequest.chunks` constrained only the *list* length (1-1000), never the
size of an individual chunk, while `InMemoryVectorStore.add_documents()` raises
`ValueError` past `max_chunk_size` (100,000 characters by default). No handler
converted that, so a caller-supplied chunk one character over the limit produced
a 500 with a logged traceback - reachable through the 10 MB request-body ceiling
and inconsistent with Req 8, which puts every other client input error in the
flat 422 envelope.

Two layers are covered here: the wire contract (`IngestRequest`) and the route's
fallback for a store configured with a tighter limit than the contract.
"""

import pytest
from httpx import ASGITransport
from httpx import AsyncClient
from pydantic import ValidationError
from pydantic_ai.models.function import FunctionModel

from app.main import create_app
from app.models.rag import MAX_CHUNK_CHARS
from app.models.rag import IngestRequest
from app.stores.vector_store import InMemoryVectorStore
from tests.conftest import build_test_settings


class TestIngestRequestValidation:
    """The wire contract rejects oversized chunks before any store is touched."""

    def test_chunk_at_limit_accepted(self) -> None:
        """A chunk exactly at the ceiling is valid (boundary)."""
        request = IngestRequest(chunks=["x" * MAX_CHUNK_CHARS])
        assert len(request.chunks[0]) == MAX_CHUNK_CHARS

    def test_chunk_over_limit_rejected(self) -> None:
        """One character over the ceiling fails validation."""
        with pytest.raises(ValidationError, match=r"(?i)maximum is"):
            IngestRequest(chunks=["x" * (MAX_CHUNK_CHARS + 1)])

    def test_error_names_the_offending_index(self) -> None:
        """The message points at which chunk was too large."""
        with pytest.raises(ValidationError, match=r"index 1"):
            IngestRequest(chunks=["ok", "x" * (MAX_CHUNK_CHARS + 1)])


class TestIngestEndpoint:
    """End-to-end status codes for the ingest route."""

    @pytest.mark.asyncio
    async def test_oversized_chunk_returns_422_not_500(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """A chunk past the wire limit is a client error, not a server error."""
        response = await client.post(
            "/v1/rag/ingest",
            json={"chunks": ["x" * (MAX_CHUNK_CHARS + 1)]},
            headers=auth_headers,
        )

        assert response.status_code == 422, (
            f"Oversized chunk should be a client error, got {response.status_code}"
        )
        body = response.json()
        assert "message" in body, "Error must use the flat ErrorResponse envelope"
        assert "detail" not in body, "Flat envelope must not nest under 'detail'"

    @pytest.mark.asyncio
    async def test_store_level_limit_also_returns_422(
        self, test_model: FunctionModel, auth_headers: dict[str, str]
    ) -> None:
        """A store configured tighter than the wire contract still yields 422.

        Builds its own app rather than using the shared `client` fixture, so the
        store can be swapped for one with a limit below `MAX_CHUNK_CHARS` after
        lifespan startup has populated `app.state`.
        """
        test_app = create_app(settings=build_test_settings(), model=test_model)

        async with (
            test_app.router.lifespan_context(test_app),
            AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://test"
            ) as http_client,
        ):
            test_app.state.vector_store = InMemoryVectorStore(max_chunk_size=100)

            response = await http_client.post(
                "/v1/rag/ingest",
                json={"chunks": ["x" * 500]},
                headers=auth_headers,
            )

        assert response.status_code == 422
        assert response.json()["code"] == "INVALID_DOCUMENT_CHUNK"

    @pytest.mark.asyncio
    async def test_normal_chunk_still_ingests(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """The happy path is unaffected."""
        response = await client.post(
            "/v1/rag/ingest",
            json={"chunks": ["a reasonably sized document chunk"]},
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.json()["ingested"] == 1
