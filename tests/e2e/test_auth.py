"""E2E tests for API key authentication.

Tests that X-API-Key header is required for /v1/ endpoints but not for /health.
"""

import httpx
import pytest
from httpx import AsyncClient


class TestAuthentication:
    """E2E tests for X-API-Key authentication."""

    @pytest.mark.asyncio
    async def test_v1_endpoint_requires_api_key(self, client: AsyncClient) -> None:
        """Requests to /v1/ endpoints without API key should return 401."""
        # Act: Request a /v1/ endpoint WITHOUT X-API-Key header
        response = await client.post(
            "/v1/agent/chat",
            json={"message": "Hello"},
        )

        # Assert: Should return 401 Unauthorized
        assert response.status_code == 401, "Requests without X-API-Key should be rejected with 401"

        # Assert: flat error envelope (Req 8.1, 8.4) - message at top level,
        # no legacy nested 'detail' key.
        data = response.json()
        assert "message" in data, "Error response should have a message field"
        assert "detail" not in data, "Error response must not nest a legacy 'detail' key"

    @pytest.mark.asyncio
    async def test_v1_endpoint_with_valid_api_key(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Requests to /v1/ endpoints with valid API key should succeed."""
        # Act: Request a /v1/ endpoint WITH valid X-API-Key header
        response = await client.post(
            "/v1/agent/chat",
            json={"message": "Hello"},
            headers=auth_headers,
        )

        # Assert: Should NOT return 401 (might be 200, 422, etc. but not auth error)
        assert response.status_code != 401, "Requests with valid X-API-Key should not return 401"

    @pytest.mark.asyncio
    async def test_v1_endpoint_with_invalid_api_key(self, client: AsyncClient) -> None:
        """Requests to /v1/ endpoints with invalid API key should return 401."""
        # Act: Request a /v1/ endpoint with WRONG X-API-Key
        response = await client.post(
            "/v1/agent/chat",
            json={"message": "Hello"},
            headers={"X-API-Key": "wrong-api-key-12345"},
        )

        # Assert: Should return 401 Unauthorized
        assert response.status_code == 401, (
            "Requests with invalid X-API-Key should be rejected with 401"
        )

    @pytest.mark.asyncio
    async def test_health_endpoint_no_auth_required(self, client: AsyncClient) -> None:
        """Health endpoint should work without authentication."""
        # Act: Request /health WITHOUT X-API-Key header
        response = await client.get("/health")

        # Assert: Should succeed (200 OK)
        assert response.status_code == 200, "/health endpoint should work without authentication"

    @pytest.mark.asyncio
    async def test_rag_ingest_requires_api_key(self, client: AsyncClient) -> None:
        """RAG ingest endpoint should require authentication."""
        # Act: Request /v1/rag/ingest WITHOUT X-API-Key header
        response = await client.post(
            "/v1/rag/ingest",
            json={"chunks": ["test chunk"]},
        )

        # Assert: Should return 401 Unauthorized
        assert response.status_code == 401, (
            "RAG ingest without X-API-Key should be rejected with 401"
        )

    @pytest.mark.asyncio
    async def test_rag_query_requires_api_key(self, client: AsyncClient) -> None:
        """RAG query endpoint should require authentication."""
        # Act: Request /v1/rag/query WITHOUT X-API-Key header
        response = await client.post(
            "/v1/rag/query",
            json={"query": "test query"},
        )

        # Assert: Should return 401 Unauthorized
        assert response.status_code == 401, (
            "RAG query without X-API-Key should be rejected with 401"
        )

    @pytest.mark.asyncio
    async def test_agent_stream_requires_api_key(self, client: AsyncClient) -> None:
        """Agent stream endpoint should require authentication."""
        # Act: Request /v1/agent/stream WITHOUT X-API-Key header
        response = await client.post(
            "/v1/agent/stream",
            json={"message": "Hello"},
        )

        # Assert: Should return 401 Unauthorized
        assert response.status_code == 401, (
            "Agent stream without X-API-Key should be rejected with 401"
        )

    @pytest.mark.asyncio
    async def test_all_rejection_reasons_yield_identical_response(
        self, client: AsyncClient
    ) -> None:
        """Non-ASCII, empty, absent, and wrong-ASCII keys are indistinguishable over HTTP.

        A fixed X-Request-ID is sent on every request so the middleware echoes
        the same value back, keeping response headers directly comparable
        (Req 5.6).
        """
        fixed_request_id = b"fixed-test-request-id"

        # An absent header is the "no X-API-Key at all" case; the others each
        # set X-API-Key to a byte sequence that must be rejected identically.
        header_variants: list[dict[bytes, bytes] | None] = [
            None,
            {b"X-API-Key": b""},
            {b"X-API-Key": "wrong-key-é".encode("latin-1")},
            {b"X-API-Key": b"wrong-ascii-key"},
        ]

        responses = []
        for variant in header_variants:
            headers: dict[bytes, bytes] = {b"X-Request-ID": fixed_request_id}
            if variant:
                headers.update(variant)
            response = await client.post(
                "/v1/agent/chat",
                json={"message": "Hello"},
                headers=headers,
            )
            assert response.status_code < 500, (
                f"expected no server error for headers={variant}, got {response.status_code}"
            )
            responses.append(response)

        # x-ratelimit-remaining decrements on every request regardless of outcome
        # (it counts requests against the window, not rejection reasons), so it
        # is excluded from the "identical response" comparison on purpose.
        def comparable_headers(response: httpx.Response) -> dict[str, str]:
            return {
                key: value
                for key, value in response.headers.items()
                if key.lower() != "x-ratelimit-remaining"
            }

        first = responses[0]
        assert all(response.status_code == 401 for response in responses)
        assert all(response.json() == first.json() for response in responses)
        assert all(
            comparable_headers(response) == comparable_headers(first) for response in responses
        )
