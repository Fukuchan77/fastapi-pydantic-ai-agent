"""E2E tests for API key authentication.

Tests that X-API-Key header is required for /v1/ endpoints but not for /health.
"""

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

        # Assert: Response should have error message
        data = response.json()
        assert "message" in data or "detail" in data, (
            "Error response should have message or detail field"
        )

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
