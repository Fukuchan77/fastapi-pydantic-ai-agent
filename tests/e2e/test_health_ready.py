"""E2E tests for readiness health check endpoint.

Deep readiness health check at GET /health/ready.
Tests the endpoint through full HTTP stack using AsyncClient.
This endpoint requires no authentication and performs live connectivity
probes against the store/provider selection currently in effect (Req 13.1).
"""

import pytest
from httpx import AsyncClient


class TestReadinessEndpoint:
    """E2E tests for GET /health/ready endpoint."""

    @pytest.mark.asyncio
    async def test_readiness_endpoint_returns_200_when_ready(self, client: AsyncClient) -> None:
        """Readiness endpoint should return HTTP 200 when all dependencies are healthy."""
        # Act: Request readiness endpoint
        response = await client.get("/health/ready")

        # Assert: Should return 200 OK when ready
        assert response.status_code == 200, (
            "Readiness endpoint should return 200 when all dependencies are healthy"
        )

        # Assert: Response should have JSON content with expected structure
        data = response.json()
        assert isinstance(data, dict), "Response should be a JSON object"
        assert "status" in data, "Response should have 'status' field"
        assert data["status"] == "ready", "Status should be 'ready' when healthy"
        assert "checks" in data, "Response should have 'checks' field"
        assert isinstance(data["checks"], dict), "Checks should be a dict"

    @pytest.mark.asyncio
    async def test_readiness_endpoint_no_auth_required(self, client: AsyncClient) -> None:
        """Readiness endpoint should not require authentication."""
        # Act: Request readiness endpoint WITHOUT X-API-Key header
        response = await client.get("/health/ready")

        # Assert: Should succeed without auth (like /health endpoint)
        assert response.status_code == 200, "Readiness endpoint should work without authentication"

    @pytest.mark.asyncio
    async def test_readiness_endpoint_checks_vector_store(self, client: AsyncClient) -> None:
        """Readiness endpoint should include vector_store in checks."""
        # Act: Request readiness endpoint
        response = await client.get("/health/ready")

        # Assert: Should have vector_store check (default test settings: in-memory, skipped)
        data = response.json()
        assert "checks" in data
        assert data["checks"]["vector_store"] in ["healthy", "skipped", "unreachable"]

    @pytest.mark.asyncio
    async def test_readiness_endpoint_checks_session_store(self, client: AsyncClient) -> None:
        """Readiness endpoint should include session_store in checks."""
        # Act: Request readiness endpoint
        response = await client.get("/health/ready")

        # Assert: Should have session_store check (default test settings: in-memory, skipped)
        data = response.json()
        assert "checks" in data
        assert data["checks"]["session_store"] in ["healthy", "skipped", "unreachable"]

    @pytest.mark.asyncio
    async def test_readiness_endpoint_checks_llm_provider(self, client: AsyncClient) -> None:
        """Readiness endpoint should include llm_provider in checks and probe it live."""
        # Act: Request readiness endpoint
        response = await client.get("/health/ready")

        # Assert: Should have llm_provider check; the test FunctionModel always succeeds
        data = response.json()
        assert "checks" in data
        assert data["checks"]["llm_provider"] == "healthy"

    @pytest.mark.asyncio
    async def test_readiness_endpoint_response_has_json_content_type(
        self, client: AsyncClient
    ) -> None:
        """Readiness endpoint should return JSON content type."""
        # Act: Request readiness endpoint
        response = await client.get("/health/ready")

        # Assert: Verify Content-Type
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"], (
            "Response should have JSON content type"
        )

    @pytest.mark.asyncio
    async def test_readiness_endpoint_all_checks_pass_in_normal_operation(
        self, client: AsyncClient
    ) -> None:
        """Readiness endpoint should be ready with default in-memory test settings."""
        # Act: Request readiness endpoint
        response = await client.get("/health/ready")

        # Assert: All checks should be non-failing in the test environment
        data = response.json()
        assert response.status_code == 200
        assert data["status"] == "ready"
        assert data["checks"]["session_store"] == "skipped"
        assert data["checks"]["vector_store"] == "skipped"
        assert data["checks"]["llm_provider"] == "healthy"
