"""E2E tests for health check endpoint.

Tests the /health endpoint through full HTTP stack using AsyncClient.
This endpoint requires no authentication and returns basic service status.
"""

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """E2E tests for GET /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_200(self, client: AsyncClient) -> None:
        """Health endpoint should return HTTP 200 with status ok."""
        # Act: Request health endpoint
        response = await client.get("/health")

        # Assert: Should return 200 OK
        assert response.status_code == 200, "Health endpoint should return 200 OK"

        # Assert: Response should have JSON content
        data = response.json()
        assert isinstance(data, dict), "Response should be a JSON object"
        assert "status" in data, "Response should have 'status' field"
        assert data["status"] == "ok", "Status should be 'ok'"

    @pytest.mark.asyncio
    async def test_health_endpoint_no_auth_required(self, client: AsyncClient) -> None:
        """Health endpoint should not require authentication."""
        # Act: Request health endpoint WITHOUT X-API-Key header
        response = await client.get("/health")

        # Assert: Should succeed without auth (unlike /v1/ endpoints)
        assert response.status_code == 200, "Health endpoint should work without authentication"

    @pytest.mark.asyncio
    async def test_health_endpoint_response_structure(self, client: AsyncClient) -> None:
        """Health endpoint should return well-formed JSON response."""
        # Act: Request health endpoint
        response = await client.get("/health")

        # Assert: Response structure
        assert response.status_code == 200
        data = response.json()

        # Verify minimal required fields
        assert "status" in data, "Response must have 'status' field"
        assert isinstance(data["status"], str), "Status must be a string"

        # Verify Content-Type
        assert "application/json" in response.headers["content-type"], (
            "Response should have JSON content type"
        )
