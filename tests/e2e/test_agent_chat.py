"""E2E tests for agent chat endpoint.

Tests the POST /v1/agent/chat endpoint through full HTTP stack using AsyncClient.
This endpoint requires authentication and returns structured chat responses.
"""

import pytest
from httpx import AsyncClient


class TestAgentChatEndpoint:
    """E2E tests for POST /v1/agent/chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_endpoint_basic_request(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Chat endpoint should return valid response for basic request."""
        # Arrange: Basic chat request
        request_data = {"message": "Hello, how are you?"}

        # Act: POST to chat endpoint with auth
        response = await client.post(
            "/v1/agent/chat",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Should return 200 OK
        assert response.status_code == 200, "Chat endpoint should return 200 OK"

        # Assert: Response should match ChatResponse schema
        data = response.json()
        assert isinstance(data, dict), "Response should be a JSON object"
        assert "reply" in data, "Response should have 'reply' field"
        assert "session_id" in data, "Response should have 'session_id' field"
        assert "tool_calls_made" in data, "Response should have 'tool_calls_made' field"

        # Assert: Reply should be non-empty string
        assert isinstance(data["reply"], str), "Reply should be a string"
        assert len(data["reply"]) > 0, "Reply should be non-empty"

        # Assert: Tool calls should be a number
        assert isinstance(data["tool_calls_made"], int), "tool_calls_made should be an integer"
        assert data["tool_calls_made"] >= 0, "tool_calls_made should be non-negative"

    @pytest.mark.asyncio
    async def test_chat_endpoint_with_session_id(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Chat endpoint should handle session_id for conversation history."""
        # Arrange: Request with session_id
        session_id = "test-session-123"
        request_data = {
            "message": "Remember my name is Alice",
            "session_id": session_id,
        }

        # Act: First request with session_id
        response1 = await client.post(
            "/v1/agent/chat",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: First response should succeed
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["session_id"] == session_id, "Should return same session_id"

        # Act: Second request with same session_id
        request_data2 = {
            "message": "What is my name?",
            "session_id": session_id,
        }
        response2 = await client.post(
            "/v1/agent/chat",
            json=request_data2,
            headers=auth_headers,
        )

        # Assert: Second response should succeed
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id, "Should return same session_id"

    @pytest.mark.asyncio
    async def test_chat_endpoint_without_session_id(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Chat endpoint should work without session_id."""
        # Arrange: Request without session_id
        request_data = {"message": "Hello"}

        # Act: POST without session_id
        response = await client.post(
            "/v1/agent/chat",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Should succeed
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] is None, "session_id should be null when not provided"

    @pytest.mark.asyncio
    async def test_chat_endpoint_validates_message_length(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Chat endpoint should validate message length constraints."""
        # Arrange: Empty message (should fail min_length validation)
        request_data = {"message": ""}

        # Act: POST with empty message
        response = await client.post(
            "/v1/agent/chat",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Should return 422 validation error
        assert response.status_code == 422, "Empty message should fail validation"

    @pytest.mark.asyncio
    async def test_chat_endpoint_requires_message_field(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Chat endpoint should require message field."""
        # Arrange: Request without message field
        request_data = {"session_id": "test"}

        # Act: POST without message
        response = await client.post(
            "/v1/agent/chat",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Should return 422 validation error
        assert response.status_code == 422, "Missing message field should fail validation"

    @pytest.mark.asyncio
    async def test_chat_endpoint_content_type(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Chat endpoint should return JSON content type."""
        # Arrange: Valid request
        request_data = {"message": "Test"}

        # Act: POST to chat endpoint
        response = await client.post(
            "/v1/agent/chat",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Content-Type should be JSON
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"], (
            "Response should have JSON content type"
        )
