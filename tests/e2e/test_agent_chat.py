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
        """Chat endpoint should handle session_id for conversation history.

        Session ids are now server-issued (Req 11.1) - the first request omits
        session_id and the server mints one, which the second request reuses
        to continue the conversation.
        """
        # Act: First request with no session_id - server mints one
        response1 = await client.post(
            "/v1/agent/chat",
            json={"message": "Remember my name is Alice"},
            headers=auth_headers,
        )

        # Assert: First response should succeed and carry a minted session_id
        assert response1.status_code == 200
        data1 = response1.json()
        session_id = data1["session_id"]
        assert isinstance(session_id, str)
        assert session_id

        # Act: Second request with the server-issued session_id
        response2 = await client.post(
            "/v1/agent/chat",
            json={"message": "What is my name?", "session_id": session_id},
            headers=auth_headers,
        )

        # Assert: Second response should succeed and echo the same session_id
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id, "Should return same session_id"

    @pytest.mark.asyncio
    async def test_chat_endpoint_rejects_foreign_session_id(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Presenting a client-supplied (unsigned) session_id is rejected with 403 (Req 11.2)."""
        response = await client.post(
            "/v1/agent/chat",
            json={"message": "Hello", "session_id": "test-session-123"},
            headers=auth_headers,
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_chat_endpoint_without_session_id(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Chat endpoint should mint a new session_id when none is provided (Req 11.1)."""
        # Arrange: Request without session_id
        request_data = {"message": "Hello"}

        # Act: POST without session_id
        response = await client.post(
            "/v1/agent/chat",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Should succeed and mint a new session_id
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["session_id"], str), "server should mint a session_id"
        assert data["session_id"], "minted session_id should not be empty"

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
