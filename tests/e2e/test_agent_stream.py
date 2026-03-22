"""E2E tests for agent streaming endpoint.

Tests the POST /v1/agent/stream endpoint through full HTTP stack using AsyncClient.
This endpoint requires authentication and returns Server-Sent Events (SSE) stream.
"""

import json

import pytest
from httpx import AsyncClient


class TestAgentStreamEndpoint:
    """E2E tests for POST /v1/agent/stream endpoint."""

    @pytest.mark.asyncio
    async def test_stream_endpoint_basic_request(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Stream endpoint should return SSE stream with delta and done events."""
        # Arrange: Basic streaming request
        request_data = {"message": "Hello, tell me a story"}

        # Act: POST to stream endpoint with auth
        async with client.stream(
            "POST",
            "/v1/agent/stream",
            json=request_data,
            headers=auth_headers,
        ) as response:
            # Assert: Should return 200 OK
            assert response.status_code == 200, "Stream endpoint should return 200 OK"

            # Assert: Content-Type should be text/event-stream
            assert "text/event-stream" in response.headers["content-type"], (
                "Response should have text/event-stream content type"
            )

            # Collect all events
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove " " prefix
                    event = json.loads(data_str)
                    events.append(event)

        # Assert: Should have received at least one event
        assert len(events) > 0, "Should receive at least one SSE event"

        # Assert: Last event should be done
        assert events[-1]["type"] == "done", "Last event should be done event"

        # Assert: Delta events should have content
        delta_events = [e for e in events if e["type"] == "delta"]
        if delta_events:
            for event in delta_events:
                assert "content" in event, "Delta events should have content field"
                assert isinstance(event["content"], str), "Content should be a string"

    @pytest.mark.asyncio
    async def test_stream_endpoint_with_session_id(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Stream endpoint should handle session_id for conversation history."""
        # Arrange: Request with session_id
        session_id = "test-stream-session-456"
        request_data = {
            "message": "Remember I like Python",
            "session_id": session_id,
        }

        # Act: Stream first message
        async with client.stream(
            "POST",
            "/v1/agent/stream",
            json=request_data,
            headers=auth_headers,
        ) as response:
            assert response.status_code == 200

            # Consume all events
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    event = json.loads(data_str)
                    events.append(event)

        # Assert: Should have done event
        assert any(e["type"] == "done" for e in events), "Should have done event"

        # Act: Stream second message with same session
        request_data2 = {
            "message": "What language do I like?",
            "session_id": session_id,
        }
        async with client.stream(
            "POST",
            "/v1/agent/stream",
            json=request_data2,
            headers=auth_headers,
        ) as response2:
            assert response2.status_code == 200

            # Consume all events
            events2 = []
            async for line in response2.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    event = json.loads(data_str)
                    events2.append(event)

        # Assert: Should have done event
        assert any(e["type"] == "done" for e in events2), "Should have done event"

    @pytest.mark.asyncio
    async def test_stream_endpoint_validates_message(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Stream endpoint should validate message field."""
        # Arrange: Request with empty message
        request_data = {"message": ""}

        # Act: POST with empty message
        response = await client.post(
            "/v1/agent/stream",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Should return 422 validation error
        assert response.status_code == 422, "Empty message should fail validation"

    @pytest.mark.asyncio
    async def test_stream_endpoint_requires_message_field(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Stream endpoint should require message field."""
        # Arrange: Request without message
        request_data = {"session_id": "test"}

        # Act: POST without message
        response = await client.post(
            "/v1/agent/stream",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Should return 422 validation error
        assert response.status_code == 422, "Missing message should fail validation"

    @pytest.mark.asyncio
    async def test_stream_endpoint_event_format(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Stream endpoint should emit properly formatted SSE events."""
        # Arrange: Basic request
        request_data = {"message": "Test"}

        # Act: Stream response
        async with client.stream(
            "POST",
            "/v1/agent/stream",
            json=request_data,
            headers=auth_headers,
        ) as response:
            assert response.status_code == 200

            # Collect and parse events
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    event = json.loads(data_str)
                    events.append(event)

        # Assert: All events should have required fields
        for event in events:
            assert "type" in event, "Each event should have type field"
            assert "content" in event, "Each event should have content field"
            assert isinstance(event["type"], str), "Event type should be string"
            assert isinstance(event["content"], str), "Event content should be string"

        # Assert: Should have exactly one done event at the end
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1, "Should have exactly one done event"
        assert events[-1]["type"] == "done", "Done event should be last"

    @pytest.mark.asyncio
    async def test_stream_endpoint_without_session_id(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Stream endpoint should work without session_id."""
        # Arrange: Request without session_id
        request_data = {"message": "Hello"}

        # Act: Stream without session_id
        async with client.stream(
            "POST",
            "/v1/agent/stream",
            json=request_data,
            headers=auth_headers,
        ) as response:
            # Assert: Should succeed
            assert response.status_code == 200

            # Consume events
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    event = json.loads(data_str)
                    events.append(event)

        # Assert: Should complete with done event
        assert any(e["type"] == "done" for e in events), "Should have done event"
