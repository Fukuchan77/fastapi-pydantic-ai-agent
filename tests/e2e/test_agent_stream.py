"""E2E tests for the typed SSE agent streaming endpoint.

Tests the POST /v1/agent/stream endpoint through the full HTTP stack via
AsyncClient. Covers the wiring-level contract (headers, event framing,
happy-path event sequence, session round-trip, request validation).
Stream-lifecycle edge cases (event cap, heartbeat, send timeout, client
disconnect, CancelledError, terminal error) are unit/integration-tested in
tests/unit/api/v1/test_stream_lifecycle.py, tests/unit/api/v1/test_stream_heartbeat.py,
and tests/integration/test_agent_stream_event_mapping.py — this file does not
duplicate them.
"""

import pytest
from httpx import AsyncClient

from app.patterns.sse import Completed
from app.patterns.sse import StepStarted
from app.patterns.sse import Token
from app.patterns.sse import parse_sse_events


class TestAgentStreamEndpoint:
    """E2E tests for POST /v1/agent/stream endpoint."""

    @pytest.mark.asyncio
    async def test_stream_sets_no_cache_and_no_buffering_headers(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Response headers disable caching and reverse-proxy buffering (Req 2.8)."""
        async with client.stream(
            "POST",
            "/v1/agent/stream",
            json={"message": "Hello"},
            headers=auth_headers,
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["x-accel-buffering"] == "no"
            async for _ in response.aiter_bytes():
                pass

    @pytest.mark.asyncio
    async def test_stream_emits_step_started_tokens_then_completed(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """The typed event sequence starts with step_started and ends with completed."""
        raw = ""
        async with client.stream(
            "POST",
            "/v1/agent/stream",
            json={"message": "Hello, tell me a story"},
            headers=auth_headers,
        ) as response:
            assert response.status_code == 200
            async for chunk in response.aiter_text():
                raw += chunk

        events = parse_sse_events(raw)

        assert len(events) > 0, "Should receive at least one SSE event"
        assert events[0] == StepStarted()
        assert events[-1] == Completed()

        tokens = [e for e in events if isinstance(e, Token)]
        assert len(tokens) > 0
        assert all(isinstance(t.content, str) for t in tokens)

    @pytest.mark.asyncio
    async def test_stream_with_session_id_continues_conversation(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """A server-issued session_id round-trips across two stream requests without error.

        Session ids are now server-issued (Req 11.1); the stream endpoint only
        authorizes an existing one (Req 11.2), so this mints one via
        /v1/agent/chat first rather than supplying an arbitrary client string.
        """
        chat_response = await client.post(
            "/v1/agent/chat",
            json={"message": "Start a session"},
            headers=auth_headers,
        )
        assert chat_response.status_code == 200
        session_id = chat_response.json()["session_id"]

        async with client.stream(
            "POST",
            "/v1/agent/stream",
            json={"message": "Remember I like Python", "session_id": session_id},
            headers=auth_headers,
        ) as response:
            assert response.status_code == 200
            raw = "".join([chunk async for chunk in response.aiter_text()])
        events = parse_sse_events(raw)
        assert events[-1] == Completed()

        async with client.stream(
            "POST",
            "/v1/agent/stream",
            json={"message": "What language do I like?", "session_id": session_id},
            headers=auth_headers,
        ) as response2:
            assert response2.status_code == 200
            raw2 = "".join([chunk async for chunk in response2.aiter_text()])
        events2 = parse_sse_events(raw2)
        assert events2[-1] == Completed()

    @pytest.mark.asyncio
    async def test_stream_without_session_id_completes_normally(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """A stateless request (no session_id) still completes normally."""
        async with client.stream(
            "POST",
            "/v1/agent/stream",
            json={"message": "Hello"},
            headers=auth_headers,
        ) as response:
            assert response.status_code == 200
            raw = "".join([chunk async for chunk in response.aiter_text()])

        events = parse_sse_events(raw)
        assert events[-1] == Completed()

    @pytest.mark.asyncio
    async def test_stream_validates_empty_message(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """An empty message fails request validation before any stream starts."""
        response = await client.post(
            "/v1/agent/stream",
            json={"message": ""},
            headers=auth_headers,
        )

        assert response.status_code == 422, "Empty message should fail validation"

    @pytest.mark.asyncio
    async def test_stream_requires_message_field(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """A missing message field fails request validation."""
        response = await client.post(
            "/v1/agent/stream",
            json={"session_id": "test"},
            headers=auth_headers,
        )

        assert response.status_code == 422, "Missing message should fail validation"
