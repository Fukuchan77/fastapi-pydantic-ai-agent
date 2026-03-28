"""Tests for silent session save failure in streaming endpoint.

Task: FIX 優先対応事項 - Issue 3
Problem: In agent.py:237-244, session save failures during streaming are silently caught
by the outer exception handler without specific logging or error handling.
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import UserPromptPart

from app.main import app


@pytest.mark.asyncio
async def test_stream_logs_session_save_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that session save failures in streaming endpoint are logged explicitly.

    RED PHASE: This test should fail because the current implementation doesn't
    have explicit error handling for save_history failures in the streaming endpoint.
    When save_history fails, it's caught by the generic exception handler without
    specific logging about the save failure.
    """
    # Set required environment variables for Settings
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")  # Must follow 'provider:model' format
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")  # Required for cloud providers

    # Mock the streaming result
    mock_result = MagicMock()

    # stream_text should return an async generator
    async def mock_stream_text(delta=True):
        async for item in async_text_generator():
            yield item

    mock_result.stream_text = mock_stream_text
    mock_result.all_messages = MagicMock(
        return_value=[
            ModelRequest(parts=[UserPromptPart(content="test")]),
        ]
    )

    # Mock the agent run_stream to return our mock result
    mock_agent = MagicMock()
    mock_agent.run_stream = MagicMock()
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=mock_result)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock(return_value=None)

    # Mock session store that fails on save
    mock_session_store = AsyncMock()
    mock_session_store.get_history = AsyncMock(return_value=[])
    mock_session_store.save_history = AsyncMock(
        side_effect=ValueError("Validation error: messages exceed limit")
    )

    # Patch dependencies - get_agent_deps is async so use AsyncMock
    with patch("app.api.v1.agent.get_agent_deps") as mock_get_deps:
        mock_deps = MagicMock()
        mock_deps.session_store = mock_session_store
        # For async functions, use AsyncMock
        mock_get_deps_async = AsyncMock(return_value=mock_deps)
        mock_get_deps.side_effect = mock_get_deps_async

        # Patch the app state to use our mock agent and required state attributes
        # Use create=True because these don't exist until lifespan runs
        with (
            patch.object(app.state, "chat_agent", mock_agent, create=True),
            patch.object(app.state, "http_client", AsyncMock(), create=True),
            patch.object(app.state, "settings", MagicMock(), create=True),
            patch.object(app.state, "session_store", mock_session_store, create=True),
        ):
            client = TestClient(app)

            # Capture logs
            with patch("app.api.v1.agent.logger") as mock_logger:
                # Make request
                response = client.post(
                    "/v1/agent/stream",
                    json={"message": "test", "session_id": "test-session"},
                    headers={"X-API-Key": "test-api-key-12345"},
                )

                # Response should be 200 (streaming starts successfully)
                assert response.status_code == 200

                # Read the stream to completion
                content = b""
                for chunk in response.iter_bytes():
                    content += chunk

                # RED PHASE: This assertion should fail because there's no specific
                # logging for session save failures
                # Check that error was logged with session save context
                error_logged = False
                for call in mock_logger.warning.call_args_list:
                    args = call[0]
                    if len(args) > 0 and "session" in str(args[0]).lower():
                        error_logged = True
                        break

                assert error_logged, (
                    "Session save failure should be logged explicitly, "
                    "not just caught by generic exception handler"
                )


@pytest.mark.asyncio
async def test_stream_handles_session_save_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that validation errors during session save are handled appropriately.

    Validation errors (like message count exceeding limit) should be logged
    with enough context to debug the issue.
    """
    # Set required environment variables for Settings
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")  # Must follow 'provider:model' format
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")  # Required for cloud providers

    mock_result = MagicMock()

    # stream_text should return an async generator
    async def mock_stream_text(delta=True):
        async for item in async_text_generator():
            yield item

    mock_result.stream_text = mock_stream_text
    mock_result.all_messages = MagicMock(
        return_value=[
            ModelRequest(parts=[UserPromptPart(content="test")]),
        ]
    )

    mock_agent = MagicMock()
    mock_agent.run_stream = MagicMock()
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=mock_result)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_session_store = AsyncMock()
    mock_session_store.get_history = AsyncMock(return_value=[])
    mock_session_store.save_history = AsyncMock(
        side_effect=ValueError("Too many messages (max 1000)")
    )

    with patch("app.api.v1.agent.get_agent_deps") as mock_get_deps:
        mock_deps = MagicMock()
        mock_deps.session_store = mock_session_store
        # For async functions, use AsyncMock
        mock_get_deps_async = AsyncMock(return_value=mock_deps)
        mock_get_deps.side_effect = mock_get_deps_async

        # Patch the app state to use our mock agent and required state attributes
        # Use create=True because these don't exist until lifespan runs
        with (
            patch.object(app.state, "chat_agent", mock_agent, create=True),
            patch.object(app.state, "http_client", AsyncMock(), create=True),
            patch.object(app.state, "settings", MagicMock(), create=True),
            patch.object(app.state, "session_store", mock_session_store, create=True),
        ):
            client = TestClient(app)

            with patch("app.api.v1.agent.logger") as mock_logger:
                response = client.post(
                    "/v1/agent/stream",
                    json={"message": "test", "session_id": "test-session"},
                    headers={"X-API-Key": "test-api-key-12345"},
                )

                assert response.status_code == 200

                # Consume stream
                for _ in response.iter_bytes():
                    pass

                # Should have logged the save failure with context
                # Now this should pass because explicit logging was added
                assert mock_logger.warning.called or mock_logger.error.called, (
                    "Session save validation error should be logged"
                )


async def async_text_generator() -> AsyncIterator[str]:
    """Helper generator for mocking stream_text."""
    yield "Hello"
    yield " "
    yield "World"


@pytest.mark.asyncio
async def test_stream_handles_unexpected_session_save_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that unexpected errors during session save are logged with full context.

    Covers lines 251-258 in agent.py - the generic Exception handler
    for unexpected errors during session save (not ValueError).
    """
    # Set required environment variables for Settings
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")

    mock_result = MagicMock()

    # stream_text should return an async generator
    async def mock_stream_text(delta=True):
        async for item in async_text_generator():
            yield item

    mock_result.stream_text = mock_stream_text
    mock_result.all_messages = MagicMock(
        return_value=[
            ModelRequest(parts=[UserPromptPart(content="test")]),
        ]
    )

    mock_agent = MagicMock()
    mock_agent.run_stream = MagicMock()
    mock_agent.run_stream.return_value.__aenter__ = AsyncMock(return_value=mock_result)
    mock_agent.run_stream.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_session_store = AsyncMock()
    mock_session_store.get_history = AsyncMock(return_value=[])
    # Raise a non-ValueError exception (e.g., RuntimeError)
    mock_session_store.save_history = AsyncMock(
        side_effect=RuntimeError("Database connection failed")
    )

    with patch("app.api.v1.agent.get_agent_deps") as mock_get_deps:
        mock_deps = MagicMock()
        mock_deps.session_store = mock_session_store
        mock_get_deps_async = AsyncMock(return_value=mock_deps)
        mock_get_deps.side_effect = mock_get_deps_async

        with (
            patch.object(app.state, "chat_agent", mock_agent, create=True),
            patch.object(app.state, "http_client", AsyncMock(), create=True),
            patch.object(app.state, "settings", MagicMock(), create=True),
            patch.object(app.state, "session_store", mock_session_store, create=True),
        ):
            client = TestClient(app)

            with patch("app.api.v1.agent.logger") as mock_logger:
                response = client.post(
                    "/v1/agent/stream",
                    json={"message": "test", "session_id": "test-session"},
                    headers={"X-API-Key": "test-api-key-12345"},
                )

                assert response.status_code == 200

                # Consume stream
                for _ in response.iter_bytes():
                    pass

                # Should have logged the unexpected error with full context
                # (exc_info=True)
                assert mock_logger.error.called
                # Verify it was called with exc_info=True for full stack trace
                call_kwargs = mock_logger.error.call_args[1]
                assert call_kwargs.get("exc_info") is True
