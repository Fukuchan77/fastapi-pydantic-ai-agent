"""Unit tests for Anonymize user messages in error logs.

This test ensures that user message content is NOT logged in error scenarios,
preventing potential leakage of sensitive information (passwords, tokens, PII).
"""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.deps import AgentDeps
from app.api.v1.agent import router


@pytest.fixture
def mock_agent_deps() -> AgentDeps:
    """Create mock AgentDeps for testing."""
    deps = MagicMock(spec=AgentDeps)
    deps.session_store = MagicMock()
    deps.session_store.get_history = AsyncMock(return_value=[])
    deps.session_store.save_history = AsyncMock()
    return deps


@pytest.fixture
def app_with_failing_agent(mock_agent_deps: AgentDeps) -> FastAPI:
    """Create test app with agent that raises errors."""
    app = FastAPI()
    app.include_router(router, prefix="/v1")

    # Mock chat agent that raises an error
    mock_agent = MagicMock()
    mock_agent.run_stream = MagicMock(side_effect=RuntimeError("LLM API failure"))
    app.state.chat_agent = mock_agent

    # Override dependencies
    from app.agents.deps import get_agent_deps
    from app.deps.auth import verify_api_key

    app.dependency_overrides[get_agent_deps] = lambda: mock_agent_deps
    app.dependency_overrides[verify_api_key] = lambda: None  # Bypass auth

    return app


def test_error_log_does_not_contain_user_message_content(
    app_with_failing_agent: FastAPI,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that error logs do NOT contain user message content.

    User messages should NOT be logged even in truncated form.
    Only metadata like message length should be logged.
    """
    client = TestClient(app_with_failing_agent)

    sensitive_message = "My password is secret123 and my SSN is 123-45-6789"

    with caplog.at_level(logging.ERROR):
        response = client.post(
            "/v1/agent/stream",
            json={"message": sensitive_message},
            headers={"X-API-Key": "test-key"},
        )

    # Should receive error response (200 for streaming endpoint that returns error in stream)
    assert response.status_code == 200

    # Check error logs
    error_logs = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(error_logs) > 0, "Expected at least one error log"

    # CRITICAL: Verify NO log contains the sensitive message content
    for record in error_logs:
        # Check message
        assert "password" not in record.message.lower(), (
            f"Log message contains sensitive content: {record.message}"
        )
        assert "secret123" not in record.message, (
            f"Log message contains sensitive content: {record.message}"
        )
        assert "123-45-6789" not in record.message, (
            f"Log message contains sensitive content: {record.message}"
        )

        # Check extra fields (the actual issue in line 303)
        if hasattr(record, "user_message") or "user_message" in record.__dict__:
            # If user_message field exists, it should NOT contain actual content
            user_msg = getattr(record, "user_message", record.__dict__.get("user_message"))
            assert "password" not in str(user_msg).lower(), (
                f"Log extra field contains sensitive content: {user_msg}"
            )
            assert "secret123" not in str(user_msg), (
                f"Log extra field contains sensitive content: {user_msg}"
            )


def test_error_log_contains_message_length_metadata(
    app_with_failing_agent: FastAPI,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that error logs contain message LENGTH instead of content.

    Logs should contain useful metadata (message length)
    for debugging without exposing sensitive content.
    """
    client = TestClient(app_with_failing_agent)

    test_message = "A" * 150  # 150 character message

    with caplog.at_level(logging.ERROR):
        _response = client.post(
            "/v1/agent/stream",
            json={"message": test_message},
            headers={"X-API-Key": "test-key"},
        )

    # Check error logs contain length metadata
    error_logs = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(error_logs) > 0

    # Should contain message_length in extra fields
    found_length_metadata = False
    for record in error_logs:
        if hasattr(record, "message_length") or "message_length" in record.__dict__:
            found_length_metadata = True
            msg_length = getattr(record, "message_length", record.__dict__.get("message_length"))
            assert msg_length == 150, f"Expected message_length=150, got {msg_length}"
            break

    assert found_length_metadata, "Error logs should contain message_length metadata for debugging"


def test_info_log_for_client_disconnect_also_anonymized(
    mock_agent_deps: AgentDeps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that client disconnect logs (line 291) are also anonymized.

    All user message logging should be anonymized,
    including INFO-level logs for normal events like client disconnect.
    """
    app = FastAPI()
    app.include_router(router, prefix="/v1")

    # Mock agent that simulates client disconnect
    mock_agent = MagicMock()

    async def mock_run_stream(*args, **kwargs):
        import asyncio

        raise asyncio.CancelledError()

    mock_agent.run_stream = mock_run_stream
    app.state.chat_agent = mock_agent

    from app.agents.deps import get_agent_deps
    from app.deps.auth import verify_api_key

    app.dependency_overrides[get_agent_deps] = lambda: mock_agent_deps
    app.dependency_overrides[verify_api_key] = lambda: None  # Bypass auth

    client = TestClient(app)

    sensitive_message = "DELETE FROM users WHERE password='admin123'"

    with caplog.at_level(logging.INFO):
        _response = client.post(
            "/v1/agent/stream",
            json={"message": sensitive_message},
            headers={"X-API-Key": "test-key"},
        )

    # Check INFO logs for client disconnect
    info_logs = [record for record in caplog.records if record.levelno == logging.INFO]

    for record in info_logs:
        # Verify NO log contains sensitive SQL injection attempt
        assert "password" not in record.message.lower(), (
            f"INFO log contains sensitive content: {record.message}"
        )
        assert "admin123" not in record.message, (
            f"INFO log contains sensitive content: {record.message}"
        )
        assert "DELETE FROM users" not in record.message, (
            f"INFO log contains sensitive content: {record.message}"
        )
