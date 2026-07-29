"""Unit tests for anonymized user-message logging on the SSE stream endpoint.

Ensures that user message content is NOT logged in error/cancellation
scenarios, preventing potential leakage of sensitive information (passwords,
tokens, PII). Only metadata like message length should be logged.
"""

import contextlib
import logging
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.agents.deps import AgentDeps
from app.agents.deps import get_agent_deps
from app.agents.guardrails import AuditTrail
from app.api.v1.agent import router
from app.deps.auth import verify_api_key
from tests.conftest import build_test_settings


@pytest.fixture
def mock_agent_deps() -> AgentDeps:
    """Create mock AgentDeps for testing."""
    deps = MagicMock(spec=AgentDeps)
    deps.session_store = MagicMock()
    deps.session_store.get_history = AsyncMock(return_value=[])
    deps.session_store.save_history = AsyncMock()
    deps.settings = build_test_settings()
    deps.audit = AuditTrail()
    return deps


def _build_app(mock_agent_deps: AgentDeps, chat_agent: Agent[AgentDeps, str]) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.chat_agent = chat_agent
    app.state.settings = build_test_settings()
    app.dependency_overrides[get_agent_deps] = lambda: mock_agent_deps
    app.dependency_overrides[verify_api_key] = lambda: None  # Bypass auth
    return app


def _failing_agent() -> Agent[AgentDeps, str]:
    async def _raise_runtime_error(messages: list, agent_info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("LLM API failure")
        yield ""  # pragma: no cover

    return Agent(
        model=FunctionModel(stream_function=_raise_runtime_error),
        deps_type=AgentDeps,
        output_type=str,
    )


def _cancelling_agent() -> Agent[AgentDeps, str]:
    import asyncio

    async def _raise_cancelled(messages: list, agent_info: AgentInfo) -> AsyncIterator[str]:
        raise asyncio.CancelledError()
        yield ""  # pragma: no cover

    return Agent(
        model=FunctionModel(stream_function=_raise_cancelled),
        deps_type=AgentDeps,
        output_type=str,
    )


def test_error_log_does_not_contain_user_message_content(
    mock_agent_deps: AgentDeps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that error logs do NOT contain user message content.

    User messages should NOT be logged even in truncated form.
    Only metadata like message length should be logged.
    """
    app = _build_app(mock_agent_deps, _failing_agent())
    client = TestClient(app)

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
        assert "password" not in record.message.lower(), (
            f"Log message contains sensitive content: {record.message}"
        )
        assert "secret123" not in record.message, (
            f"Log message contains sensitive content: {record.message}"
        )
        assert "123-45-6789" not in record.message, (
            f"Log message contains sensitive content: {record.message}"
        )


def test_error_log_contains_message_length_metadata(
    mock_agent_deps: AgentDeps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that error logs contain message LENGTH instead of content.

    Logs should contain useful metadata (message length)
    for debugging without exposing sensitive content.
    """
    app = _build_app(mock_agent_deps, _failing_agent())
    client = TestClient(app)

    test_message = "A" * 150  # 150 character message

    with caplog.at_level(logging.ERROR):
        response = client.post(
            "/v1/agent/stream",
            json={"message": test_message},
            headers={"X-API-Key": "test-key"},
        )

    assert response.status_code == 200

    error_logs = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(error_logs) > 0

    found_length_metadata = False
    for record in error_logs:
        if hasattr(record, "message_length"):
            found_length_metadata = True
            assert record.message_length == 150, (
                f"Expected message_length=150, got {record.message_length}"
            )
            break

    assert found_length_metadata, "Error logs should contain message_length metadata for debugging"


def test_info_log_for_client_disconnect_also_anonymized(
    mock_agent_deps: AgentDeps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that client disconnect (CancelledError) logs are also anonymized.

    All user message logging should be anonymized,
    including INFO-level logs for normal events like client disconnect.
    """
    app = _build_app(mock_agent_deps, _cancelling_agent())
    client = TestClient(app)

    sensitive_message = "DELETE FROM users WHERE password='admin123'"

    # The CancelledError propagates out of the response body iterator
    # (Req 2.6); the ASGI transport may surface this as a client-side
    # exception once headers are already sent — only the log content
    # anonymization matters for this test, not how the connection ends.
    with caplog.at_level(logging.INFO), contextlib.suppress(Exception):
        client.post(
            "/v1/agent/stream",
            json={"message": sensitive_message},
            headers={"X-API-Key": "test-key"},
        )

    info_logs = [record for record in caplog.records if record.levelno == logging.INFO]
    assert len(info_logs) > 0, "Expected the cancellation to be logged at INFO level"

    for record in info_logs:
        assert "password" not in record.message.lower(), (
            f"INFO log contains sensitive content: {record.message}"
        )
        assert "admin123" not in record.message, (
            f"INFO log contains sensitive content: {record.message}"
        )
        assert "DELETE FROM users" not in record.message, (
            f"INFO log contains sensitive content: {record.message}"
        )
