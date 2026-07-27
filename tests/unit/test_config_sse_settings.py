"""Unit tests for SSE stream settings (sse_max_events, sse_heartbeat_interval, sse_send_timeout)."""

import pytest
from pydantic import ValidationError


def test_sse_max_events_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that sse_max_events defaults to 1000."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("SSE_MAX_EVENTS", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.sse_max_events == 1000


def test_sse_max_events_custom_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that sse_max_events accepts a custom value."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("SSE_MAX_EVENTS", "50")

    from app.config import Settings

    settings = Settings()

    assert settings.sse_max_events == 50


def test_sse_max_events_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that sse_max_events rejects a value of 0 (must cap at least one event)."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("SSE_MAX_EVENTS", "0")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("sse_max_events",) for error in errors)


def test_sse_heartbeat_interval_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that sse_heartbeat_interval defaults to 15 seconds."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("SSE_HEARTBEAT_INTERVAL", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.sse_heartbeat_interval == 15


def test_sse_heartbeat_interval_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that sse_heartbeat_interval rejects a value of 0."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("SSE_HEARTBEAT_INTERVAL", "0")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("sse_heartbeat_interval",) for error in errors)


def test_sse_send_timeout_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that sse_send_timeout defaults to 60 seconds."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("SSE_SEND_TIMEOUT", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.sse_send_timeout == 60


def test_sse_send_timeout_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that sse_send_timeout rejects a value of 0."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("SSE_SEND_TIMEOUT", "0")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("sse_send_timeout",) for error in errors)


def test_sse_heartbeat_interval_must_not_exceed_send_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that sse_heartbeat_interval greater than sse_send_timeout is rejected.

    A heartbeat that never fires within the send-timeout budget can never keep
    the connection alive, so this combination is a configuration error.
    """
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("SSE_HEARTBEAT_INTERVAL", "90")
    monkeypatch.setenv("SSE_SEND_TIMEOUT", "60")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any("sse_heartbeat_interval" in str(error["msg"]).lower() for error in errors)
