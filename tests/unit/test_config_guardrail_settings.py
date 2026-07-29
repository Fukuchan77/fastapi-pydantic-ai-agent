"""Unit tests for agent guardrail settings (Req 4.1, 4.2, 4.6).

Covers chat_request_timeout, usage_request_limit, and usage_total_tokens_limit.
"""

import pytest
from pydantic import ValidationError


def test_chat_request_timeout_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that chat_request_timeout defaults to 60 seconds."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("CHAT_REQUEST_TIMEOUT", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.chat_request_timeout == 60


def test_chat_request_timeout_custom_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that chat_request_timeout accepts a custom value."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("CHAT_REQUEST_TIMEOUT", "10")

    from app.config import Settings

    settings = Settings()

    assert settings.chat_request_timeout == 10


def test_chat_request_timeout_rejects_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that chat_request_timeout below the 5s floor is rejected."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("CHAT_REQUEST_TIMEOUT", "1")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("chat_request_timeout",) for error in errors)


def test_usage_request_limit_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that usage_request_limit defaults to 50, matching UsageLimits' own default."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("USAGE_REQUEST_LIMIT", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.usage_request_limit == 50


def test_usage_request_limit_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that usage_request_limit rejects a value of 0."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("USAGE_REQUEST_LIMIT", "0")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("usage_request_limit",) for error in errors)


def test_usage_total_tokens_limit_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that usage_total_tokens_limit defaults to None (both checks disabled)."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("USAGE_TOTAL_TOKENS_LIMIT", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.usage_total_tokens_limit is None


def test_usage_total_tokens_limit_custom_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that usage_total_tokens_limit accepts a custom value."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("USAGE_TOTAL_TOKENS_LIMIT", "4000")

    from app.config import Settings

    settings = Settings()

    assert settings.usage_total_tokens_limit == 4000


def test_usage_total_tokens_limit_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that usage_total_tokens_limit rejects a value of 0 (use None to disable)."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("USAGE_TOTAL_TOKENS_LIMIT", "0")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("usage_total_tokens_limit",) for error in errors)
