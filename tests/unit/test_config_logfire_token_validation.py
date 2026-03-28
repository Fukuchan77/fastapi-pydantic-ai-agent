"""Unit tests for logfire_token validation in app/config.py."""

import pytest
from pydantic import ValidationError


def test_logfire_token_too_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logfire_token with less than 16 characters raises ValidationError."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LOGFIRE_TOKEN", "short")  # Only 5 characters

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("logfire_token",) for error in errors)
    assert any("16 characters" in str(error["msg"]) for error in errors)


def test_logfire_token_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logfire_token cannot be empty string."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LOGFIRE_TOKEN", "")  # Empty string

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("logfire_token",) for error in errors)
    assert any(
        "empty" in str(error["msg"]).lower() or "whitespace" in str(error["msg"]).lower()
        for error in errors
    )


def test_logfire_token_whitespace_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logfire_token cannot be whitespace only."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LOGFIRE_TOKEN", "   ")  # Whitespace only

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("logfire_token",) for error in errors)
    assert any(
        "empty" in str(error["msg"]).lower() or "whitespace" in str(error["msg"]).lower()
        for error in errors
    )


def test_logfire_token_placeholder_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logfire_token rejects placeholder values."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LOGFIRE_TOKEN", "your-token-here")  # Placeholder

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("logfire_token",) for error in errors)
    assert any("placeholder" in str(error["msg"]).lower() for error in errors)


def test_logfire_token_none_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logfire_token can be None (optional field)."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)  # Not set

    from app.config import Settings

    settings = Settings()
    assert settings.logfire_token is None  # Should be None and that's OK


def test_logfire_token_valid_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logfire_token with 16+ characters is accepted."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LOGFIRE_TOKEN", "valid-token-1234")  # 17 characters

    from app.config import Settings

    settings = Settings()
    assert settings.logfire_token.get_secret_value() == "valid-token-1234"


def test_logfire_token_exactly_16_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logfire_token with exactly 16 characters is accepted."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LOGFIRE_TOKEN", "1234567890123456")  # Exactly 16

    from app.config import Settings

    settings = Settings()
    assert settings.logfire_token.get_secret_value() == "1234567890123456"
