"""Unit tests for LLM agent timeout configuration (Task 19.3)."""

import pytest
from pydantic import ValidationError


def test_llm_agent_timeout_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_agent_timeout has default value of 30 seconds."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("LLM_AGENT_TIMEOUT", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.llm_agent_timeout == 30


def test_llm_agent_timeout_custom_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_agent_timeout accepts valid custom values."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "60")

    from app.config import Settings

    settings = Settings()

    assert settings.llm_agent_timeout == 60


def test_llm_agent_timeout_minimum_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_agent_timeout accepts minimum value of 5 seconds."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "5")

    from app.config import Settings

    settings = Settings()

    assert settings.llm_agent_timeout == 5


def test_llm_agent_timeout_maximum_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_agent_timeout accepts maximum value of 300 seconds."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "300")

    from app.config import Settings

    settings = Settings()

    assert settings.llm_agent_timeout == 300


def test_llm_agent_timeout_rejects_too_low(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_agent_timeout rejects values below 5 seconds."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "4")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("llm_agent_timeout",) for error in errors)


def test_llm_agent_timeout_rejects_too_high(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_agent_timeout rejects values above 300 seconds."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "301")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("llm_agent_timeout",) for error in errors)
