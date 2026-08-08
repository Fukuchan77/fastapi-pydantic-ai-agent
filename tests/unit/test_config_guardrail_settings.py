"""Unit tests for agent guardrail settings (Req 4.1, 4.2, 4.6, 9.1, 9.2, 9.4).

Covers chat_request_timeout, usage_request_limit, usage_total_tokens_limit,
usage_tool_calls_limit, llm_max_output_tokens, and llm_temperature.
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


def test_usage_total_tokens_limit_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that usage_total_tokens_limit defaults to 100000 (Req 9.4).

    Was `None`, which disabled both the native token check and the guarded
    pre-tool budget gate. A real default closes that gap; see spec.md's
    Requirement 9 backward-compatibility note for the accepted breaking
    change this represents.
    """
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("USAGE_TOTAL_TOKENS_LIMIT", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.usage_total_tokens_limit == 100000


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


def test_usage_tool_calls_limit_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that usage_tool_calls_limit defaults to 20 (Req 9.4).

    Closes the verification finding that `UsageLimits.tool_calls_limit` was
    never set on the guarded run.
    """
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("USAGE_TOOL_CALLS_LIMIT", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.usage_tool_calls_limit == 20


def test_usage_tool_calls_limit_custom_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that usage_tool_calls_limit accepts a custom value."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("USAGE_TOOL_CALLS_LIMIT", "5")

    from app.config import Settings

    settings = Settings()

    assert settings.usage_tool_calls_limit == 5


def test_usage_tool_calls_limit_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that usage_tool_calls_limit rejects a value of 0 (use None to disable)."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("USAGE_TOOL_CALLS_LIMIT", "0")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("usage_tool_calls_limit",) for error in errors)


def test_llm_max_output_tokens_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_max_output_tokens defaults to 4096 (Req 9.1)."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("LLM_MAX_OUTPUT_TOKENS", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.llm_max_output_tokens == 4096


def test_llm_max_output_tokens_custom_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_max_output_tokens accepts a custom value."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "1024")

    from app.config import Settings

    settings = Settings()

    assert settings.llm_max_output_tokens == 1024


def test_llm_max_output_tokens_rejects_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_max_output_tokens rejects a value of 0."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "0")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("llm_max_output_tokens",) for error in errors)


def test_llm_temperature_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_temperature defaults to 0.7 (Req 9.1)."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.llm_temperature == 0.7


def test_llm_temperature_custom_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_temperature accepts a custom value."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")

    from app.config import Settings

    settings = Settings()

    assert settings.llm_temperature == 0.2


@pytest.mark.parametrize("bad_value", ["-0.1", "2.1"])
def test_llm_temperature_rejects_out_of_range(
    monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    """Test that llm_temperature rejects values outside the [0.0, 2.0] sampling range."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LLM_TEMPERATURE", bad_value)

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("llm_temperature",) for error in errors)
