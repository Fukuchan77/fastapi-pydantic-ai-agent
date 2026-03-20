"""Unit tests for app/config.py - Pydantic Settings configuration."""

import pytest
from pydantic import ValidationError


def test_settings_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings raises ValidationError when API_KEY is missing."""
    # Clear all relevant env vars
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    # Set only LLM_MODEL, leave API_KEY missing
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

    # Import and try to instantiate Settings - should fail
    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    # Verify the error is about api_key field
    errors = exc_info.value.errors()
    assert any(error["loc"] == ("api_key",) for error in errors)


def test_settings_requires_llm_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings raises ValidationError when LLM_MODEL is missing."""
    # Clear all relevant env vars
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    # Set only API_KEY, leave LLM_MODEL missing
    monkeypatch.setenv("API_KEY", "test-key")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    # Verify the error is about llm_model field
    errors = exc_info.value.errors()
    assert any(error["loc"] == ("llm_model",) for error in errors)


def test_settings_with_all_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings successfully initializes with required fields."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

    from app.config import Settings

    settings = Settings()

    assert settings.api_key == "test-api-key"
    assert settings.llm_model == "openai:gpt-4o"
    assert settings.llm_api_key is None  # Optional field
    assert settings.llm_base_url is None  # Optional field
    assert settings.max_output_retries == 3  # Default value
    assert settings.logfire_token is None  # Optional field
    assert settings.logfire_service_name == "fastapi-pydantic-ai-agent"  # Default


def test_settings_with_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings correctly handles optional fields."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-test123")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("MAX_OUTPUT_RETRIES", "5")
    monkeypatch.setenv("LOGFIRE_TOKEN", "logfire-token-123")
    monkeypatch.setenv("LOGFIRE_SERVICE_NAME", "my-service")

    from app.config import Settings

    settings = Settings()

    assert settings.api_key == "test-key"
    assert settings.llm_model == "openai:gpt-4o"
    assert settings.llm_api_key == "sk-test123"
    # llm_base_url is now HttpUrl type, convert to string for comparison
    assert str(settings.llm_base_url) == "http://localhost:11434/v1"
    assert settings.max_output_retries == 5
    assert settings.logfire_token == "logfire-token-123"
    assert settings.logfire_service_name == "my-service"


def test_get_settings_function_exists() -> None:
    """Test that get_settings function exists and returns Settings instance."""
    from app.config import get_settings

    # Function should exist
    assert callable(get_settings)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_settings returns the same instance (cached)."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

    from app.config import get_settings

    settings1 = get_settings()
    settings2 = get_settings()

    # Should be the same object (cached)
    assert settings1 is settings2
