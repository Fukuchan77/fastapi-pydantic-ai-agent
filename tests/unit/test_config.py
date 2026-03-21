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
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")

    from app.config import Settings

    settings = Settings()

    assert settings.api_key == "test-api-key"
    assert settings.llm_model == "ollama:llama2"
    assert settings.llm_api_key is None  # Optional field for local provider
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
    assert settings.logfire_token == "logfire-token-123"  # noqa: S105
    assert settings.logfire_service_name == "my-service"


def test_get_settings_function_exists() -> None:
    """Test that get_settings function exists and returns Settings instance."""
    from app.config import get_settings

    # Function should exist
    assert callable(get_settings)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_settings returns the same instance (cached)."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")

    from app.config import get_settings

    settings1 = get_settings()
    settings2 = get_settings()

    # Should be the same object (cached)
    assert settings1 is settings2


def test_llm_model_valid_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings accepts valid llm_model formats."""
    monkeypatch.setenv("API_KEY", "test-key")

    from app.config import Settings

    # Test valid providers with model names
    # Cloud providers need LLM_API_KEY, local providers don't
    valid_models_with_key = [
        ("openai:gpt-4o", "sk-test"),
        ("anthropic:claude-3-opus", "sk-test"),
        ("groq:mixtral-8x7b", "gsk-test"),
    ]
    valid_models_without_key = [
        "ollama:llama2",
    ]

    # Test cloud providers with API key
    for model, api_key in valid_models_with_key:
        monkeypatch.setenv("LLM_MODEL", model)
        monkeypatch.setenv("LLM_API_KEY", api_key)
        settings = Settings()
        assert settings.llm_model == model

    # Test local provider without API key
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    for model in valid_models_without_key:
        monkeypatch.setenv("LLM_MODEL", model)
        settings = Settings()
        assert settings.llm_model == model


def test_llm_model_invalid_format_no_colon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings raises ValidationError when llm_model has no colon."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai-gpt-4o")  # Invalid: no colon

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("llm_model",) for error in errors)
    assert any("provider:model" in str(error["msg"]) for error in errors)


def test_llm_model_invalid_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings raises ValidationError for invalid provider."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "invalid:gpt-4o")  # Invalid provider

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("llm_model",) for error in errors)
    assert any("openai" in str(error["msg"]) for error in errors)
    assert any("anthropic" in str(error["msg"]) for error in errors)
    assert any("ollama" in str(error["msg"]) for error in errors)
    assert any("groq" in str(error["msg"]) for error in errors)


def test_llm_model_empty_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings raises ValidationError for empty provider."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", ":gpt-4o")  # Empty provider

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("llm_model",) for error in errors)


def test_llm_model_empty_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings raises ValidationError for empty model name."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:")  # Empty model

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("llm_model",) for error in errors)


def test_cloud_provider_requires_api_key_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that OpenAI provider requires llm_api_key to be set."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    # Do NOT set LLM_API_KEY - it should fail

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    # Should have a validation error (not necessarily on a specific field, could be model-level)
    assert len(errors) > 0
    error_msg = str(errors[0]["msg"]).lower()
    assert "openai" in error_msg or "api" in error_msg or "key" in error_msg


def test_cloud_provider_requires_api_key_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Anthropic provider requires llm_api_key to be set."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "anthropic:claude-3-opus")
    # Do NOT set LLM_API_KEY - it should fail

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert len(errors) > 0
    error_msg = str(errors[0]["msg"]).lower()
    assert "anthropic" in error_msg or "api" in error_msg or "key" in error_msg


def test_cloud_provider_requires_api_key_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Groq provider requires llm_api_key to be set."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "groq:mixtral-8x7b")
    # Do NOT set LLM_API_KEY - it should fail

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert len(errors) > 0
    error_msg = str(errors[0]["msg"]).lower()
    assert "groq" in error_msg or "api" in error_msg or "key" in error_msg


def test_local_provider_works_without_api_key_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Ollama (local provider) works without llm_api_key."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    # Do NOT set LLM_API_KEY - ollama should work without it

    from app.config import Settings

    settings = Settings()
    assert settings.llm_model == "ollama:llama2"
    assert settings.llm_api_key is None  # Should be None and that's OK for ollama


def test_cloud_provider_works_with_api_key_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that OpenAI provider works when llm_api_key is provided."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-openai-test")

    from app.config import Settings

    settings = Settings()
    assert settings.llm_model == "openai:gpt-4o"
    assert settings.llm_api_key == "sk-openai-test"


def test_cloud_provider_works_with_api_key_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Anthropic provider works when llm_api_key is provided."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "anthropic:claude-3-opus")
    monkeypatch.setenv("LLM_API_KEY", "sk-anthropic-test")

    from app.config import Settings

    settings = Settings()
    assert settings.llm_model == "anthropic:claude-3-opus"
    assert settings.llm_api_key == "sk-anthropic-test"


def test_cloud_provider_works_with_api_key_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Groq provider works when llm_api_key is provided."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "groq:mixtral-8x7b")
    monkeypatch.setenv("LLM_API_KEY", "gsk-test")

    from app.config import Settings

    settings = Settings()
    assert settings.llm_model == "groq:mixtral-8x7b"
    assert settings.llm_api_key == "gsk-test"
