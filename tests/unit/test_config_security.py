"""Unit tests for security hardening in app/config.py."""

import pytest
from pydantic import ValidationError


def test_llm_base_url_validates_http_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_base_url only accepts valid HTTP URLs to prevent SSRF."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_BASE_URL", "invalid-url-no-scheme")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    # Verify the error is about llm_base_url field
    errors = exc_info.value.errors()
    assert any(error["loc"] == ("llm_base_url",) for error in errors)


def test_llm_base_url_accepts_valid_http_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_base_url accepts valid HTTP URLs."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")

    from app.config import Settings

    settings = Settings()
    assert str(settings.llm_base_url) == "http://localhost:11434/v1"


def test_llm_base_url_accepts_valid_https_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_base_url accepts valid HTTPS URLs."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")

    from app.config import Settings

    settings = Settings()
    assert str(settings.llm_base_url) == "https://api.openai.com/v1"


def test_max_output_retries_rejects_negative_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that max_output_retries rejects negative values."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("MAX_OUTPUT_RETRIES", "-1")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    # Verify the error is about max_output_retries field
    errors = exc_info.value.errors()
    assert any(error["loc"] == ("max_output_retries",) for error in errors)


def test_max_output_retries_rejects_values_above_10(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that max_output_retries rejects values greater than 10."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("MAX_OUTPUT_RETRIES", "11")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    # Verify the error is about max_output_retries field
    errors = exc_info.value.errors()
    assert any(error["loc"] == ("max_output_retries",) for error in errors)


def test_max_output_retries_accepts_valid_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that max_output_retries accepts values in valid range [0, 10]."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

    from app.config import Settings

    # Test boundary values
    monkeypatch.setenv("MAX_OUTPUT_RETRIES", "0")
    settings = Settings()
    assert settings.max_output_retries == 0

    monkeypatch.setenv("MAX_OUTPUT_RETRIES", "10")
    settings = Settings()
    assert settings.max_output_retries == 10

    # Test mid-range value
    monkeypatch.setenv("MAX_OUTPUT_RETRIES", "5")
    settings = Settings()
    assert settings.max_output_retries == 5


def test_api_key_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that api_key value is not exposed in repr output to prevent accidental logging.

    Task 16.7: With SecretStr, the field name appears in repr as 'api_key=SecretStr(...)',
    but the actual secret value is hidden (shown as '**********'). This is acceptable and
    more useful for debugging than completely hiding the field.
    """
    monkeypatch.setenv("API_KEY", "secret-key-12345")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

    from app.config import Settings

    settings = Settings()
    repr_output = repr(settings)

    # The secret value should NOT appear in repr output (most important security check)
    assert "secret-key-12345" not in repr_output
    # SecretStr shows the field but hides the value as '**********'
    assert "SecretStr" in repr_output
    assert "api_key=SecretStr" in repr_output


def test_llm_api_key_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_api_key value is not exposed in repr output to prevent accidental logging.

    Task 16.7: With SecretStr, the field name appears but the value is hidden.
    """
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-llm-key-67890")

    from app.config import Settings

    settings = Settings()
    repr_output = repr(settings)

    # The secret value should NOT appear in repr output (security check)
    assert "sk-secret-llm-key-67890" not in repr_output
    # SecretStr shows the field name but hides the value
    assert "llm_api_key=SecretStr" in repr_output


def test_logfire_token_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that logfire_token value is not exposed in repr output to prevent accidental logging.

    Task 16.7: With SecretStr, the field name appears but the value is hidden.
    """
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")
    monkeypatch.setenv("LOGFIRE_TOKEN", "logfire-secret-token-abc123")

    from app.config import Settings

    settings = Settings()
    repr_output = repr(settings)

    # The secret value should NOT appear in repr output (security check)
    assert "logfire-secret-token-abc123" not in repr_output
    # SecretStr shows the field name but hides the value
    assert "logfire_token=SecretStr" in repr_output


def test_all_secrets_protected_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that all secret field values are protected in repr output when all are set.

    Task 16.7: With SecretStr, field names appear but actual secret values are hidden.
    This is the desired behavior - we want to see which fields are set while keeping values secret.
    """
    monkeypatch.setenv("API_KEY", "secret-api-key-123")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-llm-key-456")
    monkeypatch.setenv("LOGFIRE_TOKEN", "logfire-secret-789")

    from app.config import Settings

    settings = Settings()
    repr_output = repr(settings)

    # None of the secret VALUES should appear in repr output (most important check)
    assert "secret-api-key-123" not in repr_output
    assert "sk-secret-llm-key-456" not in repr_output
    assert "logfire-secret-789" not in repr_output

    # SecretStr fields appear with masked values for debugging
    assert "api_key=SecretStr" in repr_output
    assert "llm_api_key=SecretStr" in repr_output
    assert "logfire_token=SecretStr" in repr_output


def test_get_settings_uses_cache_decorator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_settings uses @cache decorator (unbounded cache)."""
    from app.config import get_settings

    # Check the cache_info to verify @cache is being used
    cache_info = get_settings.cache_info()
    # @cache decorator provides cache_info and cache_clear methods
    assert hasattr(get_settings, "cache_info")
    assert hasattr(get_settings, "cache_clear")
    # @cache decorator has maxsize=None (unbounded), unlike @lru_cache(maxsize=1)
    assert cache_info.maxsize is None
