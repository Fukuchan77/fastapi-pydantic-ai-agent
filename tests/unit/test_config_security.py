"""Unit tests for security hardening in app/config.py."""

import pytest
from pydantic import ValidationError


def test_llm_base_url_validates_http_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_base_url only accepts valid HTTP URLs to prevent SSRF."""
    monkeypatch.setenv("API_KEY", "test-key")
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
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")

    from app.config import Settings

    settings = Settings()
    assert str(settings.llm_base_url) == "http://localhost:11434/v1"


def test_llm_base_url_accepts_valid_https_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that llm_base_url accepts valid HTTPS URLs."""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")

    from app.config import Settings

    settings = Settings()
    assert str(settings.llm_base_url) == "https://api.openai.com/v1"


def test_max_output_retries_rejects_negative_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that max_output_retries rejects negative values."""
    monkeypatch.setenv("API_KEY", "test-key")
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
    monkeypatch.setenv("API_KEY", "test-key")
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
    monkeypatch.setenv("API_KEY", "test-key")
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
    """Test that api_key is not exposed in repr output to prevent accidental logging."""
    monkeypatch.setenv("API_KEY", "secret-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

    from app.config import Settings

    settings = Settings()
    repr_output = repr(settings)

    # The secret value should not appear in repr output
    assert "secret-key-12345" not in repr_output
    # The api_key field should not appear as a standalone field in repr
    # Check for patterns that indicate api_key is a field: either at start or after comma+space
    assert not repr_output.startswith("Settings(api_key=")
    assert ", api_key=" not in repr_output


def test_get_settings_cache_has_maxsize_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_settings cache is configured with maxsize=1."""
    from app.config import get_settings

    # Check the cache_info to verify maxsize
    cache_info = get_settings.cache_info()
    # lru_cache with maxsize=1 should have maxsize attribute
    # We can check this by verifying the function has cache_info and maxsize
    assert hasattr(get_settings, "cache_info")
    assert hasattr(get_settings, "cache_clear")
    # The cache_info() should show maxsize in its output
    # For lru_cache(maxsize=1), the cache should exist
    assert cache_info.maxsize == 1
