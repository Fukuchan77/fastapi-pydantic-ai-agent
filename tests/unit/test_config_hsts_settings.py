"""Unit tests for HSTS and proxy-trust settings (12.1, L4.5/L4.6).

Covers `hsts_max_age`, `hsts_include_subdomains`, and `trust_proxy_headers` on
`SecuritySettingsMixin`. `hsts_max_age`/`hsts_include_subdomains` replace the
hard-coded `Strict-Transport-Security` literal in
`app/middleware/security_headers.py` (12.2); `trust_proxy_headers` is read only
by the L4.3/12.3 startup warning and grants no trust itself.
"""

import pytest
from pydantic import ValidationError


def test_hsts_max_age_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that hsts_max_age defaults to 31536000 (1 year), matching the prior literal."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("HSTS_MAX_AGE", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.hsts_max_age == 31536000


def test_hsts_max_age_custom_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that hsts_max_age accepts a custom value, e.g. for a shorter rollout window."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("HSTS_MAX_AGE", "3600")

    from app.config import Settings

    settings = Settings()

    assert settings.hsts_max_age == 3600


def test_hsts_max_age_rejects_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that hsts_max_age rejects a negative value (not a valid max-age)."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("HSTS_MAX_AGE", "-1")

    from app.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("hsts_max_age",) for error in errors)


def test_hsts_include_subdomains_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that hsts_include_subdomains defaults to True, matching the prior literal."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("HSTS_INCLUDE_SUBDOMAINS", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.hsts_include_subdomains is True


def test_hsts_include_subdomains_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that hsts_include_subdomains can be switched off for a shared apex domain."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("HSTS_INCLUDE_SUBDOMAINS", "false")

    from app.config import Settings

    settings = Settings()

    assert settings.hsts_include_subdomains is False


def test_trust_proxy_headers_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that trust_proxy_headers defaults to False (no trust granted without opt-in)."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)

    from app.config import Settings

    settings = Settings()

    assert settings.trust_proxy_headers is False


def test_trust_proxy_headers_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that trust_proxy_headers can be set to True as the operator confirmation flag."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")

    from app.config import Settings

    settings = Settings()

    assert settings.trust_proxy_headers is True
