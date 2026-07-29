"""Unit tests for session_signing_key strength validation (Req 11.1/11.2).

Mirrors the api_key strength-validation test suite (test_config_api_key_strength.py):
a weak or placeholder signing key would let an attacker forge session ids and
defeat the IDOR protection Req 11.1/11.2 depends on.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings


def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")


class TestSessionSigningKeyPlaceholderDetection:
    """Test that placeholder session_signing_key values are rejected."""

    def test_rejects_placeholder_changeme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Placeholder 'changeme' should be rejected."""
        _base_env(monkeypatch)
        monkeypatch.setenv("SESSION_SIGNING_KEY", "changeme")

        with pytest.raises(ValidationError, match=r"(?i)placeholder|invalid|weak"):
            Settings()

    def test_rejects_placeholder_example(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Placeholder 'example' should be rejected."""
        _base_env(monkeypatch)
        monkeypatch.setenv("SESSION_SIGNING_KEY", "example")

        with pytest.raises(ValidationError, match=r"(?i)placeholder|invalid|weak"):
            Settings()


class TestSessionSigningKeyMinimumLength:
    """Test that session_signing_key meets minimum length requirements."""

    def test_rejects_key_too_short(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A signing key shorter than 16 characters should be rejected."""
        _base_env(monkeypatch)
        monkeypatch.setenv("SESSION_SIGNING_KEY", "short")

        with pytest.raises(ValidationError, match=r"(?i)at least 16 characters|too short"):
            Settings()

    def test_accepts_key_exactly_16_chars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A signing key with exactly 16 characters should be accepted (boundary)."""
        _base_env(monkeypatch)
        monkeypatch.setenv("SESSION_SIGNING_KEY", "a" * 16)

        settings = Settings()
        assert settings.session_signing_key.get_secret_value() == "a" * 16

    def test_accepts_strong_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A strong signing key (32+ chars, random) should be accepted."""
        _base_env(monkeypatch)
        strong_key = "sign-prod-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        monkeypatch.setenv("SESSION_SIGNING_KEY", strong_key)

        settings = Settings()
        assert settings.session_signing_key.get_secret_value() == strong_key


class TestSessionSigningKeyRequired:
    """Test that session_signing_key is a required field."""

    def test_missing_session_signing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings() must fail fast when SESSION_SIGNING_KEY is unset."""
        _base_env(monkeypatch)
        monkeypatch.delenv("SESSION_SIGNING_KEY", raising=False)

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("session_signing_key",) for error in errors)
