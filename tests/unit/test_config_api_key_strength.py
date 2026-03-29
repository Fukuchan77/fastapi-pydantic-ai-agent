"""Unit tests for API key strength validation ().

Tests that placeholder and weak API keys are rejected at startup.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings


class TestAPIKeyPlaceholderDetection:
    """Test that placeholder API keys are detected and rejected."""

    def test_rejects_placeholder_your_api_key_here(self, monkeypatch):
        """Placeholder 'your-api-key-here' should be rejected."""
        monkeypatch.setenv("API_KEY", "your-api-key-here")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        with pytest.raises(ValidationError, match=r"(?i)placeholder|invalid|weak"):
            Settings()

    def test_rejects_placeholder_changeme(self, monkeypatch):
        """Placeholder 'changeme' should be rejected."""
        monkeypatch.setenv("API_KEY", "changeme")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        with pytest.raises(ValidationError, match=r"(?i)placeholder|invalid|weak"):
            Settings()

    def test_rejects_placeholder_test_key(self, monkeypatch):
        """Placeholder 'test-key' should be rejected."""
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        with pytest.raises(ValidationError, match=r"(?i)placeholder|invalid|weak"):
            Settings()

    def test_rejects_placeholder_example(self, monkeypatch):
        """Placeholder 'example' should be rejected."""
        monkeypatch.setenv("API_KEY", "example")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        with pytest.raises(ValidationError, match=r"(?i)placeholder|invalid|weak"):
            Settings()


class TestAPIKeyMinimumLength:
    """Test that API keys meet minimum length requirements."""

    def test_rejects_api_key_too_short(self, monkeypatch):
        """API key shorter than 16 characters should be rejected."""
        monkeypatch.setenv("API_KEY", "short")  # 5 characters
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        with pytest.raises(ValidationError, match=r"(?i)at least 16 characters|too short"):
            Settings()

    def test_rejects_api_key_exactly_15_chars(self, monkeypatch):
        """API key with exactly 15 characters should be rejected."""
        monkeypatch.setenv("API_KEY", "a" * 15)
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        with pytest.raises(ValidationError, match=r"(?i)at least 16 characters|too short"):
            Settings()

    def test_accepts_api_key_exactly_16_chars(self, monkeypatch):
        """API key with exactly 16 characters should be accepted (boundary)."""
        monkeypatch.setenv("API_KEY", "a" * 16)
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        settings = Settings()
        assert settings.api_key.get_secret_value() == "a" * 16

    def test_accepts_strong_api_key(self, monkeypatch):
        """Strong API key (32+ chars, random) should be accepted."""
        strong_key = "sk-prod-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
        monkeypatch.setenv("API_KEY", strong_key)
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        settings = Settings()
        assert settings.api_key.get_secret_value() == strong_key


class TestAPIKeyEdgeCases:
    """Test edge cases in API key validation."""

    def test_rejects_empty_api_key(self, monkeypatch):
        """Empty API key should be rejected."""
        monkeypatch.setenv("API_KEY", "")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        with pytest.raises(ValidationError):
            Settings()

    def test_rejects_whitespace_only_api_key(self, monkeypatch):
        """Whitespace-only API key should be rejected."""
        monkeypatch.setenv("API_KEY", "   ")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        with pytest.raises(ValidationError):
            Settings()

    def test_placeholder_detection_is_case_insensitive(self, monkeypatch):
        """Placeholder detection should be case-insensitive."""
        monkeypatch.setenv("API_KEY", "CHANGEME")  # uppercase
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        with pytest.raises(ValidationError, match=r"(?i)placeholder|invalid|weak"):
            Settings()

    def test_accepts_api_key_containing_but_not_equal_to_placeholder(self, monkeypatch):
        """API key containing 'test' but not equal to 'test-key' should be accepted."""
        # This is a valid key that happens to contain "test" in it
        valid_key = "sk-prod-test-a1b2c3d4e5f6g7h8"  # 16+ chars, not a placeholder
        monkeypatch.setenv("API_KEY", valid_key)
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")

        settings = Settings()
        assert settings.api_key.get_secret_value() == valid_key
