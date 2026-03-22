"""Tests for enable_mock_tools production guard.

This module tests that enable_mock_tools cannot be enabled in production
environments to prevent security vulnerabilities.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_enable_mock_tools_blocked_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that enable_mock_tools=True raises error when APP_ENV=production.

    This prevents accidentally enabling mock tools in production, which could
    expose security vulnerabilities.
    """
    # Arrange: Set production environment
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("ENABLE_MOCK_TOOLS", "true")

    # Act & Assert: Should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        Settings()

    # Verify error message mentions production and mock tools
    error_str = str(exc_info.value)
    assert "production" in error_str.lower()
    assert "mock" in error_str.lower()


def test_enable_mock_tools_allowed_in_development(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that enable_mock_tools=True is allowed when APP_ENV=development."""
    # Arrange: Set development environment
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("ENABLE_MOCK_TOOLS", "true")

    # Act: Should succeed
    settings = Settings()

    # Assert: Mock tools should be enabled
    assert settings.enable_mock_tools is True


def test_enable_mock_tools_allowed_when_false_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that enable_mock_tools=False is allowed in production."""
    # Arrange: Set production environment with mock tools disabled
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("ENABLE_MOCK_TOOLS", "false")

    # Act: Should succeed
    settings = Settings()

    # Assert: Mock tools should be disabled
    assert settings.enable_mock_tools is False


def test_enable_mock_tools_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that enable_mock_tools defaults to False when not set."""
    # Arrange: Don't set ENABLE_MOCK_TOOLS
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    # Act: Should succeed
    settings = Settings()

    # Assert: Mock tools should be disabled by default
    assert settings.enable_mock_tools is False


def test_app_env_defaults_to_development_when_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that APP_ENV defaults to 'development' when not set, allowing mock tools."""
    # Arrange: Don't set APP_ENV, but enable mock tools
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("ENABLE_MOCK_TOOLS", "true")

    # Act: Should succeed (defaults to development)
    settings = Settings()

    # Assert: Mock tools should be enabled
    assert settings.enable_mock_tools is True
    assert settings.app_env == "development"
