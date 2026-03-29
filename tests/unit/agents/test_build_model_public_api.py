"""Tests for public build_model API.

Verify that build_model is exposed as a public function
and can be imported and used by other modules.
"""

import pytest

from app.agents.chat_agent import build_model


def test_build_model_is_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that build_model is accessible as a public API."""
    # Arrange: Set up test environment
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")

    # Clear settings cache
    from app.config import get_settings

    get_settings.cache_clear()

    settings = get_settings()

    # Act: Call build_model (should not raise ImportError or AttributeError)
    model = build_model(settings)

    # Assert: Model should be created successfully
    assert model is not None
    # Verify it's a Model instance by checking it has model_name attribute
    assert hasattr(model, "model_name")


def test_build_model_supports_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that build_model correctly creates OpenAI model."""
    # Arrange
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")

    from app.config import get_settings

    get_settings.cache_clear()

    settings = get_settings()

    # Act
    model = build_model(settings)

    # Assert: Model should be created successfully
    assert model is not None
    assert hasattr(model, "model_name")
    assert "gpt-4o" in str(model.model_name)


def test_build_model_supports_ollama_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that build_model correctly creates Ollama model."""
    # Arrange
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")

    from app.config import get_settings

    get_settings.cache_clear()

    settings = get_settings()

    # Act
    model = build_model(settings)

    # Assert: Model should be created successfully
    assert model is not None
    assert hasattr(model, "model_name")
    assert "llama2" in str(model.model_name)
