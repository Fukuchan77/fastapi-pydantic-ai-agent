"""Tests for public build_model API.

Verify that build_model is exposed as a public function
and can be imported and used by other modules.
"""

import pytest
from pydantic_ai.models import infer_model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai_litellm import LiteLLMModel

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
    """Test that build_model correctly creates OpenAI model.

    Asserts the exact LiteLLM-form identifier rather than a substring match
    (Req 6.12): `"gpt-4o" in str(model.model_name)` would still pass on a
    regression that forgot the "provider:model" -> "provider/model" rewrite
    (the colon-joined "openai:gpt-4o" also contains "gpt-4o"), so it cannot
    verify the spelling explicitly.
    """
    # Arrange
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")

    from app.config import get_settings

    get_settings.cache_clear()

    settings = get_settings()

    # Act
    model = build_model(settings)

    # Assert: Model should be created successfully, with the exact rewritten
    # LiteLLM-form identifier - not the bare "openai:gpt-4o" that reached
    # `build_model`.
    assert model is not None
    assert isinstance(model, LiteLLMModel)
    assert model.model_name == "openai/gpt-4o"


def test_build_model_openai_prefix_bypasses_v2_native_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm v2's changed bare-`openai:` meaning never reaches model construction.

    Req 6.12: pydantic-ai v2 changed what its own model registry
    (`pydantic_ai.models.infer_model`) resolves a bare `"openai:"`-prefixed
    identifier to - constructing `OpenAIResponsesModel` (the Responses API)
    rather than the pre-v2 Chat Completions model. That change is real, as
    demonstrated below rather than assumed, but it cannot reach this
    repository's model construction: `build_model()` splits the configured
    "provider:model" string and hands LiteLLM's own "provider/model" form
    directly to `LiteLLMModel`, never passing the raw string through
    `infer_model`'s provider-prefix resolution.
    """
    # Demonstrate the v2 registry change is real: pydantic-ai's own inference
    # of a bare "openai:" identifier constructs OpenAIResponsesModel. No real
    # network call happens here - constructing the provider only builds a
    # lazy HTTP client.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-for-infer-model-check")
    natively_inferred_model = infer_model("openai:gpt-4o")
    assert isinstance(natively_inferred_model, OpenAIResponsesModel)

    # Confirm build_model() never lets that changed meaning apply: it rewrites
    # the same configured identifier into LiteLLM's form before construction,
    # so the model this repository actually uses is a LiteLLMModel with the
    # exact rewritten spelling, not an OpenAIResponsesModel.
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")

    from app.config import get_settings

    get_settings.cache_clear()

    settings = get_settings()

    model = build_model(settings)

    assert isinstance(model, LiteLLMModel)
    assert not isinstance(model, OpenAIResponsesModel)
    assert model.model_name == "openai/gpt-4o"


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
