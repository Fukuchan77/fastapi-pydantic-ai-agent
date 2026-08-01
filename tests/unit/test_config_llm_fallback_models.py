"""Unit tests for the `llm_fallback_models` Settings field (Task 7: FallbackModel).

Covers Req 10.1's provider-chain configuration: a list of additional
"provider:model" identifiers tried in order after `llm_model`, parsed the same
way as `cors_origins` (comma-separated string, JSON array, or list) and
validated with the same "provider:model" format rule as `llm_model`.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings


def _build_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "openai:gpt-4o",
        "llm_api_key": "test-llm-api-key-12345",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_llm_fallback_models_defaults_to_empty_list() -> None:
    """llm_fallback_models defaults to an empty list when not configured."""
    settings = _build_settings()

    assert settings.llm_fallback_models == []


def test_llm_fallback_models_parses_comma_separated_string() -> None:
    """A comma-separated string is parsed into a list, mirroring cors_origins."""
    settings = _build_settings(
        llm_fallback_models="anthropic:claude-3-5-sonnet-20241022,groq:llama3-70b-8192"
    )

    assert settings.llm_fallback_models == [
        "anthropic:claude-3-5-sonnet-20241022",
        "groq:llama3-70b-8192",
    ]


def test_llm_fallback_models_parses_json_array_string() -> None:
    """A JSON array string is parsed into a list, mirroring cors_origins."""
    settings = _build_settings(llm_fallback_models='["anthropic:claude-3-5-sonnet-20241022"]')

    assert settings.llm_fallback_models == ["anthropic:claude-3-5-sonnet-20241022"]


def test_llm_fallback_models_accepts_list() -> None:
    """A list value is accepted as-is."""
    settings = _build_settings(llm_fallback_models=["ollama:llama3.2"])

    assert settings.llm_fallback_models == ["ollama:llama3.2"]


def test_llm_fallback_models_rejects_missing_colon() -> None:
    """Each entry must follow 'provider:model' format, same rule as llm_model."""
    with pytest.raises(ValidationError, match="must follow 'provider:model' format"):
        _build_settings(llm_fallback_models=["gpt-4o"])


def test_llm_fallback_models_rejects_unsupported_provider() -> None:
    """Each entry's provider must be in the same allow-list as llm_model."""
    with pytest.raises(ValidationError, match="provider must be one of"):
        _build_settings(llm_fallback_models=["unsupported:model-name"])


def test_llm_fallback_models_normalizes_uppercase_provider() -> None:
    """Provider names are normalized to lowercase, same as llm_model."""
    settings = _build_settings(llm_fallback_models=["ANTHROPIC:claude-3-5-sonnet-20241022"])

    assert settings.llm_fallback_models == ["anthropic:claude-3-5-sonnet-20241022"]


def test_llm_fallback_models_rejects_malformed_json() -> None:
    """A string starting with '[' that isn't valid JSON must fail loudly, not silently."""
    with pytest.raises(ValidationError, match="not valid JSON"):
        _build_settings(llm_fallback_models="[anthropic:claude-3-5-sonnet-20241022")


def test_llm_fallback_models_cloud_provider_without_api_key_raises() -> None:
    """A cloud-provider fallback must be authenticated too, not just llm_model.

    `build_fallback_model()` builds each fallback via `model_copy()`, which
    does not re-run validators - an unauthenticated cloud fallback must fail
    at Settings construction (startup), not the first time the primary model
    fails over to it.
    """
    with pytest.raises(ValidationError, match=r"llm_api_key is required.*anthropic"):
        Settings(
            api_key="test-api-key-12345",
            llm_model="ollama:llama3.2",
            llm_api_key=None,
            llm_fallback_models=["anthropic:claude-3-5-sonnet-20241022"],
        )
