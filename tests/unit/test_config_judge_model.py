"""Unit tests for the `judge_model` Settings field (Req 6.4: independent eval judge).

Format validation only - `_build_judge_model()`'s runtime behavior (fallback
to `llm_model`, cross-provider base_url handling, cloud-provider API key
enforcement) is covered separately in `tests/unit/evals/test_runner.py`.
`judge_model` is deliberately exempt from `Settings.validate_cloud_provider_api_key`
so a misconfigured judge (evals-only) can never fail production startup.
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


def test_judge_model_defaults_to_none() -> None:
    """judge_model is optional and defaults to None (falls back to llm_model)."""
    settings = _build_settings()

    assert settings.judge_model is None


def test_judge_model_accepts_valid_provider_model_format() -> None:
    """A well-formed 'provider:model' judge_model is accepted verbatim."""
    settings = _build_settings(judge_model="anthropic:claude-3-5-sonnet-20241022")

    assert settings.judge_model == "anthropic:claude-3-5-sonnet-20241022"


def test_judge_model_rejects_missing_colon() -> None:
    """judge_model must follow 'provider:model' format, same rule as llm_model."""
    with pytest.raises(ValidationError, match="must follow 'provider:model' format"):
        _build_settings(judge_model="gpt-4o")


def test_judge_model_rejects_unsupported_provider() -> None:
    """judge_model's provider must be in the same allow-list as llm_model."""
    with pytest.raises(ValidationError, match="provider must be one of"):
        _build_settings(judge_model="unsupported:model-name")


def test_judge_model_without_llm_api_key_does_not_fail_settings_construction() -> None:
    """A cloud judge_model with no llm_api_key must NOT fail at Settings() time.

    That check is deferred to `evals._build_judge_model()` (evals-only, never
    read by production `uvicorn` startup) - see
    `tests/unit/evals/test_runner.py::test_cloud_judge_model_without_llm_api_key_raises`.
    """
    settings = _build_settings(
        llm_model="ollama:granite3.3",
        llm_api_key=None,
        judge_model="anthropic:claude-3-5-sonnet-20241022",
    )

    assert settings.judge_model == "anthropic:claude-3-5-sonnet-20241022"
