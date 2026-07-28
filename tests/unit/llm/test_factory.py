"""Unit tests for `app/llm/factory.py` (Task 7: FallbackModel + NativeOutput gate).

Covers Req 10.1 (eager `FallbackModel` chain construction from settings) and
the `supports_native_output()` capability gate (Req 10.2/10.3), including the
`FallbackModel`-has-no-profile special case.
"""

from unittest.mock import patch

import pytest
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile

from app.config import Settings
from app.llm.factory import build_fallback_model
from app.llm.factory import supports_native_output


def _build_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "openai:gpt-4o",
        "llm_api_key": "test-llm-api-key-12345",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class TestBuildFallbackModel:
    """build_fallback_model always returns a FallbackModel, chain-of-one by default."""

    def test_returns_fallback_model_wrapping_default_model_only(self) -> None:
        """With no llm_fallback_models configured, the chain has exactly one model."""
        settings = _build_settings()

        model = build_fallback_model(settings)

        assert isinstance(model, FallbackModel)
        assert len(model.models) == 1

    def test_includes_configured_fallbacks_in_order(self) -> None:
        """Fallback models are appended after the default model, in configured order."""
        settings = _build_settings(
            llm_fallback_models=["anthropic:claude-3-5-sonnet-20241022", "groq:llama3-70b-8192"]
        )

        model = build_fallback_model(settings)

        assert len(model.models) == 3
        assert model.models[0].model_name == "openai/gpt-4o"
        assert model.models[1].model_name == "anthropic/claude-3-5-sonnet-20241022"
        assert model.models[2].model_name == "groq/llama3-70b-8192"

    def test_propagates_default_model_build_failure_eagerly(self) -> None:
        """A misconfigured default model fails while building the chain, not on first use."""
        settings = _build_settings()

        with (
            patch("app.agents.chat_agent.LiteLLMModel", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            build_fallback_model(settings)


class TestSupportsNativeOutput:
    """supports_native_output reads model.profile.supports_json_schema_output."""

    def test_test_model_default_profile_reports_false(self) -> None:
        """TestModel's default profile does not support JSON-schema output."""
        assert supports_native_output(TestModel()) is False

    def test_model_with_explicit_true_profile_reports_true(self) -> None:
        """A model whose profile explicitly reports the capability is honored."""
        model = TestModel(profile=ModelProfile(supports_json_schema_output=True))

        assert supports_native_output(model) is True

    def test_function_model_default_profile_reports_true(self) -> None:
        """FunctionModel defaults to supports_json_schema_output=True unless overridden."""

        def _echo(messages: list, info: object) -> object:  # pragma: no cover - never called
            raise NotImplementedError

        assert supports_native_output(FunctionModel(_echo)) is True

    def test_fallback_model_reads_primary_models_profile(self) -> None:
        """FallbackModel.profile raises NotImplementedError; the gate reads models[0] instead."""
        primary = TestModel(profile=ModelProfile(supports_json_schema_output=True))
        secondary = TestModel(profile=ModelProfile(supports_json_schema_output=False))
        fallback = FallbackModel(primary, secondary)

        with pytest.raises(NotImplementedError):
            _ = fallback.profile  # sanity: confirms the special case is real

        assert supports_native_output(fallback) is True

    def test_fallback_model_false_when_primary_model_does_not_support_it(self) -> None:
        """The gate follows the primary model, even if a later fallback would support it."""
        primary = TestModel(profile=ModelProfile(supports_json_schema_output=False))
        secondary = TestModel(profile=ModelProfile(supports_json_schema_output=True))
        fallback = FallbackModel(primary, secondary)

        assert supports_native_output(fallback) is False
