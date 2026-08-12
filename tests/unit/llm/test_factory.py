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
from app.llm.factory import settings_for_model_id
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

    def test_cross_provider_fallback_does_not_inherit_llm_base_url(self) -> None:
        """A fallback on a different provider must not reuse the primary's base_url.

        `llm_base_url` is provider-specific (e.g. a self-hosted LiteLLM proxy) -
        carrying it over unmodified would point the fallback's (different-
        provider) client at the wrong endpoint.
        """
        settings = _build_settings(
            llm_model="ollama:granite3.3",
            llm_base_url="http://localhost:11434",
            llm_fallback_models=["anthropic:claude-3-5-sonnet-20241022"],
        )

        model = build_fallback_model(settings)

        fallback = model.models[1]
        assert fallback.model_name == "anthropic/claude-3-5-sonnet-20241022"
        assert fallback.settings is None or "litellm_api_base" not in fallback.settings

    def test_same_provider_fallback_keeps_llm_base_url(self) -> None:
        """A fallback on the SAME provider as the primary should keep the shared base_url."""
        settings = _build_settings(
            llm_model="ollama:granite3.3",
            llm_base_url="http://localhost:11434",
            llm_fallback_models=["ollama:llama3.1"],
        )

        model = build_fallback_model(settings)

        fallback = model.models[1]
        assert fallback.settings is not None
        assert fallback.settings["litellm_api_base"] == "http://localhost:11434/"


class TestSettingsForModelId:
    """settings_for_model_id() clears llm_base_url only across a provider change."""

    def test_clears_base_url_when_provider_differs(self) -> None:
        """A different provider's model_id must not inherit the primary's base_url."""
        settings = _build_settings(
            llm_model="ollama:granite3.3", llm_base_url="http://localhost:11434"
        )

        derived = settings_for_model_id(settings, "anthropic:claude-3-5-sonnet-20241022")

        assert derived.llm_model == "anthropic:claude-3-5-sonnet-20241022"
        assert derived.llm_base_url is None

    def test_keeps_base_url_when_provider_matches(self) -> None:
        """A same-provider model_id keeps the shared base_url (e.g. two local Ollama models)."""
        settings = _build_settings(
            llm_model="ollama:granite3.3", llm_base_url="http://localhost:11434"
        )

        derived = settings_for_model_id(settings, "ollama:llama3.1")

        assert derived.llm_model == "ollama:llama3.1"
        assert str(derived.llm_base_url) == "http://localhost:11434/"


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

    def test_mapping_shaped_profile_reporting_capability_reports_true(self) -> None:
        """A v2-style mapping-shaped profile reporting the capability is honored (R2.4).

        `pydantic-ai` 2.x turns `ModelProfile` into a `TypedDict`, so a model's
        `.profile` can be a plain mapping instead of a dataclass instance.
        `cached_property` only computes a value when the instance `__dict__`
        doesn't already hold one under that name, so assigning directly
        overrides it without going through `Model.profile`'s dataclass-specific
        internals - exercising exactly the mapping shape the read must tolerate.
        """
        model = TestModel()
        model.profile = {"supports_json_schema_output": True}

        assert supports_native_output(model) is True

    def test_mapping_shaped_profile_missing_field_reports_false(self) -> None:
        """A mapping-shaped profile omitting the field is treated as unsupported (R2.2).

        `.get(field, False)` must yield `False` directly - no `None` sentinel,
        no `KeyError` - for a v2-shaped profile that simply never sets the key.
        """
        model = TestModel()
        model.profile = {}

        assert supports_native_output(model) is False

    def test_object_shaped_profile_missing_field_reports_false(self) -> None:
        """A 1.x-shaped profile object omitting the field is treated as unsupported (R2.2).

        A bare `object()` is attribute-accessed, not mapping-accessed, and has
        no `supports_json_schema_output` attribute at all - proving the
        `getattr(obj, field, False)` fallback branch, independent of
        `ModelProfile`'s own dataclass default for that field.
        """
        model = TestModel()
        model.profile = object()

        assert supports_native_output(model) is False

    def test_fallback_model_reads_primary_mapping_shaped_profile(self) -> None:
        """FallbackModel's primary-model read holds for a mapping-shaped profile (R2.1, R2.3).

        Combines the `FallbackModel` special case with the v2 mapping shape:
        the primary model's mapping-shaped profile must still be the one read,
        even though the secondary's dataclass-shaped profile disagrees.
        """
        primary = TestModel()
        primary.profile = {"supports_json_schema_output": True}
        secondary = TestModel(profile=ModelProfile(supports_json_schema_output=False))
        fallback = FallbackModel(primary, secondary)

        assert supports_native_output(fallback) is True

    def test_shared_test_model_fixture_keeps_capability_disabled(
        self, test_model: FunctionModel
    ) -> None:
        """The shared `test_model` fixture stays on the plain-text path (R2.8).

        Guards against this change silently moving the suite onto the
        native-output path: the fixture's explicit
        `ModelProfile(supports_json_schema_output=False)` must still read as
        `False` through `supports_native_output`.
        """
        assert supports_native_output(test_model) is False
