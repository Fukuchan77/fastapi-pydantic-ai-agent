"""LLM provider factory: eager FallbackModel chain + NativeOutput capability gate.

Wraps `app.agents.chat_agent.build_model()` (the single-model builder, still
the source of truth for the LiteLLM provider-routing logic) in a
`pydantic_ai.models.fallback.FallbackModel` chain, and exposes a model's
JSON-schema-output capability so the chat agent can gate `NativeOutput`
accordingly (Req 10).
"""

from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel

from app.config import Settings


def settings_for_model_id(settings: Settings, model_id: str) -> Settings:
    """Return a `settings` copy for building a *different* model than `llm_model`.

    Shared by `build_fallback_model()` (each fallback) and
    `evals.runner._build_judge_model()` (an independent judge model):
    both build a model other than `settings.llm_model` via
    `settings.model_copy(update={"llm_model": ...})`, which does not
    re-run validators.

    `llm_base_url` is provider-specific (e.g. an Azure OpenAI endpoint or a
    self-hosted LiteLLM proxy) — carrying it over unmodified when `model_id`
    names a *different* provider than `settings.llm_model` would silently
    point that provider's client at the wrong endpoint. It is preserved only
    when both share the same provider (e.g. two Ollama models against the
    same local server).

    Args:
        settings: The base settings, providing `llm_model`/`llm_base_url`.
        model_id: The "provider:model" identifier to build instead.

    Returns:
        Settings: A copy with `llm_model` set to `model_id`, and
        `llm_base_url` cleared if `model_id`'s provider differs from
        `settings.llm_model`'s.
    """
    primary_provider = settings.llm_model.split(":", 1)[0]
    target_provider = model_id.split(":", 1)[0]
    updates: dict[str, object] = {"llm_model": model_id}
    if target_provider != primary_provider:
        updates["llm_base_url"] = None
    return settings.model_copy(update=updates)


def build_fallback_model(settings: Settings) -> FallbackModel:
    """Build a `FallbackModel` chain from `llm_model` + `llm_fallback_models`.

    Always returns a `FallbackModel`, even when no fallback models are
    configured (a chain of one), so misconfiguration is discovered eagerly
    when this is called during lifespan startup rather than deferred to the
    first request.

    Args:
        settings: Application settings providing the model chain.

    Returns:
        FallbackModel wrapping the primary model and any configured fallbacks.
    """
    # Deferred import: app.agents.chat_agent imports supports_native_output
    # from this module, so importing build_model at module level here would
    # create a circular import.
    from app.agents.chat_agent import build_model

    default_model = build_model(settings)
    fallback_models: list[Model] = [
        build_model(settings_for_model_id(settings, model_id))
        for model_id in settings.llm_fallback_models
    ]
    return FallbackModel(default_model, *fallback_models)


def supports_native_output(model: Model) -> bool:
    """Report whether `model` can produce grammar-constrained JSON-schema output.

    `FallbackModel.profile` raises `NotImplementedError` ("FallbackModel does
    not have its own model profile.") — capability is read from its primary
    (first) model instead, since that's the model actually used absent a
    failure.

    Args:
        model: The model to check (a plain `Model` or a `FallbackModel`).

    Returns:
        True if the model profile reports `supports_json_schema_output`.
    """
    target = model.models[0] if isinstance(model, FallbackModel) else model
    return target.profile.supports_json_schema_output
