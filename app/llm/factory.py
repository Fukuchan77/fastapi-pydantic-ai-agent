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
        build_model(settings.model_copy(update={"llm_model": model_id}))
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
