"""Integration tests: create_app()'s lifespan builds the FallbackModel chain eagerly.

Task 7.2, Req 10.1. A test-injected `model` override bypasses it entirely
(existing test-isolation contract, unchanged).
"""

from unittest.mock import patch

import pytest
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app


def _build_settings(**overrides: object) -> Settings:
    """Build a valid Settings instance directly, without touching os.environ."""
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "openai:gpt-4o",
        "llm_api_key": "test-llm-api-key-12345",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_lifespan_builds_fallback_model_when_no_model_override() -> None:
    """Without a test model override, the chat agent's model is a FallbackModel chain."""
    app = create_app(settings=_build_settings())

    async with app.router.lifespan_context(app):
        assert isinstance(app.state.chat_agent.model, FallbackModel)


@pytest.mark.asyncio
async def test_lifespan_model_override_skips_fallback_model_build() -> None:
    """A test-injected model bypasses build_fallback_model entirely (existing contract)."""
    test_model = TestModel()
    app = create_app(settings=_build_settings(), model=test_model)

    async with app.router.lifespan_context(app):
        assert app.state.chat_agent.model is test_model


@pytest.mark.asyncio
async def test_lifespan_aborts_startup_when_fallback_model_build_fails() -> None:
    """A misconfigured provider chain must fail startup, not defer to the first request."""
    app = create_app(settings=_build_settings())

    with (
        patch("app.main.build_fallback_model", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        async with app.router.lifespan_context(app):
            pytest.fail("lifespan should have aborted before yielding")
