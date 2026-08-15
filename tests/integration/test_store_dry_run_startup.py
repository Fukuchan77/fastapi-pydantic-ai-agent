"""Integration tests: a failing store dry-run aborts application startup (Req 5.4, 5.5).

Uses a real `OllamaEmbeddingVectorStore` pointed at an unreachable local port
(nothing listens there, so the connection is refused immediately - hermetic,
no real network dependency) to prove `create_app()`'s lifespan fails startup
rather than deferring the error to the first request.
"""

import pytest
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app
from app.stores.factory import StoreDryRunError


def _build_settings(**overrides: object) -> Settings:
    """Build a valid Settings instance directly, without touching os.environ.

    `embedding_model`/`embedding_base_url` default to `None` explicitly so a
    developer's local `.env` values for those fields can never leak in via
    pydantic-settings' env-file fallback.
    """
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "openai:gpt-4o",
        "llm_api_key": "test-llm-api-key-12345",
        "embedding_model": None,
        "embedding_base_url": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_lifespan_aborts_startup_when_vector_store_dry_run_fails() -> None:
    """An unreachable Ollama vector store backend must fail startup, not defer to first request."""
    settings = _build_settings(
        vector_store_backend="ollama",
        embedding_model="nomic-embed-text:latest",
        # Port 1 is a reserved, unassigned port: nothing listens there, so the
        # connection is refused immediately without needing a real network.
        embedding_base_url="http://localhost:1/v1",
    )
    app = create_app(settings=settings, model=TestModel())

    with pytest.raises(StoreDryRunError):
        async with app.router.lifespan_context(app):
            pytest.fail("lifespan should have aborted before yielding")


@pytest.mark.asyncio
async def test_lifespan_starts_successfully_with_default_in_memory_stores() -> None:
    """The default in-memory store selection is unaffected by the dry-run gate."""
    settings = _build_settings()
    app = create_app(settings=settings, model=TestModel())

    async with app.router.lifespan_context(app):
        assert hasattr(app.state, "vector_store")
        assert hasattr(app.state, "session_store")
