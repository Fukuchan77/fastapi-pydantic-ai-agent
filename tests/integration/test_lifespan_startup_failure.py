"""Startup failure must release every resource created so far, then propagate.

Task 10.3/10.4, Req 4.1, 4.4, 4.5. `_shutdown(app)` already isolates failures
between its own steps (proven by `test_lifespan_shutdown_isolation.py`); this
module proves it is also *reached* when startup itself fails partway through
- not only after a normal `yield` - leaving no running cleanup task and no
unclosed store or HTTP client behind, while the original exception still
propagates unchanged.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app
from app.stores.session_store.in_memory import InMemorySessionStore
from app.stores.vector_store.in_memory import InMemoryVectorStore


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
async def test_agent_construction_failure_releases_every_resource() -> None:
    """An injected failure building the chat agent must still clean up.

    `build_chat_agent` runs after the HTTP client, both stores, and the
    cleanup task already exist on `app.state`, so this exercises the same
    resources Req 4.6 names.
    """
    app = create_app(settings=_build_settings(), model=TestModel())

    with (
        patch.object(InMemoryVectorStore, "close", AsyncMock(return_value=None)) as vs_close,
        patch.object(InMemorySessionStore, "close", AsyncMock(return_value=None)) as ss_close,
        patch("app.lifespan.build_chat_agent", side_effect=RuntimeError("agent boom")),
        pytest.raises(RuntimeError, match="agent boom"),
    ):
        async with app.router.lifespan_context(app):
            pytest.fail("lifespan should have aborted before yielding")

    assert app.state.cleanup_task.cancelled()
    vs_close.assert_awaited_once()
    ss_close.assert_awaited_once()
    assert app.state.http_client.is_closed


@pytest.mark.asyncio
async def test_observability_configuration_failure_releases_every_resource() -> None:
    """An injected failure configuring observability must still clean up.

    `configure_logfire` runs after the chat agent is already built, so this
    covers the later failure point Req 4.6 names.
    """
    app = create_app(settings=_build_settings(), model=TestModel())

    with (
        patch.object(InMemoryVectorStore, "close", AsyncMock(return_value=None)) as vs_close,
        patch.object(InMemorySessionStore, "close", AsyncMock(return_value=None)) as ss_close,
        patch("app.lifespan.configure_logfire", side_effect=RuntimeError("logfire boom")),
        pytest.raises(RuntimeError, match="logfire boom"),
    ):
        async with app.router.lifespan_context(app):
            pytest.fail("lifespan should have aborted before yielding")

    assert app.state.cleanup_task.cancelled()
    vs_close.assert_awaited_once()
    ss_close.assert_awaited_once()
    assert app.state.http_client.is_closed
