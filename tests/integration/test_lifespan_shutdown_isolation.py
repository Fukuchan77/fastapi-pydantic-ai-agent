"""A failure closing one store during shutdown must not skip closing the rest.

Regression test: `lifespan`'s shutdown block
used to await `vector_store.close()`, `session_store.close()`, and
`http_client.aclose()` back-to-back with no error isolation, so an exception
from any one of them (e.g. a Redis connection error) would propagate out of
the `async with app.router.lifespan_context(app)` block and skip closing
everything after it - leaking the HTTP client's connection pool.
"""

from unittest.mock import AsyncMock

import pytest
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app


def _build_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "openai:gpt-4o",
        "llm_api_key": "test-llm-api-key",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_vector_store_close_failure_does_not_skip_session_store_or_http_client() -> None:
    """A raising vector_store.close() must not prevent the later shutdown steps.

    The patch is assigned inside the `async with` block (i.e. after startup,
    before shutdown), so it is active precisely when `lifespan_context`'s
    `__aexit__` runs the shutdown cleanup on exit from the block below.
    """
    app = create_app(settings=_build_settings(), model=TestModel())

    async with app.router.lifespan_context(app):
        session_store_close = AsyncMock(wraps=app.state.session_store.close)
        app.state.session_store.close = session_store_close
        http_client_aclose = AsyncMock(wraps=app.state.http_client.aclose)
        app.state.http_client.aclose = http_client_aclose
        app.state.vector_store.close = AsyncMock(side_effect=RuntimeError("boom"))

    session_store_close.assert_awaited_once()
    http_client_aclose.assert_awaited_once()
