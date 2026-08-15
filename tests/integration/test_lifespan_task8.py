"""Integration tests for Wiring components in app/main.py lifespan."""

import pytest
from fastapi.routing import APIRoute
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app
from app.stores.vector_store import InMemoryVectorStore


def _build_settings(**overrides: object) -> Settings:
    """Build a valid Settings instance directly, without touching os.environ."""
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "ollama:test-model",
        "llm_api_key": "test-llm-api-key",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class TestTask8ComponentWiring:
    """Test that vector_store, chat_agent, and v1 router are properly initialized."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_vector_store(self) -> None:
        """Lifespan must initialize vector_store in app.state."""
        app = create_app(settings=_build_settings(), model=TestModel())

        async with app.router.lifespan_context(app):
            assert hasattr(app.state, "vector_store"), "vector_store should be initialized"
            assert app.state.vector_store is not None
            # Verify it's the correct type
            assert isinstance(app.state.vector_store, InMemoryVectorStore)

    @pytest.mark.asyncio
    async def test_lifespan_initializes_chat_agent(self) -> None:
        """Lifespan must initialize chat_agent in app.state, wired with the injected model."""
        test_model = TestModel()
        app = create_app(settings=_build_settings(), model=test_model)

        async with app.router.lifespan_context(app):
            assert hasattr(app.state, "chat_agent"), "chat_agent should be initialized"
            assert app.state.chat_agent is not None
            # Verify it's a Pydantic AI Agent built with the injected model
            assert isinstance(app.state.chat_agent, Agent)
            assert app.state.chat_agent.model is test_model

    def test_v1_router_is_registered(self) -> None:
        """Test that v1 router is included in the app."""
        app = create_app(settings=_build_settings(), model=TestModel())

        # Check that v1 routes are registered by filtering APIRoute instances
        route_paths = [route.path for route in app.routes if isinstance(route, APIRoute)]

        # Expected v1 routes
        expected_v1_routes = [
            "/v1/agent/chat",
            "/v1/agent/stream",
            "/v1/rag/query",
            "/v1/rag/ingest",
        ]

        for expected_route in expected_v1_routes:
            assert expected_route in route_paths, f"Route {expected_route} should be registered"
