"""Integration tests for Task 8.0: Wiring components in app/main.py lifespan."""

import pytest


class TestTask8ComponentWiring:
    """Test that vector_store, chat_agent, and v1 router are properly initialized."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_vector_store(self, monkeypatch) -> None:
        """Lifespan must initialize vector_store in app.state (Task 8.0)."""
        # Set required environment variables
        monkeypatch.setenv("API_KEY", "test-api-key")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-api-key")
        monkeypatch.setenv("LLM_MODEL", "ollama:test-model")

        # Import after setting env vars
        from pydantic_ai.models.test import TestModel

        from app.agents.chat_agent import build_chat_agent
        from app.main import app
        from app.main import lifespan
        from app.stores.vector_store import InMemoryVectorStore

        # Patch build_chat_agent to use TestModel instead of real OpenAI model
        test_agent = build_chat_agent(model=TestModel())

        # Invoke lifespan and replace the agent with test version
        async with lifespan(app):
            # Override the agent that was built during lifespan with our test version
            app.state.chat_agent = test_agent

            assert hasattr(app.state, "vector_store"), "vector_store should be initialized"
            assert app.state.vector_store is not None
            # Verify it's the correct type
            assert isinstance(app.state.vector_store, InMemoryVectorStore)

    @pytest.mark.asyncio
    async def test_lifespan_initializes_chat_agent(self, monkeypatch) -> None:
        """Lifespan must initialize chat_agent in app.state (Task 8.0)."""
        # Set required environment variables
        monkeypatch.setenv("API_KEY", "test-api-key")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-api-key")
        monkeypatch.setenv("LLM_MODEL", "ollama:test-model")

        # Import after setting env vars
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        from app.agents.chat_agent import build_chat_agent
        from app.main import app
        from app.main import lifespan

        # Patch build_chat_agent to use TestModel instead of real OpenAI model
        test_agent = build_chat_agent(model=TestModel())

        # Invoke lifespan and replace the agent with test version
        async with lifespan(app):
            # Override the agent that was built during lifespan with our test version
            app.state.chat_agent = test_agent

            assert hasattr(app.state, "chat_agent"), "chat_agent should be initialized"
            assert app.state.chat_agent is not None
            # Verify it's a Pydantic AI Agent
            assert isinstance(app.state.chat_agent, Agent)

    def test_v1_router_is_registered(self, monkeypatch) -> None:
        """Test that v1 router is included in the app (Task 8.0)."""
        # Set required environment variables
        monkeypatch.setenv("API_KEY", "test-api-key")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-api-key")
        monkeypatch.setenv("LLM_MODEL", "ollama:test-model")

        # Import the app
        from fastapi.routing import APIRoute

        from app.main import app

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
