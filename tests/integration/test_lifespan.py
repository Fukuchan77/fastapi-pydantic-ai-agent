"""Integration tests for FastAPI application lifespan management."""

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestLifespanManagement:
    """Test application lifespan startup and shutdown behavior."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_session_store(self, monkeypatch) -> None:
        """Lifespan must initialize session_store in app.state."""
        # Set required environment variables for the lifespan to work
        monkeypatch.setenv("API_KEY", "test-api-key-12345")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-api-key")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

        # Import app and lifespan to test them directly
        from app.main import app
        from app.main import lifespan

        # Manually invoke the lifespan context manager to verify it works
        async with lifespan(app):
            # During the lifespan context, app.state should have session_store
            assert hasattr(app.state, "session_store"), "session_store should be initialized"
            assert app.state.session_store is not None

    @pytest.mark.asyncio
    async def test_session_cleanup_task_is_running(self, monkeypatch) -> None:
        """Lifespan must start a background cleanup task for expired sessions."""
        # Set required environment variables for the lifespan to work
        monkeypatch.setenv("API_KEY", "test-api-key-12345")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-api-key")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

        # Import app and lifespan to test them directly
        from app.main import app
        from app.main import lifespan

        # Manually invoke the lifespan context manager
        async with lifespan(app):
            # Verify session_store exists
            assert hasattr(app.state, "session_store")
            session_store = app.state.session_store

            # Save a session that will expire quickly
            # We need a store with short TTL for testing
            await session_store.save_history("test-session", [])

            # Verify session exists initially
            history = await session_store.get_history("test-session")
            assert history == []

            # For a full integration test, we would wait for TTL to expire
            # and verify the cleanup task removes it. However, this would
            # require waiting for the default TTL (3600 seconds) which is
            # not practical for testing.
            #
            # Instead, we verify the cleanup mechanism exists and can be called:
            # The cleanup_expired_sessions method should be callable
            assert hasattr(session_store, "cleanup_expired_sessions")
            assert callable(session_store.cleanup_expired_sessions)

            # We can also verify the cleanup task attribute exists on app.state
            # This proves the background task was created during lifespan
            assert hasattr(app.state, "cleanup_task"), (
                "cleanup_task should be stored in app.state to prove background task was created"
            )
            assert app.state.cleanup_task is not None
            assert isinstance(app.state.cleanup_task, asyncio.Task)

            # Verify the task is not done (still running in background)
            assert not app.state.cleanup_task.done(), (
                "cleanup_task should be running in the background"
            )

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_cancels_background_task(self, monkeypatch) -> None:
        """Lifespan shutdown must cancel the cleanup background task."""
        # Set required environment variables for the lifespan to work
        monkeypatch.setenv("API_KEY", "test-api-key-12345")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-api-key")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

        # Create a new app instance for this test to have clean state
        from fastapi import FastAPI

        from app.main import lifespan

        test_app = FastAPI()

        # Manually invoke the lifespan context manager
        async with lifespan(test_app):
            # Inside the context, cleanup_task should exist and be running
            assert hasattr(test_app.state, "cleanup_task")
            cleanup_task = test_app.state.cleanup_task
            assert not cleanup_task.done()

        # After exiting the context (lifespan shutdown), task should be cancelled
        # Wait a tiny bit for cancellation to complete
        await asyncio.sleep(0.1)

        assert cleanup_task.done(), "cleanup_task should be done after lifespan shutdown"
        # The task should be cancelled, not just finished normally
        assert cleanup_task.cancelled() or cleanup_task.exception() is not None, (
            "cleanup_task should be cancelled during shutdown"
        )

    @pytest.mark.asyncio
    @patch("app.observability.logfire.instrument_pydantic_ai")
    @patch("app.observability.logfire.configure")
    async def test_lifespan_configures_observability(
        self,
        mock_logfire_configure: MagicMock,
        mock_instrument_pydantic: MagicMock,
        monkeypatch,
    ) -> None:
        """Lifespan must configure Logfire observability during startup."""
        # Set required environment variables
        monkeypatch.setenv("API_KEY", "test-api-key-12345")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-api-key")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LOGFIRE_TOKEN", "test-logfire-token")

        # Clear the settings cache so new settings with LOGFIRE_TOKEN are loaded
        from app.config import get_settings

        get_settings.cache_clear()

        # Create a new app instance for this test
        from fastapi import FastAPI

        from app.main import lifespan

        test_app = FastAPI()

        # Manually invoke the lifespan context manager
        async with lifespan(test_app):
            # Verify logfire.configure was called with token and service_name
            mock_logfire_configure.assert_called_once_with(
                token="test-logfire-token",  # noqa: S106
                service_name="fastapi-pydantic-ai-agent",
            )
            # Verify logfire.instrument_pydantic_ai was called
            mock_instrument_pydantic.assert_called_once()
