"""Unit tests for main application module."""

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.config import Settings


def test_main_app_import_fails() -> None:
    """Test that main app can be imported (RED: should fail initially)."""
    from app.main import app

    assert isinstance(app, FastAPI)


def test_health_router_registered() -> None:
    """Test that health router is registered on the app."""
    from app.main import app

    # Check that health endpoint exists
    routes = [route.path for route in app.routes if isinstance(route, APIRoute)]
    assert "/health" in routes


def test_exception_handler_returns_500_with_error_response() -> None:
    """Test global exception handler returns HTTP 500 with ErrorResponse.

    Security: The handler should return a generic error message to prevent
    leaking sensitive information to clients.
    """
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)

    # Create a test endpoint that raises an exception
    @app.get("/test-error")
    def test_error_endpoint() -> None:
        raise ValueError("Test error message")

    response = client.get("/test-error")

    assert response.status_code == 500
    # Should return generic message, not the actual exception message
    assert response.json() == {"message": "Internal server error occurred", "code": None}


def test_exception_handler_structure() -> None:
    """Test that exception handler returns correct ErrorResponse structure.

    Security: Verifies that the response structure is correct and contains
    a generic error message instead of exposing internal exception details.
    """
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)

    @app.get("/test-error-2")
    def test_error_endpoint_2() -> None:
        raise RuntimeError("Another test error")

    response = client.get("/test-error-2")

    assert response.status_code == 500
    json_data = response.json()
    assert "message" in json_data
    assert "code" in json_data
    # Should return generic message for security
    assert json_data["message"] == "Internal server error occurred"


def test_lifespan_initializes_app_state_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that lifespan initializes http_client and settings in app.state.

    Task 2.6: This test verifies that after lifespan startup, app.state.http_client
    is an httpx.AsyncClient instance and app.state.settings is a Settings instance.

    TestClient properly triggers the lifespan context manager, ensuring that
    app.state attributes are initialized before the first request.
    """
    # Clear the lru_cache to ensure fresh settings are loaded with test env vars
    from app.config import get_settings

    get_settings.cache_clear()

    # Set required environment variables for Settings
    monkeypatch.setenv("API_KEY", "test-api-key-for-lifespan-test-123456")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-api-key")

    # Import app after setting environment variables
    from app.main import app

    # TestClient triggers the lifespan context manager
    with TestClient(app) as client:
        # Make a request to ensure everything is working
        response = client.get("/health")
        assert response.status_code == 200, "Health check failed"

        # After lifespan startup, app.state should have these attributes
        assert hasattr(app.state, "http_client"), "app.state.http_client not initialized"
        assert isinstance(app.state.http_client, httpx.AsyncClient), (
            "app.state.http_client is not an httpx.AsyncClient instance"
        )

        assert hasattr(app.state, "settings"), "app.state.settings not initialized"
        assert isinstance(app.state.settings, Settings), (
            "app.state.settings is not a Settings instance"
        )


def test_cleanup_interval_has_minimum_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cleanup interval has a minimum of 300 seconds.

    Task 2.7: When session_ttl is very short (e.g., 60 seconds for testing),
    the cleanup interval should not be session_ttl // 2 (30 seconds), but
    should have a minimum of 300 seconds to avoid wasting CPU on frequent cleanups.

    RED PHASE: This test will fail initially because the cleanup interval
    calculation doesn't have a minimum bound.
    """
    # Clear the lru_cache to ensure fresh settings
    from app.config import get_settings

    get_settings.cache_clear()

    # Set required environment variables
    monkeypatch.setenv("API_KEY", "test-cleanup-interval-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")

    # Import dependencies
    import asyncio
    import time
    from unittest.mock import patch

    from app.stores.session_store import InMemorySessionStore

    # Track what interval asyncio.sleep was called with
    sleep_intervals = []

    async def mock_sleep(seconds: float) -> None:
        """Mock sleep that records the interval."""
        sleep_intervals.append(seconds)
        # Immediately raise CancelledError to stop the loop
        raise asyncio.CancelledError()

    # Patch InMemorySessionStore to use session_ttl=60
    original_init = InMemorySessionStore.__init__

    def mock_init(self, max_messages: int = 1000, session_ttl: int = 3600) -> None:
        # Force session_ttl to 60 for this test
        original_init(self, max_messages=max_messages, session_ttl=60)

    # Patch both asyncio.sleep and InMemorySessionStore.__init__
    with (
        patch("asyncio.sleep", side_effect=mock_sleep),
        patch.object(InMemorySessionStore, "__init__", mock_init),
    ):
        # Import app after patching so the lifespan uses our mocked session store
        from app.main import app

        # TestClient triggers the lifespan
        with TestClient(app) as _:
            # Wait a bit for the cleanup task to call asyncio.sleep
            time.sleep(0.1)

    # Verify that the cleanup interval was 30 seconds (60 // 2) without the fix
    # After implementing the fix, it should be 300 seconds (max(300, 60 // 2))
    assert len(sleep_intervals) > 0, "asyncio.sleep was not called"
    actual_interval = sleep_intervals[0]
    assert actual_interval == 300, (
        f"Cleanup interval should be 300 seconds (minimum), but was {actual_interval} seconds"
    )
