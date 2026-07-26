"""Unit tests for main application module and the create_app() factory."""

import asyncio
import time
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app
from app.stores.session_store import InMemorySessionStore


def _build_settings(**overrides: object) -> Settings:
    """Build a valid Settings instance directly, without touching os.environ."""
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "openai:gpt-4",
        "llm_api_key": "test-llm-key-12345",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_main_app_can_be_imported() -> None:
    """Test that the module-level app singleton (used by uvicorn) can be imported."""
    from app.main import app

    assert isinstance(app, FastAPI)


def test_create_app_returns_fastapi_instance() -> None:
    """create_app() should return a configured FastAPI instance."""
    app = create_app(settings=_build_settings(), model=TestModel())

    assert isinstance(app, FastAPI)


def test_create_app_builds_without_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_app() with explicit settings must not require env vars (Req 8.1, 8.3).

    Tests construct the app via the factory with an explicit Settings instance
    and injected model, so no API_KEY/LLM_MODEL/LLM_API_KEY need to be set.
    """
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    app = create_app(settings=_build_settings(), model=TestModel())

    assert isinstance(app, FastAPI)


def test_create_app_does_not_call_get_settings_when_settings_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_app() must not call get_settings() when settings is explicitly provided (Req 8.1)."""
    import app.main as main_module

    def _fail_if_called() -> Settings:
        raise AssertionError("get_settings() should not be called when settings is provided")

    monkeypatch.setattr(main_module, "get_settings", _fail_if_called)

    # Must not raise: get_settings() is never invoked because settings is provided.
    create_app(settings=_build_settings(), model=TestModel())


def test_create_app_wires_injected_model_into_chat_agent() -> None:
    """create_app(model=...) must build the chat agent with the injected model (Req 8.2)."""
    test_model = TestModel()
    app = create_app(settings=_build_settings(), model=test_model)

    with TestClient(app):
        assert app.state.chat_agent.model is test_model


def test_health_router_registered() -> None:
    """Test that health router is registered on the app."""
    app = create_app(settings=_build_settings(), model=TestModel())

    routes = [route.path for route in app.routes if isinstance(route, APIRoute)]
    assert "/health" in routes


def test_exception_handler_returns_500_with_error_response() -> None:
    """Test global exception handler returns HTTP 500 with ErrorResponse.

    Security: The handler should return a generic error message to prevent
    leaking sensitive information to clients.
    """
    app = create_app(settings=_build_settings(), model=TestModel())
    client = TestClient(app, raise_server_exceptions=False)

    @app.get("/test-error")
    def test_error_endpoint() -> None:
        raise ValueError("Test error message")

    response = client.get("/test-error")

    assert response.status_code == 500
    # Should return generic message, not the actual exception message
    assert response.json() == {
        "message": "Internal server error occurred",
        "code": "INTERNAL_ERROR",
    }


def test_exception_handler_structure() -> None:
    """Test that exception handler returns correct ErrorResponse structure.

    Security: Verifies that the response structure is correct and contains
    a generic error message instead of exposing internal exception details.
    """
    app = create_app(settings=_build_settings(), model=TestModel())
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


def test_lifespan_initializes_app_state_attributes() -> None:
    """Test that lifespan initializes http_client and settings in app.state.

    TestClient properly triggers the lifespan context manager, ensuring that
    app.state attributes are initialized before the first request.
    """
    settings = _build_settings(
        api_key="test-api-key-for-lifespan-test-123456",
        llm_model="openai:gpt-4o",
    )
    app = create_app(settings=settings, model=TestModel())

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
        assert app.state.settings is settings, (
            "app.state.settings should be the exact Settings instance injected via create_app()"
        )


def test_cleanup_interval_has_minimum_bound() -> None:
    """Test that cleanup interval has a minimum of 300 seconds.

    When session_ttl is very short (e.g., 60 seconds for testing),
    the cleanup interval should not be session_ttl // 2 (30 seconds), but
    should have a minimum of 300 seconds to avoid wasting CPU on frequent cleanups.
    """
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
        app = create_app(settings=_build_settings(), model=TestModel())

        # TestClient triggers the lifespan
        with TestClient(app) as _:
            # Wait a bit for the cleanup task to call asyncio.sleep
            time.sleep(0.1)

    # Verify that the cleanup interval was 300 seconds (max(300, 60 // 2))
    assert len(sleep_intervals) > 0, "asyncio.sleep was not called"
    actual_interval = sleep_intervals[0]
    assert actual_interval == 300, (
        f"Cleanup interval should be 300 seconds (minimum), but was {actual_interval} seconds"
    )


def test_cors_middleware_respects_injected_settings() -> None:
    """Test that CORS middleware uses the cors_origins from the injected Settings.

    Security: CORS should use configured origins, not wildcard. Because settings
    are now injected explicitly via create_app(), this test controls the exact
    configured origin instead of depending on whatever default was baked in at
    module import time.
    """
    settings = _build_settings(cors_origins=["http://localhost:3000"])
    app = create_app(settings=settings, model=TestModel())

    with TestClient(app) as client:
        # Test with the configured origin
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )

        assert response.status_code == 200
        # CORS header should be set for configured origin
        assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

        # Test with disallowed origin (not in settings)
        response2 = client.get(
            "/health",
            headers={"Origin": "https://malicious-site.com"},
        )

        assert response2.status_code == 200
        # CORS header should NOT be set for disallowed origin
        assert "Access-Control-Allow-Origin" not in response2.headers
