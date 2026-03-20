"""Unit tests for main application module."""

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient


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
