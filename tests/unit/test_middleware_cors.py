"""Unit tests for CORS middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.cors import CORSMiddleware


@pytest.fixture
def app_with_cors() -> FastAPI:
    """Create a FastAPI app with CORS middleware for testing."""
    app = FastAPI()

    # Add CORS middleware with test configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://example.com", "https://app.example.com"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app_with_cors: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app_with_cors)


def test_cors_preflight_request_allowed_origin(client: TestClient) -> None:
    """Test CORS preflight request with allowed origin returns correct headers."""
    response = client.options(
        "/test",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-API-Key",
        },
    )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "https://example.com"
    assert response.headers["Access-Control-Allow-Credentials"] == "true"
    assert "POST" in response.headers["Access-Control-Allow-Methods"]
    assert "X-API-Key" in response.headers["Access-Control-Allow-Headers"]


def test_cors_preflight_request_disallowed_origin(client: TestClient) -> None:
    """Test CORS preflight request with disallowed origin is rejected."""
    response = client.options(
        "/test",
        headers={
            "Origin": "https://malicious.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    # Should not include CORS headers for disallowed origins
    assert "Access-Control-Allow-Origin" not in response.headers


def test_cors_actual_request_allowed_origin(client: TestClient) -> None:
    """Test actual request with allowed origin includes CORS headers."""
    response = client.get(
        "/test",
        headers={"Origin": "https://example.com"},
    )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "https://example.com"
    assert response.headers["Access-Control-Allow-Credentials"] == "true"


def test_cors_actual_request_disallowed_origin(client: TestClient) -> None:
    """Test actual request with disallowed origin does not include CORS headers."""
    response = client.get(
        "/test",
        headers={"Origin": "https://malicious.com"},
    )

    # Request should succeed but without CORS headers
    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" not in response.headers


def test_cors_wildcard_origin() -> None:
    """Test CORS middleware with wildcard origin allows all origins."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # Credentials incompatible with wildcard
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)

    response = client.get(
        "/test",
        headers={"Origin": "https://any-domain.com"},
    )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    assert "Access-Control-Allow-Credentials" not in response.headers
