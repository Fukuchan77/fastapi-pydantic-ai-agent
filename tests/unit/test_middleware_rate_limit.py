"""Unit tests for rate limiting middleware."""

import time

import pytest
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware.rate_limit import add_rate_limiting


@pytest.fixture
def app_with_rate_limit() -> FastAPI:
    """Create a FastAPI app with rate limiting for testing."""
    app = FastAPI()

    # Add rate limiting with test configuration
    # Use a very low limit for testing: 3 requests per minute
    limiter = add_rate_limiting(app, default_limits=["3/minute"])

    @app.get("/test")
    @limiter.limit("3/minute")
    async def test_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    @app.get("/unlimited")
    async def unlimited_endpoint() -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    return app


@pytest.fixture
def client(app_with_rate_limit: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app_with_rate_limit)


def test_rate_limit_allows_requests_within_limit(client: TestClient) -> None:
    """Test that requests within rate limit are allowed."""
    # First 3 requests should succeed
    for _ in range(3):
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_rate_limit_blocks_requests_exceeding_limit(client: TestClient) -> None:
    """Test that requests exceeding rate limit are blocked with 429."""
    # First 3 requests succeed
    for _ in range(3):
        response = client.get("/test")
        assert response.status_code == 200

    # 4th request should be rate limited
    response = client.get("/test")
    assert response.status_code == 429
    # Our custom error handler returns ErrorResponse with "message" and "code" fields
    assert "message" in response.json()
    assert response.json()["code"] == "RATE_LIMIT_EXCEEDED"


def test_rate_limit_headers_included_in_response(client: TestClient) -> None:
    """Test that rate limit headers are included in responses."""
    response = client.get("/test")

    assert response.status_code == 200
    # Check for standard rate limit headers
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers


def test_rate_limit_remaining_decreases(client: TestClient) -> None:
    """Test that X-RateLimit-Remaining decreases with each request."""
    # First request
    response1 = client.get("/test")
    remaining1 = int(response1.headers["X-RateLimit-Remaining"])

    # Second request
    response2 = client.get("/test")
    remaining2 = int(response2.headers["X-RateLimit-Remaining"])

    # Remaining should decrease
    assert remaining2 == remaining1 - 1


def test_rate_limit_unlimited_endpoint(client: TestClient) -> None:
    """Test that endpoints without rate limit decorator are not limited."""
    # Should be able to make many requests
    for _ in range(10):
        response = client.get("/unlimited")
        assert response.status_code == 200


def test_rate_limit_reset_after_window() -> None:
    """Test that rate limit resets after the time window expires.

    Uses its own 3/second-limited app rather than the shared `client`
    fixture's 3/minute one: now that `Limiter._inject_headers` actually
    builds `X-RateLimit-Reset` (Req 1.3), that header holds a real reset
    timestamp instead of the default `0` this test used to read, and a
    1-minute window would make this test sleep for up to a minute.
    """
    app = FastAPI()
    limiter = add_rate_limiting(app, default_limits=["3/second"])

    @app.get("/fast")
    @limiter.limit("3/second")
    async def fast_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    client = TestClient(app)

    for _ in range(3):
        response = client.get("/fast")
        assert response.status_code == 200

    response = client.get("/fast")
    assert response.status_code == 429

    # `X-RateLimit-Reset` is a real (float-string) epoch timestamp now, not
    # the always-0 fallback the old, dead header-computation code path left
    # behind.
    reset_time = float(response.headers["X-RateLimit-Reset"])
    current_time = time.time()
    wait_time = reset_time - current_time + 1  # 1s buffer

    assert 0 < wait_time < 5, f"expected a bounded ~1s wait for a 1-second window, got {wait_time}"
    time.sleep(wait_time)

    # After reset, request should succeed
    response = client.get("/fast")
    assert response.status_code == 200
