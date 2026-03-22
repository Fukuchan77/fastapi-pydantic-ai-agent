"""Unit tests for rate limiting middleware with proxy support."""

import pytest
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware.rate_limit import add_rate_limiting


@pytest.fixture
def app_with_rate_limit_proxy() -> FastAPI:
    """Create a FastAPI app with rate limiting that considers proxy headers."""
    app = FastAPI()

    # Add rate limiting with test configuration
    # Use a very low limit for testing: 2 requests per minute
    limiter = add_rate_limiting(app, default_limits=["2/minute"])

    @app.get("/test")
    @limiter.limit("2/minute")
    async def test_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    @app.get("/health")
    async def health_endpoint() -> JSONResponse:
        return JSONResponse(content={"status": "healthy"})

    return app


@pytest.fixture
def client(app_with_rate_limit_proxy: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app_with_rate_limit_proxy)


def test_rate_limit_considers_x_forwarded_for_header(client: TestClient) -> None:
    """Test that rate limiting uses X-Forwarded-For header when present.

    This test verifies that requests from different IPs in the X-Forwarded-For
    header are tracked separately, which is important for apps behind proxies/load balancers.
    """
    # Request from IP 1.2.3.4 via X-Forwarded-For
    for _ in range(2):
        response = client.get("/test", headers={"X-Forwarded-For": "1.2.3.4"})
        assert response.status_code == 200

    # 3rd request from same IP should be rate limited
    response = client.get("/test", headers={"X-Forwarded-For": "1.2.3.4"})
    assert response.status_code == 429

    # But request from different IP should succeed
    response = client.get("/test", headers={"X-Forwarded-For": "5.6.7.8"})
    assert response.status_code == 200


def test_rate_limit_uses_first_ip_in_forwarded_chain(client: TestClient) -> None:
    """Test that rate limiting uses the first IP in X-Forwarded-For chain.

    When multiple proxies are involved, X-Forwarded-For contains a comma-separated
    list of IPs. The first IP is the actual client IP.
    """
    # Request with proxy chain: client -> proxy1 -> proxy2
    for _ in range(2):
        response = client.get(
            "/test", headers={"X-Forwarded-For": "10.0.0.1, 192.168.1.1, 172.16.0.1"}
        )
        assert response.status_code == 200

    # 3rd request from same client IP (first in chain) should be rate limited
    response = client.get("/test", headers={"X-Forwarded-For": "10.0.0.1, 192.168.2.2, 172.16.0.2"})
    assert response.status_code == 429


def test_rate_limit_fallback_to_remote_address_without_forwarded(client: TestClient) -> None:
    """Test that rate limiting falls back to remote address when X-Forwarded-For is absent."""
    # Requests without X-Forwarded-For should use remote address
    for _ in range(2):
        response = client.get("/test")
        assert response.status_code == 200

    # 3rd request should be rate limited
    response = client.get("/test")
    assert response.status_code == 429


def test_health_endpoint_not_rate_limited(client: TestClient) -> None:
    """Test that /health endpoint is not subject to rate limiting.

    Health check endpoints should not be rate limited to allow monitoring systems
    to check service health without being blocked.
    """
    # Should be able to make many requests to /health
    for _ in range(10):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    # Health endpoint should not have rate limit headers
    response = client.get("/health")
    assert "X-RateLimit-Limit" not in response.headers
