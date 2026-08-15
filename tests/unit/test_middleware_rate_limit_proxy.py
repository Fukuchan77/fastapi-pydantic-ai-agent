"""Unit tests for rate limiting middleware with proxy support."""

import pytest
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware.rate_limit import add_rate_limiting


# Simulates a request arriving from a trusted proxy. `trusted_proxies` accepts
# only IP addresses and CIDR networks, so TestClient must be given a real source
# address rather than its default "testclient" host.
TRUSTED_PROXY_CLIENT = ("10.0.0.1", 12345)


@pytest.fixture
def app_with_rate_limit_proxy(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Create a FastAPI app with rate limiting that considers proxy headers."""
    # Configure trusted proxies to cover the simulated proxy address below.
    # A CIDR network is used here so this fixture also exercises network matching.
    # This allows the test to properly test X-Forwarded-For header handling
    # Note: trusted_proxies is a list[str], so we need to provide JSON format
    monkeypatch.setenv("TRUSTED_PROXIES", '["10.0.0.0/8"]')

    # Clear the settings cache so the new environment variable is picked up
    from app.config import get_settings

    get_settings.cache_clear()

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
    return TestClient(app_with_rate_limit_proxy, client=TRUSTED_PROXY_CLIENT)


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


def test_rate_limit_uses_last_untrusted_ip_in_forwarded_chain(client: TestClient) -> None:
    """Test that rate limiting keys on the last untrusted IP in the X-Forwarded-For chain.

    With multiple proxies, X-Forwarded-For is a comma-separated chain that every
    documented proxy *appends* to, so its leftmost element is whatever the client
    itself sent. The chain is walked right-to-left, skipping hops inside
    `trusted_proxies` (here `10.0.0.0/8`), and the first address that is not
    becomes the bucket key.

    The third request below varies the leftmost element - the part a caller
    controls - while keeping the observed address the same, and must still be
    rate limited. If the leftmost element were the key, a caller could rotate it
    and never exhaust a budget.
    """
    # Request with proxy chain: <client claim>, <observed address>, <trusted hop>
    for _ in range(2):
        response = client.get(
            "/test", headers={"X-Forwarded-For": "1.1.1.1, 203.0.113.9, 10.0.0.9"}
        )
        assert response.status_code == 200

    # 3rd request: different claimed prefix, same observed address -> same bucket
    response = client.get("/test", headers={"X-Forwarded-For": "9.9.9.9, 203.0.113.9, 10.0.0.9"})
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
