"""Unit tests for security headers middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.security_headers import SecurityHeadersMiddleware


@pytest.fixture
def app_with_security_headers() -> FastAPI:
    """Create a FastAPI app with security headers middleware for testing."""
    app = FastAPI()

    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)  # type: ignore[arg-type]

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app_with_security_headers: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app_with_security_headers)


def test_security_headers_included_in_response(client: TestClient) -> None:
    """Test that security headers are included in all responses."""
    response = client.get("/test")

    assert response.status_code == 200

    # Check for essential security headers
    assert "X-Content-Type-Options" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"

    assert "X-Frame-Options" in response.headers
    assert response.headers["X-Frame-Options"] == "DENY"

    # Task 20.5: X-XSS-Protection removed - deprecated in modern browsers
    # CSP supersedes it (tested below)
    assert "X-XSS-Protection" not in response.headers

    assert "Referrer-Policy" in response.headers
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_security_headers_on_error_responses(client: TestClient) -> None:
    """Test that security headers are included even in error responses."""
    response = client.get("/nonexistent")

    assert response.status_code == 404

    # Security headers should be present even on error responses
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers


def test_hsts_header_included(client: TestClient) -> None:
    """Test that Strict-Transport-Security header is included."""
    response = client.get("/test")

    assert response.status_code == 200
    assert "Strict-Transport-Security" in response.headers
    # Should have max-age and includeSubDomains
    hsts = response.headers["Strict-Transport-Security"]
    assert "max-age=" in hsts
    assert "includeSubDomains" in hsts


def test_csp_header_included(client: TestClient) -> None:
    """Test that Content-Security-Policy header is included."""
    response = client.get("/test")

    assert response.status_code == 200
    assert "Content-Security-Policy" in response.headers
    # Should have at least default-src directive
    csp = response.headers["Content-Security-Policy"]
    assert "default-src" in csp


def test_permissions_policy_header_included(client: TestClient) -> None:
    """Test that Permissions-Policy header is included."""
    response = client.get("/test")

    assert response.status_code == 200
    assert "Permissions-Policy" in response.headers
    # Should restrict sensitive features
    permissions = response.headers["Permissions-Policy"]
    assert "geolocation=" in permissions or "camera=" in permissions


def test_custom_security_headers() -> None:
    """Test that custom security headers can be configured."""
    app = FastAPI()

    # Add security headers middleware with custom headers
    custom_headers = {
        "X-Custom-Header": "custom-value",
        "X-Frame-Options": "SAMEORIGIN",  # Override default
    }
    app.add_middleware(  # type: ignore[arg-type]
        SecurityHeadersMiddleware,
        custom_headers=custom_headers,
    )

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    assert response.headers["X-Custom-Header"] == "custom-value"
    assert response.headers["X-Frame-Options"] == "SAMEORIGIN"


def test_x_xss_protection_not_included() -> None:
    """Test that X-XSS-Protection header is NOT included.

    Task 20.5: X-XSS-Protection header was removed from modern browsers
    (Chrome 2019) and can cause XSS vulnerabilities in older IE versions.
    Content-Security-Policy supersedes it and provides better protection.
    """
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)  # type: ignore[arg-type]

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    # X-XSS-Protection should NOT be present (deprecated)
    assert "X-XSS-Protection" not in response.headers
    # CSP should be present instead (provides better XSS protection)
    assert "Content-Security-Policy" in response.headers
