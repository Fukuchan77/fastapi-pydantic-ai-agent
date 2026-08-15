"""Unit tests for security headers middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.security_headers import SecurityHeadersMiddleware
from tests.conftest import build_test_settings


@pytest.fixture
def app_with_security_headers() -> FastAPI:
    """Create a FastAPI app with security headers middleware for testing."""
    app = FastAPI()

    # Add security headers middleware
    app.add_middleware(  # type: ignore[arg-type]
        SecurityHeadersMiddleware,
        settings=build_test_settings(),
    )

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app_with_security_headers: FastAPI) -> TestClient:
    """Create test client speaking plain HTTP (default TestClient base_url)."""
    return TestClient(app_with_security_headers)


@pytest.fixture
def https_client(app_with_security_headers: FastAPI) -> TestClient:
    """Create test client whose requests carry a secure ASGI scope.

    `base_url="https://testserver"` makes the underlying ASGI transport build
    `scope["scheme"] == "https"`, driving the real per-request scheme check
    rather than reading a forwarded header (Req 11.3/11.4, ADR-5).
    """
    return TestClient(app_with_security_headers, base_url="https://testserver")


def test_security_headers_included_in_response(client: TestClient) -> None:
    """Test that security headers are included in all responses."""
    response = client.get("/test")

    assert response.status_code == 200

    # Check for essential security headers
    assert "X-Content-Type-Options" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"

    assert "X-Frame-Options" in response.headers
    assert response.headers["X-Frame-Options"] == "DENY"

    # X-XSS-Protection removed - deprecated in modern browsers
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


def test_hsts_header_absent_over_plaintext(client: TestClient) -> None:
    """Strict-Transport-Security must be omitted entirely over plain HTTP (Req 11.3).

    Sending HSTS unconditionally over plaintext is worse than not sending it:
    it asserts a promise ("this host is always HTTPS") that a plaintext
    response cannot back up.
    """
    response = client.get("/test")

    assert response.status_code == 200
    assert "Strict-Transport-Security" not in response.headers


def test_hsts_header_present_over_https(https_client: TestClient) -> None:
    """Strict-Transport-Security is sent with the configured max-age over HTTPS (Req 11.4)."""
    response = https_client.get("/test")

    assert response.status_code == 200
    assert "Strict-Transport-Security" in response.headers
    hsts = response.headers["Strict-Transport-Security"]
    assert "max-age=31536000" in hsts
    assert "includeSubDomains" in hsts


def test_hsts_max_age_reads_from_settings() -> None:
    """The max-age value comes from `Settings.hsts_max_age`, not a hard-coded literal."""
    app = FastAPI()
    app.add_middleware(  # type: ignore[arg-type]
        SecurityHeadersMiddleware,
        settings=build_test_settings(hsts_max_age=600, hsts_include_subdomains=False),
    )

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app, base_url="https://testserver")
    response = client.get("/test")

    hsts = response.headers["Strict-Transport-Security"]
    assert hsts == "max-age=600"


def test_csp_header_included(client: TestClient) -> None:
    """Test that Content-Security-Policy header is included."""
    response = client.get("/test")

    assert response.status_code == 200
    assert "Content-Security-Policy" in response.headers
    # Should have at least default-src directive
    csp = response.headers["Content-Security-Policy"]
    assert "default-src" in csp


def test_csp_strict_by_default_no_unsafe_inline_or_cdn(client: TestClient) -> None:
    """Non-documentation paths get the strict CSP: no inline scripts, no CDN hosts (Req 11.6)."""
    response = client.get("/test")

    csp = response.headers["Content-Security-Policy"]
    assert "'unsafe-inline'" not in csp
    assert "cdn.jsdelivr.net" not in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'self'" in csp
    assert "frame-ancestors 'none'" in csp


def test_csp_has_no_trailing_or_duplicated_whitespace(client: TestClient) -> None:
    """CSP directives have no trailing/duplicated whitespace and are terminated (Req 11.5)."""
    csp = client.get("/test").headers["Content-Security-Policy"]

    assert csp == csp.strip()
    assert "  " not in csp
    assert csp.endswith(";")


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
        settings=build_test_settings(),
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


def test_custom_headers_override_computed_hsts_and_csp() -> None:
    """Custom headers take precedence over the *computed* HSTS/CSP too, not just the static headers.

    `SecurityHeadersMiddleware.__init__`'s docstring documents that
    `custom_headers` overrides "any default or computed header, including HSTS
    and CSP" (dispatch() applies it last); this was previously only exercised
    for the always-static `X-Frame-Options`/`X-Custom-Header` pair.
    """
    app = FastAPI()
    custom_headers = {
        "Strict-Transport-Security": "max-age=0",
        "Content-Security-Policy": "default-src 'none';",
    }
    app.add_middleware(  # type: ignore[arg-type]
        SecurityHeadersMiddleware,
        settings=build_test_settings(),
        custom_headers=custom_headers,
    )

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app, base_url="https://testserver")
    response = client.get("/test")

    assert response.status_code == 200
    assert response.headers["Strict-Transport-Security"] == "max-age=0"
    assert response.headers["Content-Security-Policy"] == "default-src 'none';"


def test_x_xss_protection_not_included() -> None:
    """Test that X-XSS-Protection header is NOT included.

    X-XSS-Protection header was removed from modern browsers
    (Chrome 2019) and can cause XSS vulnerabilities in older IE versions.
    Content-Security-Policy supersedes it and provides better protection.
    """
    app = FastAPI()
    app.add_middleware(  # type: ignore[arg-type]
        SecurityHeadersMiddleware,
        settings=build_test_settings(),
    )

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
