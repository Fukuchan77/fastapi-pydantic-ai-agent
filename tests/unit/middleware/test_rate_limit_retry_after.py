"""Unit tests for Retry-After header in rate limit responses.

Task 20.11: Verify that rate limit errors include Retry-After header
to improve client UX and reduce retry storms.
"""

import pytest
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware.rate_limit import add_rate_limiting


@pytest.fixture
def app_with_rate_limiting() -> FastAPI:
    """Create FastAPI app with rate limiting configured."""
    app = FastAPI()

    # Add very restrictive rate limiting for testing
    limiter = add_rate_limiting(app, default_limits=["2/minute"])

    @app.get("/test")
    @limiter.limit("2/minute")
    async def test_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    return app


@pytest.fixture
def client(app_with_rate_limiting: FastAPI) -> TestClient:
    """Create test client with rate limiting enabled."""
    return TestClient(app_with_rate_limiting)


def test_rate_limit_response_includes_retry_after_header(client: TestClient):
    """Test that 429 response includes Retry-After header.

    Task 20.11: When a client exceeds the rate limit, the response should
    include a Retry-After header indicating how many seconds to wait before
    retrying. This is a standard HTTP header (RFC 6585, RFC 7231) that
    improves client UX and reduces retry storms.
    """
    # Make requests until rate limit is hit (2/minute limit)
    client.get("/test")  # Request 1
    client.get("/test")  # Request 2
    response = client.get("/test")  # Request 3 - should be rate limited

    # Verify we got rate limited
    assert response.status_code == 429

    # Verify Retry-After header is present
    assert "Retry-After" in response.headers, (
        "Rate limit response should include Retry-After header "
        "to tell clients how long to wait before retrying"
    )

    # Verify Retry-After value is a positive integer (seconds)
    retry_after = response.headers["Retry-After"]
    assert retry_after.isdigit(), "Retry-After should be an integer (seconds)"
    assert int(retry_after) > 0, "Retry-After should be positive"
    assert int(retry_after) <= 60, "Retry-After should be reasonable (≤60s for 1-minute window)"


def test_retry_after_header_value_is_reasonable(client: TestClient):
    """Test that Retry-After value corresponds to the rate limit window.

    Task 20.11: For a 1-minute rate limit window, the Retry-After value
    should be between 1 and 60 seconds (the remaining time in the window).
    """
    # Hit rate limit
    client.get("/test")
    client.get("/test")
    response = client.get("/test")

    assert response.status_code == 429
    assert "Retry-After" in response.headers

    retry_after_seconds = int(response.headers["Retry-After"])

    # Should be within the 1-minute window
    assert 1 <= retry_after_seconds <= 60, (
        f"Retry-After should be 1-60 seconds for a 1-minute window, got {retry_after_seconds}"
    )


def test_retry_after_header_format_is_integer_not_http_date(client: TestClient):
    """Test that Retry-After uses delay-seconds format, not HTTP-date format.

    Task 20.11: The Retry-After header can be either an HTTP-date or an
    integer representing seconds. For rate limiting, the integer format
    is more appropriate and easier for clients to parse.
    """
    # Hit rate limit
    client.get("/test")
    client.get("/test")
    response = client.get("/test")

    assert response.status_code == 429
    assert "Retry-After" in response.headers

    retry_after = response.headers["Retry-After"]

    # Should be a simple integer, not an HTTP-date string
    assert retry_after.isdigit(), (
        "Retry-After should use delay-seconds format (integer), "
        f"not HTTP-date format. Got: {retry_after}"
    )
