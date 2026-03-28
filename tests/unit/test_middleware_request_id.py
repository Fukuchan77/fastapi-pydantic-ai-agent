"""Unit tests for Request ID middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.request_id import RequestIDMiddleware


@pytest.fixture
def app_with_request_id() -> FastAPI:
    """Create a FastAPI app with Request ID middleware for testing."""
    app = FastAPI()

    # Add Request ID middleware
    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app_with_request_id: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app_with_request_id)


def test_request_id_generated_when_not_provided(client: TestClient) -> None:
    """Test that a request ID is automatically generated when not provided."""
    response = client.get("/test")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    # Should be a valid UUID format (36 characters with hyphens)
    assert len(response.headers["X-Request-ID"]) == 36


def test_request_id_accepted_when_valid(client: TestClient) -> None:
    """Test that a valid request ID from client is accepted and returned."""
    valid_request_id = "abc123-def456-ghi789"
    response = client.get(
        "/test",
        headers={"X-Request-ID": valid_request_id},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == valid_request_id


def test_request_id_rejects_crlf_injection(client: TestClient) -> None:
    """Test that request ID with CRLF characters is rejected (security)."""
    # Attempt CRLF injection to add malicious headers
    malicious_request_id = "abc123\r\nEvil-Header: malicious-content"
    response = client.get(
        "/test",
        headers={"X-Request-ID": malicious_request_id},
    )

    assert response.status_code == 200
    # Should generate new UUID instead of using malicious input
    assert "Evil-Header" not in response.headers
    # Response should have a clean request ID (not the malicious one)
    assert "\r\n" not in response.headers["X-Request-ID"]
    assert response.headers["X-Request-ID"] != malicious_request_id


def test_request_id_rejects_newline_injection(client: TestClient) -> None:
    """Test that request ID with newline characters is rejected (security)."""
    malicious_request_id = "abc123\nMalicious-Header: evil"
    response = client.get(
        "/test",
        headers={"X-Request-ID": malicious_request_id},
    )

    assert response.status_code == 200
    # Should generate new UUID instead of using malicious input
    assert "Malicious-Header" not in response.headers
    assert "\n" not in response.headers["X-Request-ID"]
    assert response.headers["X-Request-ID"] != malicious_request_id


def test_request_id_rejects_invalid_characters(client: TestClient) -> None:
    """Test that request ID with invalid characters is rejected."""
    # Characters outside the allowed set: a-zA-Z0-9_-
    invalid_request_ids = [
        "abc@123",  # @ character
        "abc#123",  # # character
        "abc$123",  # $ character
        "abc%123",  # % character
        "abc&123",  # & character
        "abc=123",  # = character
        "abc+123",  # + character
        "abc/123",  # / character
        "abc\\123",  # \ character
        "abc:123",  # : character
        "abc;123",  # ; character
        "abc<123",  # < character
        "abc>123",  # > character
    ]

    for invalid_id in invalid_request_ids:
        response = client.get(
            "/test",
            headers={"X-Request-ID": invalid_id},
        )

        assert response.status_code == 200
        # Should generate new UUID instead of using invalid input
        assert response.headers["X-Request-ID"] != invalid_id
        # Response should be a valid UUID (36 chars with hyphens)
        assert len(response.headers["X-Request-ID"]) == 36


def test_request_id_rejects_oversized_id(client: TestClient) -> None:
    """Test that request ID longer than 64 characters is rejected."""
    # Create a 65-character request ID (exceeds limit)
    oversized_id = "a" * 65
    response = client.get(
        "/test",
        headers={"X-Request-ID": oversized_id},
    )

    assert response.status_code == 200
    # Should generate new UUID instead of using oversized input
    assert response.headers["X-Request-ID"] != oversized_id
    assert len(response.headers["X-Request-ID"]) == 36


def test_request_id_rejects_empty_string(client: TestClient) -> None:
    """Test that empty request ID is rejected."""
    response = client.get(
        "/test",
        headers={"X-Request-ID": ""},
    )

    assert response.status_code == 200
    # Should generate new UUID instead of using empty string
    assert response.headers["X-Request-ID"] != ""
    assert len(response.headers["X-Request-ID"]) == 36


def test_request_id_accepts_valid_uuid_format(client: TestClient) -> None:
    """Test that valid UUID format is accepted."""
    valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
    response = client.get(
        "/test",
        headers={"X-Request-ID": valid_uuid},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == valid_uuid


def test_request_id_accepts_alphanumeric_with_hyphens_underscores(client: TestClient) -> None:
    """Test that alphanumeric IDs with hyphens and underscores are accepted."""
    valid_ids = [
        "abc123",
        "ABC123",
        "abc-123",
        "abc_123",
        "abc-def_123",
        "a1b2c3",
        "request-id-12345",
        "request_id_12345",
        "a" * 64,  # Exactly 64 characters (should be accepted)
    ]

    for valid_id in valid_ids:
        response = client.get(
            "/test",
            headers={"X-Request-ID": valid_id},
        )

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == valid_id
