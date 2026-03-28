"""Tests for consistent use of ErrorResponse.code field.

Task 16.9: Verify that all ErrorResponse instances include appropriate
error codes for programmatic error handling.
"""

from fastapi.testclient import TestClient


def test_auth_error_has_code() -> None:
    """Test that 401 Unauthorized errors include error code.

    Task 16.9: ErrorResponse for authentication failures should include
    a "code" field with value "UNAUTHORIZED" for programmatic handling.
    """
    from app.main import app

    # Use context manager to properly trigger lifespan
    with TestClient(app) as client:
        # Make request to protected endpoint without API key header
        response = client.post(
            "/v1/agent/chat",
            json={"message": "Hello"},
        )

        assert response.status_code == 401

        # FastAPI wraps the ErrorResponse in a 'detail' key for HTTPException
        response_json = response.json()
        assert "detail" in response_json
        error_data = response_json["detail"]

        # Verify error response structure includes code field
        assert "message" in error_data
        assert "code" in error_data

        # Task 16.9: Verify error code is set to UNAUTHORIZED
        assert error_data["code"] == "UNAUTHORIZED", (
            f"Expected code='UNAUTHORIZED', got code={error_data['code']!r}"
        )
        assert error_data["message"] == "Unauthorized"


def test_internal_error_has_code() -> None:
    """Test that 500 Internal Server errors include error code.

    Task 16.9: ErrorResponse for unhandled exceptions should include
    a "code" field with value "INTERNAL_ERROR" for programmatic handling.
    """
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)

    # Create a test endpoint that raises an exception
    @app.get("/test-error-code-check")
    def test_error_endpoint() -> None:
        raise RuntimeError("Simulated internal error")

    # Make request that will trigger unhandled exception
    response = client.get("/test-error-code-check")

    assert response.status_code == 500

    # Verify error response structure includes code field
    error_data = response.json()
    assert "message" in error_data
    assert "code" in error_data

    # Task 16.9: Verify error code is set to INTERNAL_ERROR
    assert error_data["code"] == "INTERNAL_ERROR", (
        f"Expected code='INTERNAL_ERROR', got code={error_data['code']!r}"
    )
    assert error_data["message"] == "Internal server error occurred"
