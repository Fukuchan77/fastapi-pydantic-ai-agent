"""Tests for middleware order issue.

Task: FIX 優先対応事項 - Issue 2
Problem: Middleware order is reversed in main.py:155-159, causing RequestSizeLimitMiddleware
to execute before RequestIDMiddleware. This means 413 responses don't have X-Request-ID headers.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_request_id_in_413_response() -> None:
    """Test that 413 Payload Too Large responses include X-Request-ID header.

    RED PHASE: This test should fail because RequestSizeLimitMiddleware is added
    after RequestIDMiddleware, which means it executes BEFORE RequestIDMiddleware
    (FastAPI middleware executes in reverse order of addition). Therefore, 413 responses
    won't have the X-Request-ID header.
    """
    client = TestClient(app)

    # Create a payload that exceeds 10MB limit
    large_payload = {"message": "x" * (11 * 1024 * 1024), "session_id": "test"}

    # Send request that should trigger 413
    response = client.post(
        "/v1/agent/chat",
        json=large_payload,
        headers={"X-API-Key": "test-api-key"},
    )

    # Should return 413
    assert response.status_code == 413

    # Should have X-Request-ID header - this will fail in RED phase
    assert "X-Request-ID" in response.headers, (
        "413 response missing X-Request-ID header - middleware order is incorrect"
    )
    assert response.headers["X-Request-ID"], "X-Request-ID header should not be empty"


def test_request_id_in_normal_response() -> None:
    """Test that normal responses have X-Request-ID header (baseline check)."""
    client = TestClient(app)

    # Send normal request
    response = client.get("/health")

    # Should return 200
    assert response.status_code == 200

    # Should have X-Request-ID header
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]


def test_request_id_propagates_to_error_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that X-Request-ID is present in various error responses.

    This ensures middleware order is correct for all response types.
    """
    # Set required environment variables for Settings
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")

    # Use context manager to run lifespan and initialize app.state
    with TestClient(app) as client:
        # Test 404 response
        response = client.get(
            "/nonexistent",
        )
        assert response.status_code == 404
        assert "X-Request-ID" in response.headers

        # Test 401 response (missing API key)
        response = client.post(
            "/v1/agent/chat",
            json={"message": "test"},
        )
        assert response.status_code == 401
        assert "X-Request-ID" in response.headers
