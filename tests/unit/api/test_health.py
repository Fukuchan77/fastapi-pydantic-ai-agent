"""Unit tests for health endpoint."""


def test_health_endpoint_import_fails() -> None:
    """Test that health router can be imported (RED: should fail initially)."""
    from app.api.health import router

    assert router is not None


def test_health_endpoint_returns_ok_status() -> None:
    """Test that health endpoint returns correct status."""
    from app.api.health import health_check

    # Call the route handler directly (unit test - no HTTP stack)
    result = health_check()

    assert result == {"status": "ok"}


def test_health_endpoint_response_structure() -> None:
    """Test that health endpoint returns dict with status key."""
    from app.api.health import health_check

    result = health_check()

    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "ok"
