"""Unit tests for health endpoint."""

import pytest


def test_health_router_can_be_imported() -> None:
    """Test that health router can be imported successfully.

    Task 16.3: Fixed misleading test name from 'test_health_endpoint_import_fails'.
    The test verifies successful import, not failure.
    """
    from app.api.health import router

    assert router is not None


@pytest.mark.asyncio
async def test_health_endpoint_returns_ok_status() -> None:
    """Test that health endpoint returns correct status.

    Task 16.16: Updated to await async health_check().
    """
    from app.api.health import health_check

    # Call the route handler directly (unit test - no HTTP stack)
    result = await health_check()

    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_endpoint_response_structure() -> None:
    """Test that health endpoint returns dict with status key.

    Task 16.16: Updated to await async health_check().
    """
    from app.api.health import health_check

    result = await health_check()

    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] == "ok"
