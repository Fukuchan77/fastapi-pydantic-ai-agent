"""Unit tests for async health_check endpoint - Task 16.16."""

import inspect


def test_health_check_is_async_function() -> None:
    """Test that health_check is defined as async def (Task 16.16).

    FastAPI best practice: All route handlers should be async def to avoid
    blocking the event loop, even if they don't perform I/O operations.
    This ensures consistent async patterns across the codebase.
    """
    from app.api.health import health_check

    # Verify the function is a coroutine function (async def)
    assert inspect.iscoroutinefunction(health_check), (
        "health_check() must be defined as 'async def' not 'def' (Task 16.16 requirement)"
    )
