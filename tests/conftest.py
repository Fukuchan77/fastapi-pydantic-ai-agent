"""Shared pytest fixtures for all tests."""

import pytest


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear get_settings cache after each test to prevent pollution.

    The get_settings() function uses @lru_cache(maxsize=1), so settings are
    cached globally. Tests use monkeypatch to set different environment variables,
    but without clearing the cache, one test's settings could leak into another.

    This fixture runs automatically after every test (autouse=True) to ensure
    test isolation.
    """
    # Yield first to let the test run
    yield

    # Clear the cache after the test completes
    from app.config import get_settings

    get_settings.cache_clear()
