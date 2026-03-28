"""Pytest fixtures for local Ollama tests.

This module provides fixtures that check for Ollama availability before running tests.
"""

import httpx
import pytest


@pytest.fixture(scope="session", autouse=True)
def require_ollama() -> None:
    """Check if Ollama is running and available.

    This fixture runs once per test session and automatically applies to all tests
    in the tests/local/ directory. If Ollama is not available, all tests are skipped.

    The fixture checks Ollama availability by making a GET request to the /api/tags
    endpoint, which lists available models. The default Ollama API base URL is
    http://localhost:11434.

    Raises:
        pytest.skip: If Ollama is not running or not accessible.
    """
    ollama_base_url = "http://localhost:11434"
    try:
        response = httpx.get(f"{ollama_base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        pytest.skip(
            f"Ollama is not running or not accessible at {ollama_base_url}. "
            f"Error: {e}. "
            "Please start Ollama before running local tests: 'ollama serve'"
        )
