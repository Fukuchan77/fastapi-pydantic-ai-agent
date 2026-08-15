"""Pytest fixtures for local Ollama tests.

This module provides fixtures that gate local tests on Ollama availability.

Gating happens at two levels, because a reachable Ollama server does not imply
the model a test needs has been pulled:

1. `require_ollama` (session, autouse) skips everything when the server is
   unreachable, and returns the set of model names that *are* pulled.
2. `skip_unless_model_pulled` (`tests/support/ollama.py`) lets each test
   module skip itself when its specific model is missing, instead of failing
   with a `ModelHTTPError` (status 500, "model ... not found") from deep
   inside LiteLLM.

Without level 2 a developer who has Ollama running but hasn't pulled a given
model sees a hard failure that looks like a code defect. The `ollama` marker
plus the `EXPECT_LIVE_TESTS` plugin keep these skips from going unnoticed.
"""

import pytest

from tests.support.ollama import list_pulled_models


@pytest.fixture(scope="session", autouse=True)
def require_ollama() -> frozenset[str]:
    """Skip all local tests unless Ollama is reachable, and list pulled models.

    This fixture runs once per test session and automatically applies to all tests
    in the tests/local/ directory. If Ollama is not available, all tests are skipped.

    Tests that need a specific model should request this fixture and pass its
    value to `tests.support.ollama.skip_unless_model_pulled`.

    Returns:
        Names of the models currently pulled, as reported by /api/tags
        (e.g. `{"llama3.2:latest", "granite4.1:8b"}`).

    Raises:
        pytest.skip: If Ollama is not running or not accessible.
    """
    return list_pulled_models()
