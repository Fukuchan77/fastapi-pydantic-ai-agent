"""Shared helpers for gating tests on local Ollama availability and model presence.

Lives alongside `tests/support/hermetic.py` rather than in `tests/local/conftest.py`
so it can be imported from outside `tests/local/` (e.g. a hermetic unit test that
exercises the gating logic itself) without reaching across package boundaries into
another directory's conftest.
"""

import httpx
import pytest


OLLAMA_BASE_URL = "http://localhost:11434"
"""Default Ollama API base URL (LiteLLM's default for the ollama provider)."""


def list_pulled_models() -> frozenset[str]:
    """Fetch the set of model names currently pulled into the local Ollama server.

    Returns:
        Names of the models currently pulled, as reported by /api/tags
        (e.g. `{"llama3.2:latest", "granite4.1:8b"}`).

    Raises:
        pytest.skip: If Ollama is not running or not accessible.
    """
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        pytest.skip(
            f"Ollama is not running or not accessible at {OLLAMA_BASE_URL}. "
            f"Error: {e}. "
            "Please start Ollama before running local tests: 'ollama serve'"
        )

    payload = response.json()
    return frozenset(model["name"] for model in payload.get("models", []))


def skip_unless_model_pulled(pulled_models: frozenset[str], model: str) -> None:
    """Skip the current test unless `model` has been pulled into Ollama.

    Without this check, a reachable-but-missing model surfaces as a hard
    `ModelHTTPError` (status 500, "model ... not found") from deep inside
    LiteLLM, which looks like a code defect rather than a local setup gap.

    Args:
        pulled_models: Model names from `list_pulled_models()`.
        model: Bare Ollama model name to require, without the `ollama:` provider
            prefix (e.g. `"granite4.1:8b"`).

    Raises:
        pytest.skip: If `model` is not among `pulled_models`.
    """
    if model in pulled_models:
        return

    available = ", ".join(sorted(pulled_models)) or "(none)"
    pytest.skip(
        f"Ollama model '{model}' is not pulled at {OLLAMA_BASE_URL}. "
        f"Run 'ollama pull {model}' to enable this test. "
        f"Currently pulled: {available}"
    )
