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

OLLAMA_LIVE_TEST_COUNT = 6
"""Number of `ollama`-marked test functions gated behind Ollama reachability (Req 6.7).

Modelled on `tests/support/redis.py::REDIS_LIVE_TEST_COUNT`. Unlike the Redis
and Chroma lanes, whose live cases each live in one gated module, the
`ollama` marker is applied per test function across every module in
`tests/local/`, so this count is the sum across all of them:
`test_llm_granite41.py` (3), `test_llm_llama32.py` (2), and
`test_readiness_probe_live.py` (1).

Pass `EXPECT_LIVE_TESTS=$OLLAMA_LIVE_TEST_COUNT` alongside a reachable
server (e.g. `EXPECT_LIVE_TESTS=6 mise run test:local`) so a lane that
silently collects zero live cases fails instead of reporting success (Req
13.8). `tests/unit/test_local_test_gating.py` guards this value against drift
in four places: `test_ollama_live_test_count_matches_gated_modules` as tests
are added to or removed from `tests/local/`, and one test each for the
pre-push hook's `EXPECT_LIVE_TESTS` literal and its prose restatements in
`CLAUDE.md`/`AGENTS.md`. `docs/adapter-probe-report*.md` also states a count
and is deliberately excluded - Req 6.8 forbids editing a gate-evidence
artifact when this count rises.
"""


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
