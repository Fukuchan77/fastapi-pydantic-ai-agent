"""Live test for the readiness probe's direct model request (Req 6.6).

`app/api/health.py::_probe_llm_provider` is the only place in this codebase
that calls `Model.request()` directly rather than through `Agent.run()` -
every other LLM call goes through the agent abstraction, which resolves and
merges `ModelRequestParameters` on the caller's behalf. v2 expanded that
resolution, so this is the only call site where the change is observable at
all, and only when driven against a real provider: the hermetic unit tests
in `tests/unit/api/test_health_ready.py` exercise `_probe_llm_provider`
against `FunctionModel`/`TestModel`, which never resolves real request
parameters.

This module calls `_probe_llm_provider` itself (not a reimplementation of
its logic) against a `Model` built from real Ollama settings, so a
regression in v2's parameter resolution surfaces here rather than only in
production.
"""

import pytest
from pydantic import SecretStr

from app.agents.chat_agent import build_model
from app.api.health import _probe_llm_provider
from app.config import Settings
from tests.support.ollama import skip_unless_model_pulled


READINESS_PROBE_MODEL = "granite4.1:8b"
"""Bare Ollama model name required by this module."""


@pytest.fixture
def ollama_settings_readiness_probe(
    monkeypatch: pytest.MonkeyPatch,
    require_ollama: frozenset[str],
) -> Settings:
    """Settings for the readiness probe's direct model request.

    LLM_BASE_URL is optional - LiteLLM defaults to http://localhost:11434.
    LLM_API_KEY is not required for Ollama (local provider).

    Args:
        monkeypatch: Pytest monkeypatch fixture used to drop LLM_API_KEY.
        require_ollama: Pulled Ollama model names; the test is skipped when
            granite4.1:8b is absent.

    Returns:
        Settings pointing at the local granite4.1:8b model.
    """
    skip_unless_model_pulled(require_ollama, READINESS_PROBE_MODEL)

    # Remove LLM_API_KEY from environment to test Ollama without API key
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    return Settings(
        api_key=SecretStr("local-dev-api-key-12345"),
        llm_model=f"ollama:{READINESS_PROBE_MODEL}",
    )


@pytest.mark.ollama
@pytest.mark.asyncio
async def test_readiness_probe_llm_provider_healthy_against_real_ollama(
    ollama_settings_readiness_probe: Settings,
) -> None:
    """The readiness probe's direct `model.request()` call succeeds against real Ollama.

    Drives the exact code path `/health/ready` runs in production - a
    single-token completion request issued straight at the `Model`, bypassing
    `Agent.run()` - so v2's request-parameter resolution is verified
    end-to-end rather than only through a mocked model.
    """
    model = build_model(ollama_settings_readiness_probe)

    result = await _probe_llm_provider(model)

    assert result == "healthy", "Readiness probe should report a reachable Ollama model as healthy"
