"""Unit tests for the readiness health check endpoint (Req 13.1, 13.2).

`/health/ready` performs live connectivity probes against every store or
provider currently selected by configuration: the Redis session store (only
when enabled), the active vector store backend, and the LLM provider.
In-memory backends have no external dependency to probe and are reported as
"skipped" rather than "healthy" (mirrors the startup dry-run's own
skip-in-memory convention, Task 3). Any "unreachable" probe returns 503.

Redis/Ollama connectivity is mocked at their client boundaries (matching
`tests/unit/stores/test_factory.py`'s conventions) so these tests remain
hermetic - no real Redis/Ollama/LLM network required. The LLM provider probe
is exercised via `FunctionModel`, which raises/succeeds synchronously with no
real network call.
"""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from httpx import Response
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app.stores.session_store import InMemorySessionStore
from app.stores.session_store import RedisSessionStore
from app.stores.vector_store import InMemoryVectorStore
from app.stores.vector_store import OllamaEmbeddingVectorStore


def _mock_request(session_store: object, vector_store: object, model: object) -> MagicMock:
    """Build a mock `Request` exposing the app.state attributes the endpoint reads."""
    mock_request = MagicMock()
    mock_request.app.state.session_store = session_store
    mock_request.app.state.vector_store = vector_store
    mock_request.app.state.chat_agent.model = model
    return mock_request


def _success_llm_function(messages: list, agent_info: AgentInfo) -> ModelResponse:
    """A `FunctionModel` function that always succeeds, matching a healthy provider."""
    return ModelResponse(parts=[TextPart(content="pong")])


def _failing_llm_function(messages: list, agent_info: AgentInfo) -> ModelResponse:
    """A `FunctionModel` function that always raises, simulating an unreachable provider."""
    raise ConnectionError("Connection refused")


# ---------------------------------------------------------------------------
# In-memory backends are skipped (nothing external to probe)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_check_import() -> None:
    """readiness_check can be imported."""
    from app.api.health import readiness_check

    assert readiness_check is not None


@pytest.mark.asyncio
async def test_readiness_check_skips_in_memory_backends_and_is_ready() -> None:
    """Default in-memory session/vector stores are 'skipped', not probed, and pass."""
    from app.api.health import readiness_check

    mock_request = _mock_request(
        InMemorySessionStore(), InMemoryVectorStore(), FunctionModel(_success_llm_function)
    )

    response = await readiness_check(mock_request)
    body = json.loads(response.body)

    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["checks"]["session_store"] == "skipped"
    assert body["checks"]["vector_store"] == "skipped"
    assert body["checks"]["llm_provider"] == "healthy"


@pytest.mark.asyncio
async def test_readiness_check_response_structure() -> None:
    """Response is a JSON object with 'status' and 'checks' keys only."""
    from app.api.health import readiness_check

    mock_request = _mock_request(
        InMemorySessionStore(), InMemoryVectorStore(), FunctionModel(_success_llm_function)
    )

    response = await readiness_check(mock_request)
    body = json.loads(response.body)

    assert set(body.keys()) == {"status", "checks"}
    assert set(body["checks"].keys()) == {"session_store", "vector_store", "llm_provider"}


# ---------------------------------------------------------------------------
# Redis session store probe (Req 13.1: "Redis only when the Redis session
# store is enabled")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_check_redis_session_store_healthy() -> None:
    """A reachable Redis session store reports 'healthy' and the endpoint is ready."""
    from app.api.health import readiness_check

    mock_redis_client = AsyncMock()
    mock_redis_client.get = AsyncMock(return_value=None)
    with patch("redis.asyncio.from_url", return_value=mock_redis_client):
        session_store = RedisSessionStore(redis_url="redis://localhost:6379/0")

    mock_request = _mock_request(
        session_store, InMemoryVectorStore(), FunctionModel(_success_llm_function)
    )

    response = await readiness_check(mock_request)
    body = json.loads(response.body)

    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["checks"]["session_store"] == "healthy"


@pytest.mark.asyncio
async def test_readiness_check_redis_session_store_unreachable_returns_503() -> None:
    """An unreachable Redis session store reports 'unreachable' and 503 not_ready."""
    from app.api.health import readiness_check

    mock_redis_client = AsyncMock()
    mock_redis_client.get = AsyncMock(side_effect=ConnectionError("Connection refused"))
    with patch("redis.asyncio.from_url", return_value=mock_redis_client):
        session_store = RedisSessionStore(redis_url="redis://localhost:6379/0")

    mock_request = _mock_request(
        session_store, InMemoryVectorStore(), FunctionModel(_success_llm_function)
    )

    response = await readiness_check(mock_request)
    body = json.loads(response.body)

    assert response.status_code == 503
    assert body["status"] == "not_ready"
    assert body["checks"]["session_store"] == "unreachable"


# ---------------------------------------------------------------------------
# Vector store probe (Req 13.1: "the active vector store backend")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_check_ollama_vector_store_healthy() -> None:
    """A reachable Ollama-backed vector store reports 'healthy'."""
    from app.api.health import readiness_check

    mock_client = AsyncMock()
    mock_response = Mock(spec=Response)
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})
    mock_client.post = AsyncMock(return_value=mock_response)
    vector_store = OllamaEmbeddingVectorStore(
        embedding_model="nomic-embed-text:latest", http_client=mock_client
    )

    mock_request = _mock_request(
        InMemorySessionStore(), vector_store, FunctionModel(_success_llm_function)
    )

    response = await readiness_check(mock_request)
    body = json.loads(response.body)

    assert response.status_code == 200
    assert body["checks"]["vector_store"] == "healthy"


@pytest.mark.asyncio
async def test_readiness_check_ollama_vector_store_unreachable_returns_503() -> None:
    """An unreachable Ollama-backed vector store reports 'unreachable' and 503 not_ready."""
    from app.api.health import readiness_check

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=ConnectionError("Connection refused"))
    vector_store = OllamaEmbeddingVectorStore(
        embedding_model="nomic-embed-text:latest", http_client=mock_client
    )

    mock_request = _mock_request(
        InMemorySessionStore(), vector_store, FunctionModel(_success_llm_function)
    )

    response = await readiness_check(mock_request)
    body = json.loads(response.body)

    assert response.status_code == 503
    assert body["status"] == "not_ready"
    assert body["checks"]["vector_store"] == "unreachable"


# ---------------------------------------------------------------------------
# LLM provider probe (Req 13.1: "the LLM provider")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_check_llm_provider_healthy() -> None:
    """A reachable LLM provider reports 'healthy'."""
    from app.api.health import readiness_check

    mock_request = _mock_request(
        InMemorySessionStore(), InMemoryVectorStore(), TestModel()
    )

    response = await readiness_check(mock_request)
    body = json.loads(response.body)

    assert response.status_code == 200
    assert body["checks"]["llm_provider"] == "healthy"


@pytest.mark.asyncio
async def test_readiness_check_llm_provider_unreachable_returns_503() -> None:
    """An unreachable LLM provider reports 'unreachable' and 503 not_ready."""
    from app.api.health import readiness_check

    mock_request = _mock_request(
        InMemorySessionStore(), InMemoryVectorStore(), FunctionModel(_failing_llm_function)
    )

    response = await readiness_check(mock_request)
    body = json.loads(response.body)

    assert response.status_code == 503
    assert body["status"] == "not_ready"
    assert body["checks"]["llm_provider"] == "unreachable"


# ---------------------------------------------------------------------------
# Multiple simultaneous failures (Req 13.2: identify the failed dependency)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_check_identifies_every_failed_dependency() -> None:
    """Every failing probe is individually named 'unreachable' in the checks dict."""
    from app.api.health import readiness_check

    mock_redis_client = AsyncMock()
    mock_redis_client.get = AsyncMock(side_effect=ConnectionError("Connection refused"))
    with patch("redis.asyncio.from_url", return_value=mock_redis_client):
        session_store = RedisSessionStore(redis_url="redis://localhost:6379/0")

    mock_request = _mock_request(
        session_store, InMemoryVectorStore(), FunctionModel(_failing_llm_function)
    )

    response = await readiness_check(mock_request)
    body = json.loads(response.body)

    assert response.status_code == 503
    assert body["checks"]["session_store"] == "unreachable"
    assert body["checks"]["llm_provider"] == "unreachable"
    assert body["checks"]["vector_store"] == "skipped"
