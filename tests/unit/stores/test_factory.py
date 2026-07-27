"""Unit tests for the store factory (Task 3: store factory wiring + dry-run).

Covers the settings -> implementation selection matrix (Req 5.1, 5.2, 5.3) and
the startup connectivity dry-run (Req 5.4, 5.5). Redis and Ollama connectivity
are mocked at their respective client boundaries (matching the existing
`test_redis_session_store.py` / `test_ollama_embedding_response_validation.py`
conventions) so these tests remain hermetic - no real Redis/Ollama required.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from httpx import Response

from app.config import Settings
from app.stores.factory import StoreDryRunError
from app.stores.factory import build_session_store
from app.stores.factory import build_vector_store
from app.stores.factory import dry_run_stores
from app.stores.session_store import InMemorySessionStore
from app.stores.session_store import RedisSessionStore
from app.stores.vector_store import InMemoryVectorStore
from app.stores.vector_store import OllamaEmbeddingVectorStore


def _build_settings(**overrides: object) -> Settings:
    """Build a valid Settings instance directly, without touching os.environ.

    `embedding_model`/`embedding_base_url` default to `None` explicitly (not
    just omitted) so a developer's local `.env` values for those fields can
    never leak into these tests via pydantic-settings' env-file fallback.
    """
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "openai:gpt-4o",
        "llm_api_key": "test-llm-api-key-12345",
        "embedding_model": None,
        "embedding_base_url": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_session_store selection matrix (Req 5.1, 5.2)
# ---------------------------------------------------------------------------


def test_build_session_store_default_returns_in_memory() -> None:
    """Default settings (redis disabled) select InMemorySessionStore."""
    store = build_session_store(_build_settings())

    assert isinstance(store, InMemorySessionStore)


def test_build_session_store_redis_enabled_returns_redis_session_store() -> None:
    """redis_session_store_enabled=True selects RedisSessionStore with the configured URL."""
    mock_redis_client = AsyncMock()
    settings = _build_settings(
        redis_session_store_enabled=True,
        redis_url="redis://localhost:6379/0",
    )

    with patch("redis.asyncio.from_url", return_value=mock_redis_client) as mock_from_url:
        store = build_session_store(settings)

    assert isinstance(store, RedisSessionStore)
    mock_from_url.assert_called_once_with("redis://localhost:6379/0", decode_responses=False)


# ---------------------------------------------------------------------------
# build_vector_store selection matrix (Req 5.1, 5.3)
# ---------------------------------------------------------------------------


def test_build_vector_store_default_returns_in_memory() -> None:
    """Default settings (vector_store_backend='memory') select InMemoryVectorStore."""
    store = build_vector_store(_build_settings())

    assert isinstance(store, InMemoryVectorStore)


def test_build_vector_store_chroma_backend_constructs_chroma_store() -> None:
    """vector_store_backend='chroma' constructs ChromaVectorStore."""
    settings = _build_settings(vector_store_backend="chroma")
    mock_chroma_cls = MagicMock()

    with patch("app.stores.factory.ChromaVectorStore", mock_chroma_cls):
        build_vector_store(settings)

    mock_chroma_cls.assert_called_once_with()


def test_build_vector_store_chroma_backend_passes_embedding_model_when_set() -> None:
    """vector_store_backend='chroma' forwards an explicit embedding_model."""
    settings = _build_settings(
        vector_store_backend="chroma",
        embedding_model="all-MiniLM-L6-v2",
    )
    mock_chroma_cls = MagicMock()

    with patch("app.stores.factory.ChromaVectorStore", mock_chroma_cls):
        build_vector_store(settings)

    mock_chroma_cls.assert_called_once_with(embedding_model="all-MiniLM-L6-v2")


def test_build_vector_store_ollama_backend_constructs_ollama_store() -> None:
    """vector_store_backend='ollama' constructs OllamaEmbeddingVectorStore."""
    settings = _build_settings(
        vector_store_backend="ollama",
        embedding_model="nomic-embed-text:latest",
    )
    mock_ollama_cls = MagicMock()

    with patch("app.stores.factory.OllamaEmbeddingVectorStore", mock_ollama_cls):
        build_vector_store(settings)

    mock_ollama_cls.assert_called_once_with(embedding_model="nomic-embed-text:latest")


def test_build_vector_store_ollama_backend_passes_base_url_when_set() -> None:
    """vector_store_backend='ollama' forwards an explicit embedding_base_url."""
    settings = _build_settings(
        vector_store_backend="ollama",
        embedding_model="nomic-embed-text:latest",
        embedding_base_url="http://localhost:11434/v1",
    )
    mock_ollama_cls = MagicMock()

    with patch("app.stores.factory.OllamaEmbeddingVectorStore", mock_ollama_cls):
        build_vector_store(settings)

    mock_ollama_cls.assert_called_once_with(
        embedding_model="nomic-embed-text:latest",
        base_url="http://localhost:11434/v1",
    )


# ---------------------------------------------------------------------------
# dry_run_stores (Req 5.4, 5.5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_stores_skips_in_memory_stores() -> None:
    """In-memory stores have no external dependency and pass the dry-run trivially."""
    await dry_run_stores(InMemorySessionStore(), InMemoryVectorStore())


@pytest.mark.asyncio
async def test_dry_run_stores_probes_redis_session_store_success() -> None:
    """A reachable Redis session store passes the dry-run without raising."""
    mock_redis_client = AsyncMock()
    mock_redis_client.get = AsyncMock(return_value=None)
    with patch("redis.asyncio.from_url", return_value=mock_redis_client):
        store = RedisSessionStore(redis_url="redis://localhost:6379/0")

    await dry_run_stores(store)

    mock_redis_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_dry_run_stores_raises_store_dry_run_error_on_redis_failure() -> None:
    """An unreachable Redis session store fails the dry-run with StoreDryRunError."""
    mock_redis_client = AsyncMock()
    mock_redis_client.get = AsyncMock(side_effect=ConnectionError("Connection refused"))
    with patch("redis.asyncio.from_url", return_value=mock_redis_client):
        store = RedisSessionStore(redis_url="redis://localhost:6379/0")

    with pytest.raises(StoreDryRunError, match="Redis session store dry-run failed"):
        await dry_run_stores(store)


@pytest.mark.asyncio
async def test_dry_run_stores_probes_ollama_vector_store_success() -> None:
    """A reachable Ollama embedding store passes the dry-run without raising."""
    mock_client = AsyncMock()
    mock_response = Mock(spec=Response)
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})
    mock_client.post = AsyncMock(return_value=mock_response)
    store = OllamaEmbeddingVectorStore(
        embedding_model="nomic-embed-text:latest", http_client=mock_client
    )

    await dry_run_stores(store)

    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_dry_run_stores_raises_store_dry_run_error_on_ollama_failure() -> None:
    """An unreachable Ollama embedding store fails the dry-run with StoreDryRunError."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=ConnectionError("Connection refused"))
    store = OllamaEmbeddingVectorStore(
        embedding_model="nomic-embed-text:latest", http_client=mock_client
    )

    with pytest.raises(StoreDryRunError, match="Ollama vector store dry-run failed"):
        await dry_run_stores(store)


@pytest.mark.asyncio
async def test_dry_run_stores_probes_multiple_stores_in_one_call() -> None:
    """dry_run_stores probes every store passed to it, not just the first."""
    mock_redis_client = AsyncMock()
    mock_redis_client.get = AsyncMock(return_value=None)
    with patch("redis.asyncio.from_url", return_value=mock_redis_client):
        session_store = RedisSessionStore(redis_url="redis://localhost:6379/0")

    await dry_run_stores(session_store, InMemoryVectorStore())

    mock_redis_client.get.assert_called_once()
