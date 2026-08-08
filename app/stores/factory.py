"""Store factory: select and construct session/vector store implementations from settings.

Centralizes the settings -> implementation mapping for `SessionStore` and
`VectorStore` (Constitution Principle 2: extend through Protocols) instead of
`app/main.py` hard-coding the in-memory implementations, and runs a startup
connectivity dry-run against whichever external stores were selected so a
misconfigured deployment fails at startup rather than on the first request.
"""

import logging

from app.config import Settings
from app.stores.session_store import InMemorySessionStore
from app.stores.session_store import RedisSessionStore
from app.stores.session_store import SessionStore
from app.stores.vector_store import ChromaVectorStore
from app.stores.vector_store import InMemoryVectorStore
from app.stores.vector_store import OllamaEmbeddingVectorStore
from app.stores.vector_store import VectorStore


logger = logging.getLogger(__name__)

# Reserved identifier used only to probe session-store connectivity; a plain
# read, never persisted. The vector-store probe (`OllamaEmbeddingVectorStore.ping()`)
# needs no reserved identifier since it never touches the stored corpus.
_DRY_RUN_SESSION_ID = "dry-run-probe"


class StoreDryRunError(RuntimeError):
    """Raised when a startup connectivity dry-run against a selected store fails."""


def build_session_store(settings: Settings) -> SessionStore:
    """Select and construct the session-store implementation from settings.

    Args:
        settings: Application settings.

    Returns:
        `RedisSessionStore` when `redis_session_store_enabled` is true,
        otherwise `InMemorySessionStore`.
    """
    if settings.redis_session_store_enabled:
        if settings.redis_url is None:
            # Settings.validate_redis_session_store_requires_url() should have
            # already rejected this configuration; this is defense in depth.
            raise RuntimeError("redis_url is required when redis_session_store_enabled is True")
        return RedisSessionStore(
            redis_url=settings.redis_url, max_messages=settings.session_max_messages
        )
    return InMemorySessionStore(max_messages=settings.session_max_messages)


def build_vector_store(settings: Settings) -> VectorStore:
    """Select and construct the vector-store implementation from settings.

    Args:
        settings: Application settings.

    Returns:
        `ChromaVectorStore` or `OllamaEmbeddingVectorStore` when
        `vector_store_backend` selects one, otherwise `InMemoryVectorStore`.
    """
    if settings.vector_store_backend == "chroma":
        if settings.embedding_model:
            return ChromaVectorStore(embedding_model=settings.embedding_model)
        return ChromaVectorStore()

    if settings.vector_store_backend == "ollama":
        if settings.embedding_model is None:
            # Settings.validate_vector_store_backend_requires_embedding_model()
            # should have already rejected this configuration; defense in depth.
            raise RuntimeError("embedding_model is required when vector_store_backend is 'ollama'")
        if settings.embedding_base_url:
            return OllamaEmbeddingVectorStore(
                embedding_model=settings.embedding_model,
                base_url=str(settings.embedding_base_url),
            )
        return OllamaEmbeddingVectorStore(embedding_model=settings.embedding_model)

    return InMemoryVectorStore()


async def dry_run_stores(*stores: object) -> None:
    """Probe connectivity for every selected external store.

    In-memory stores and the embedded `ChromaVectorStore` (local library, no
    network service) have no external dependency to probe and are silently
    skipped. `RedisSessionStore` and `OllamaEmbeddingVectorStore` each make one
    real round-trip against their backing service; a failure is wrapped in
    `StoreDryRunError` so `create_app`'s lifespan can fail startup instead of
    deferring the error to the first request.

    Args:
        *stores: Store instances to probe (session and/or vector stores).

    Raises:
        StoreDryRunError: If any external store's connectivity probe fails.
    """
    for store in stores:
        if isinstance(store, RedisSessionStore):
            await _dry_run_redis_session_store(store)
        elif isinstance(store, OllamaEmbeddingVectorStore):
            await _dry_run_ollama_vector_store(store)


async def _dry_run_redis_session_store(store: RedisSessionStore) -> None:
    """Probe Redis connectivity via a harmless read of a reserved session id.

    Args:
        store: The Redis-backed session store to probe.

    Raises:
        StoreDryRunError: If the Redis round-trip fails.
    """
    try:
        await store.get_history(_DRY_RUN_SESSION_ID)
    except Exception as exc:
        raise StoreDryRunError(f"Redis session store dry-run failed: {exc}") from exc


async def _dry_run_ollama_vector_store(store: OllamaEmbeddingVectorStore) -> None:
    """Probe Ollama connectivity via a single non-destructive embeddings round-trip.

    Uses `store.ping()` rather than `add_documents()` + `clear()`: the latter
    would wipe the entire live corpus, which is harmless at startup (the
    corpus is still empty) but destructive if this same probe is reused by a
    periodic readiness check against a populated store (Req 13.1's
    `/health/ready` does exactly that).

    Args:
        store: The Ollama-backed vector store to probe.

    Raises:
        StoreDryRunError: If the embeddings round-trip fails.
    """
    try:
        await store.ping()
    except Exception as exc:
        raise StoreDryRunError(f"Ollama vector store dry-run failed: {exc}") from exc
