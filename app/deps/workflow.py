"""FastAPI dependencies for workflow injection."""

import threading
import weakref
from functools import lru_cache

from fastapi import Request
from pydantic_ai.models import Model

from app.agents.chat_agent import build_model
from app.config import get_settings
from app.stores.vector_store import VectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


# Task 28.1: Use WeakKeyDictionary to cache workflow instances keyed by vector_store object.
# This prevents memory leaks (vector stores can be GC'd) and id() collision bugs.
# Workflows are stateless (per-run state lives in Context), so reusing them
# is safe and avoids rebuilding Agent instances on every request.
_workflow_cache: weakref.WeakKeyDictionary[VectorStore, CorrectiveRAGWorkflow] = (
    weakref.WeakKeyDictionary()
)

# Task 29.1: Use threading.Lock to prevent race condition in get_rag_workflow.
# Protects the check-then-set pattern from concurrent access.
_workflow_cache_lock: threading.Lock = threading.Lock()


@lru_cache(maxsize=8)
def _get_cached_model(
    llm_model: str,
    llm_base_url: str | None = None,
) -> Model:
    """Cache the LLM model instance keyed by model name and base URL.

    Task 30.1: Changed from no-arg function to accept model configuration
    parameters as cache keys. This allows the cache to invalidate automatically
    when settings change (e.g., via hot reload or environment variable changes).

    The cache keys on llm_model and llm_base_url because these determine
    the model structure and endpoint. API keys are NOT included in the cache
    key for security (they shouldn't appear in logs/traces) and because
    changing only the API key doesn't require rebuilding the model instance.

    Cache size increased from 1 to 8 to support multiple model configurations
    (e.g., different models for different endpoints or A/B testing scenarios).

    Args:
        llm_model: The model identifier (e.g., "openai:gpt-4", "ollama:llama3").
        llm_base_url: Optional base URL for custom endpoints (e.g., Ollama local server).

    Returns:
        Cached Model instance for the given configuration.
    """
    settings = get_settings()
    return build_model(settings)


def get_rag_workflow(req: Request) -> CorrectiveRAGWorkflow:
    """Return a cached CorrectiveRAGWorkflow instance for the given request.

    Task 28.1: Caches workflow instances using WeakKeyDictionary keyed by vector_store object.
    This prevents memory leaks (deleted vector stores are auto-removed from cache) and
    avoids id() collision bugs (uses object identity, not id() which can be reused).

    Task 29.1: Uses threading.Lock to prevent race condition in check-then-set pattern.
    Without the lock, concurrent requests could create multiple workflow instances
    instead of reusing the cached instance.

    Workflow instances are stateless (per-run state lives in llama-index
    Context objects), so reusing them is safe and avoids re-creating Agent
    instances on every request.

    Args:
        req: FastAPI request object containing app.state.vector_store.

    Returns:
        Cached CorrectiveRAGWorkflow instance configured with the application's
        vector store and LLM settings.
    """
    vector_store = req.app.state.vector_store

    # Task 29.1: Protect check-then-set with lock to prevent race condition
    with _workflow_cache_lock:
        if vector_store not in _workflow_cache:
            settings = get_settings()
            # Task 30.1: Pass settings values as cache keys
            model = _get_cached_model(
                llm_model=settings.llm_model,
                llm_base_url=settings.llm_base_url,
            )
            _workflow_cache[vector_store] = CorrectiveRAGWorkflow(
                vector_store=vector_store,
                llm_settings=settings,
                llm_model=model,
            )

        return _workflow_cache[vector_store]
