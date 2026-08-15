"""FastAPI dependencies for workflow injection."""

import threading
import weakref

from fastapi import HTTPException
from fastapi import Request

from app.stores.vector_store import VectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


# Use WeakKeyDictionary to cache workflow instances keyed by vector_store object.
# This prevents memory leaks (vector stores can be GC'd) and id() collision bugs.
# Workflows are stateless (per-run state lives in Context), so reusing them
# is safe and avoids rebuilding Agent instances on every request.
_workflow_cache: weakref.WeakKeyDictionary[VectorStore, CorrectiveRAGWorkflow] = (
    weakref.WeakKeyDictionary()
)

# Use threading.Lock to prevent race condition in get_rag_workflow.
# Protects the check-then-set pattern from concurrent access.
_workflow_cache_lock: threading.Lock = threading.Lock()


def get_rag_workflow(req: Request) -> CorrectiveRAGWorkflow:
    """Return a cached CorrectiveRAGWorkflow instance for the given request.

    Reads `settings` and `llm_model` from `req.app.state` (Req 3.3) instead
    of process-global `get_settings()` and a model cache keyed on model name
    and base URL (removed - Req 3.7). `app.state.llm_model` is the single
    model instance `app/lifespan.py` resolves once per application and
    shares with the chat agent (Req 3.1, 3.2), so this dependency and the
    chat agent share exactly one model. Neither singleton falls back to a
    process-global default when absent (Req 3.4) - a fallback would hide the
    exact bug this dependency used to have.

    Caches workflow instances using WeakKeyDictionary keyed by vector_store object.
    This prevents memory leaks (deleted vector stores are auto-removed from cache) and
    avoids id() collision bugs (uses object identity, not id() which can be reused).

    Uses threading.Lock to prevent race condition in check-then-set pattern.
    Without the lock, concurrent requests could create multiple workflow instances
    instead of reusing the cached instance.

    Workflow instances are stateless (per-run state lives in llama-index
    Context objects), so reusing them is safe and avoids re-creating Agent
    instances on every request.

    Args:
        req: FastAPI request object, reading app.state.vector_store,
            app.state.settings, and app.state.llm_model.

    Returns:
        Cached CorrectiveRAGWorkflow instance configured with the
        application's vector store, injected settings, and shared LLM model.

    Raises:
        HTTPException: 503 with `code="DEPENDENCY_NOT_INITIALIZED"` when
            `app.state.settings` or `app.state.llm_model` is absent (Req 3.4).
    """
    vector_store = req.app.state.vector_store

    # Protect check-then-set with lock to prevent race condition
    with _workflow_cache_lock:
        if vector_store not in _workflow_cache:
            settings = getattr(req.app.state, "settings", None)
            model = getattr(req.app.state, "llm_model", None)
            if settings is None or model is None:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "message": "Required application dependency is not initialized.",
                        "code": "DEPENDENCY_NOT_INITIALIZED",
                    },
                )
            _workflow_cache[vector_store] = CorrectiveRAGWorkflow(
                vector_store=vector_store,
                llm_settings=settings,
                llm_model=model,
            )

        return _workflow_cache[vector_store]
