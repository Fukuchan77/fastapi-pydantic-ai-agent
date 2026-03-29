"""FastAPI dependency injection functions.

Module initializer that exports public dependency functions.

This module provides a clean public API for dependency injection functions
used throughout the FastAPI application. Instead of importing from submodules
like app.deps.auth or app.deps.workflow, consumers can import directly from
app.deps.
"""

from app.deps.auth import verify_api_key
from app.deps.workflow import get_rag_workflow


__all__ = [
    "get_rag_workflow",
    "verify_api_key",
]
