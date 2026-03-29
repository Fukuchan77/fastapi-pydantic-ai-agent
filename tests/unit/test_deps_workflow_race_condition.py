"""Tests for get_rag_workflow race condition fix (Task 29.1).

This test demonstrates the check-then-set race condition in get_rag_workflow
where concurrent requests can create multiple workflow instances instead of
sharing a single cached instance.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

from app.deps.workflow import get_rag_workflow


class MockVectorStore:
    """Mock VectorStore for testing (implements VectorStore protocol)."""

    async def add_documents(self, documents: list[str]) -> None:
        """Mock add_documents."""
        pass

    async def query(self, query: str, top_k: int = 5) -> list[str]:
        """Mock query."""
        return []

    async def clear(self) -> None:
        """Mock clear."""
        pass

    async def close(self) -> None:
        """Mock close."""
        pass


@pytest.fixture
def mock_app() -> FastAPI:
    """Create a FastAPI app with a mock vector store."""
    app = FastAPI()
    app.state.vector_store = MockVectorStore()
    return app


@pytest.fixture
def mock_request(mock_app: FastAPI) -> Any:
    """Create a mock request with the app state."""
    request = MagicMock()
    request.app = mock_app
    return request


@pytest.mark.asyncio
async def test_get_rag_workflow_race_condition(mock_request: Any) -> None:
    """Test that concurrent calls to get_rag_workflow don't create duplicate instances.

    RED PHASE: This test demonstrates the race condition where concurrent requests
    create multiple workflow instances instead of reusing the cached instance.

    The race condition occurs in the check-then-set pattern:
    1. Request A checks: vector_store not in cache -> True
    2. Request B checks: vector_store not in cache -> True
    3. Request A creates workflow and stores it
    4. Request B creates workflow and stores it (overwrites A's instance)

    Expected behavior: Only ONE workflow instance should be created
    Actual behavior (without fix): Multiple instances are created
    """
    from app.deps import workflow as workflow_module

    # Clear cache before test
    workflow_module._workflow_cache.clear()

    # Track how many workflow instances are created
    created_instances: list[Any] = []
    original_init = workflow_module.CorrectiveRAGWorkflow.__init__

    def tracked_init(self: Any, *args: Any, **kwargs: Any) -> None:
        """Track workflow instance creation."""
        created_instances.append(self)
        # Add a small delay to increase chance of race condition
        # This simulates the time taken to create the workflow
        import time

        time.sleep(0.01)
        original_init(self, *args, **kwargs)

    # Patch the __init__ method to track instance creation
    workflow_module.CorrectiveRAGWorkflow.__init__ = tracked_init  # type: ignore

    try:
        # Create multiple concurrent requests
        num_concurrent = 10
        tasks = [
            asyncio.create_task(asyncio.to_thread(get_rag_workflow, mock_request))
            for _ in range(num_concurrent)
        ]

        # Wait for all requests to complete
        workflows = await asyncio.gather(*tasks)

        # All requests should get the same workflow instance
        assert len(set(id(w) for w in workflows)) == 1, (
            f"Expected all requests to get the same workflow instance, "
            f"but got {len(set(id(w) for w in workflows))} different instances"
        )

        # Only ONE workflow instance should have been created
        assert len(created_instances) == 1, (
            f"Race condition detected: {len(created_instances)} workflow instances "
            f"were created instead of 1. This indicates the check-then-set pattern "
            f"is not thread-safe."
        )

    finally:
        # Restore original __init__
        workflow_module.CorrectiveRAGWorkflow.__init__ = original_init  # type: ignore
        # Clean up cache
        workflow_module._workflow_cache.clear()
