"""Tests for workflow instance caching and isolation.

Task 26.3: Verify that get_rag_workflow() returns cached CorrectiveRAGWorkflow
instances across requests. Workflows are stateless (per-run state lives in
llama-index Context objects), so reusing them is safe and avoids rebuilding
Agent instances on every request.

Note: This supersedes the Task 16.21 per-request isolation requirement.
"""

import pytest
from fastapi import FastAPI
from fastapi import Request

from app.deps.workflow import get_rag_workflow
from app.stores.vector_store import InMemoryVectorStore


@pytest.fixture
def mock_app() -> FastAPI:
    """Create a mock FastAPI app with vector_store in state."""
    app = FastAPI()
    app.state.vector_store = InMemoryVectorStore()
    return app


@pytest.fixture
def mock_request(mock_app: FastAPI) -> Request:
    """Create a mock Request object with app.state populated."""
    # Create a mock request scope with app attached
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "app": mock_app,  # Attach the app to the scope
    }
    return Request(scope=scope, receive=None)  # type: ignore


def test_get_rag_workflow_returns_cached_instance(mock_request: Request, mock_app: FastAPI) -> None:
    """Test that get_rag_workflow() returns the same cached instance on each call.

    Task 26.3: Verifies that workflow instances are cached and reused across
    requests. Workflows are stateless so sharing is safe and improves performance.
    """
    # Act: Call get_rag_workflow multiple times
    workflow1 = get_rag_workflow(mock_request)
    workflow2 = get_rag_workflow(mock_request)
    workflow3 = get_rag_workflow(mock_request)

    # Assert: Each call should return the same cached instance
    assert workflow1 is workflow2, "First and second calls should return the same cached instance"
    assert workflow2 is workflow3, "Second and third calls should return the same cached instance"

    # Assert: All instances should have the same vector_store reference (shared resource)
    assert workflow1.vector_store is workflow2.vector_store, (
        "Workflows should share the same vector_store from app.state"
    )
    assert workflow2.vector_store is workflow3.vector_store, (
        "Workflows should share the same vector_store from app.state"
    )


def test_different_vector_stores_get_different_workflows(
    mock_app: FastAPI,
) -> None:
    """Test that different vector_store instances produce different workflows.

    Task 26.3: Verifies that the cache key is the vector_store identity,
    so apps with different vector stores each get their own workflow instance.
    """
    app1 = FastAPI()
    app1.state.vector_store = InMemoryVectorStore()

    app2 = FastAPI()
    app2.state.vector_store = InMemoryVectorStore()

    def make_request(app: FastAPI) -> Request:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"",
            "headers": [],
            "app": app,
        }
        return Request(scope=scope, receive=None)  # type: ignore

    workflow1 = get_rag_workflow(make_request(app1))
    workflow2 = get_rag_workflow(make_request(app2))

    assert workflow1 is not workflow2, (
        "Different vector_store instances should produce different workflow instances"
    )
    assert workflow1.vector_store is app1.state.vector_store
    assert workflow2.vector_store is app2.state.vector_store
