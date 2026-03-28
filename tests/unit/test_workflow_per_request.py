"""Tests for per-request workflow instance isolation.

Task 16.21: Verify that get_rag_workflow() creates a new CorrectiveRAGWorkflow
instance for each request to ensure proper isolation between concurrent requests.
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


def test_get_rag_workflow_returns_new_instance_per_call(
    mock_request: Request, mock_app: FastAPI
) -> None:
    """Test that get_rag_workflow() returns a new instance on each call.

    This verifies that workflow instances are not shared between requests,
    ensuring proper isolation for concurrent request handling.
    """
    # Act: Call get_rag_workflow multiple times
    workflow1 = get_rag_workflow(mock_request)
    workflow2 = get_rag_workflow(mock_request)
    workflow3 = get_rag_workflow(mock_request)

    # Assert: Each call should return a different instance
    assert workflow1 is not workflow2, "First and second calls should return different instances"
    assert workflow2 is not workflow3, "Second and third calls should return different instances"
    assert workflow1 is not workflow3, "First and third calls should return different instances"

    # Assert: All instances should have the same vector_store reference (shared resource)
    assert workflow1.vector_store is workflow2.vector_store, (
        "Workflows should share the same vector_store from app.state"
    )
    assert workflow2.vector_store is workflow3.vector_store, (
        "Workflows should share the same vector_store from app.state"
    )


def test_workflow_instances_have_independent_state(
    mock_request: Request, mock_app: FastAPI
) -> None:
    """Test that workflow instances maintain independent state.

    Verifies that modifying one workflow instance doesn't affect another,
    even though they share immutable configuration (vector_store, llm_settings).
    """
    # Act: Create two workflow instances
    workflow1 = get_rag_workflow(mock_request)
    workflow2 = get_rag_workflow(mock_request)

    # Assert: Workflows are different objects with their own identity
    assert id(workflow1) != id(workflow2), "Workflows should have different object IDs"

    # Assert: Both workflows have their own agent instances
    # (Even though they use the same model/settings, they're separate Agent objects)
    assert workflow1._eval_agent is not workflow2._eval_agent, (
        "Each workflow should have its own eval_agent instance"
    )
    assert workflow1._synth_agent is not workflow2._synth_agent, (
        "Each workflow should have its own synth_agent instance"
    )
