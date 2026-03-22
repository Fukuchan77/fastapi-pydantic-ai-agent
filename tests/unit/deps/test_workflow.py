"""Unit tests for workflow dependency functions."""

from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Clear get_settings() cache before each test."""
    from app.config import get_settings

    get_settings.cache_clear()


def test_get_rag_workflow_returns_workflow_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_rag_workflow returns a CorrectiveRAGWorkflow instance."""
    # Setup: Configure environment for Settings
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Import after setting env vars
    from app.deps.workflow import get_rag_workflow
    from app.stores.vector_store import InMemoryVectorStore
    from app.workflows.corrective_rag import CorrectiveRAGWorkflow

    # Arrange: Create mock request with vector_store in app.state
    mock_request = Mock()
    mock_vector_store = InMemoryVectorStore()
    mock_request.app.state.vector_store = mock_vector_store

    # Act: Call the dependency function
    workflow = get_rag_workflow(mock_request)

    # Assert: Should return CorrectiveRAGWorkflow instance
    assert isinstance(workflow, CorrectiveRAGWorkflow)


def test_get_rag_workflow_uses_app_state_vector_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_rag_workflow uses vector store from app.state."""
    # Setup: Configure environment for Settings
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Import after setting env vars
    from app.deps.workflow import get_rag_workflow
    from app.stores.vector_store import InMemoryVectorStore

    # Arrange: Create mock request with specific vector store
    mock_request = Mock()
    mock_vector_store = InMemoryVectorStore()
    mock_request.app.state.vector_store = mock_vector_store

    # Act: Call the dependency function
    workflow = get_rag_workflow(mock_request)

    # Assert: Workflow should use the same vector store instance
    assert workflow.vector_store is mock_vector_store


def test_get_rag_workflow_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_rag_workflow passes settings to workflow."""
    # Setup: Configure environment for Settings
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Import after setting env vars
    from app.config import Settings
    from app.deps.workflow import get_rag_workflow
    from app.stores.vector_store import InMemoryVectorStore

    # Arrange: Create mock request
    mock_request = Mock()
    mock_request.app.state.vector_store = InMemoryVectorStore()

    # Act: Call the dependency function
    workflow = get_rag_workflow(mock_request)

    # Assert: Workflow should have llm_settings
    assert isinstance(workflow.llm_settings, Settings)
    assert workflow.llm_settings.llm_model is not None
