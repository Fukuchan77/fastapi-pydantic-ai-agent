"""Unit tests for readiness health check endpoint.

Task 16.1: Deep readiness health check that verifies dependencies are operational.
"""

from unittest.mock import MagicMock


def test_readiness_check_import() -> None:
    """Test that readiness_check function can be imported."""
    from app.api.health import readiness_check

    assert readiness_check is not None


def test_readiness_check_with_all_dependencies_healthy() -> None:
    """Test readiness check returns ready when all dependencies are healthy."""
    from app.api.health import readiness_check

    # Arrange: Mock Request with all healthy dependencies
    mock_request = MagicMock()
    mock_request.app.state.vector_store = MagicMock()
    mock_request.app.state.session_store = MagicMock()
    mock_request.app.state.chat_agent = MagicMock()
    mock_request.app.state.cleanup_task = MagicMock()
    mock_request.app.state.cleanup_task.done.return_value = False  # Task still running

    # Act: Call readiness check
    result = readiness_check(mock_request)

    # Assert: Should return ready status
    assert result["status"] == "ready"
    assert result["checks"]["vector_store"] == "healthy"
    assert result["checks"]["session_store"] == "healthy"
    assert result["checks"]["chat_agent"] == "healthy"
    assert result["checks"]["cleanup_task"] == "healthy"


def test_readiness_check_with_missing_vector_store() -> None:
    """Test readiness check returns not ready when vector_store is missing."""
    from app.api.health import readiness_check

    # Arrange: Mock Request with missing vector_store
    mock_request = MagicMock()
    mock_request.app.state.session_store = MagicMock()
    mock_request.app.state.chat_agent = MagicMock()
    mock_request.app.state.cleanup_task = MagicMock()
    mock_request.app.state.cleanup_task.done.return_value = False
    # Remove vector_store to simulate missing dependency
    del mock_request.app.state.vector_store

    # Act: Call readiness check
    result = readiness_check(mock_request)

    # Assert: Should return not ready status
    assert result["status"] == "not_ready"
    assert result["checks"]["vector_store"] == "missing"


def test_readiness_check_with_stopped_cleanup_task() -> None:
    """Test readiness check returns not ready when cleanup task has stopped."""
    from app.api.health import readiness_check

    # Arrange: Mock Request with stopped cleanup_task
    mock_request = MagicMock()
    mock_request.app.state.vector_store = MagicMock()
    mock_request.app.state.session_store = MagicMock()
    mock_request.app.state.chat_agent = MagicMock()
    mock_request.app.state.cleanup_task = MagicMock()
    mock_request.app.state.cleanup_task.done.return_value = True  # Task stopped

    # Act: Call readiness check
    result = readiness_check(mock_request)

    # Assert: Should return not ready status
    assert result["status"] == "not_ready"
    assert result["checks"]["cleanup_task"] == "stopped"


def test_readiness_check_with_all_dependencies_missing() -> None:
    """Test readiness check returns not ready when all dependencies are missing."""
    from app.api.health import readiness_check

    # Arrange: Mock Request with all dependencies missing
    mock_request = MagicMock()
    # Remove all dependencies
    del mock_request.app.state.vector_store
    del mock_request.app.state.session_store
    del mock_request.app.state.chat_agent
    del mock_request.app.state.cleanup_task

    # Act: Call readiness check
    result = readiness_check(mock_request)

    # Assert: Should return not ready status with all checks missing
    assert result["status"] == "not_ready"
    assert result["checks"]["vector_store"] == "missing"
    assert result["checks"]["session_store"] == "missing"
    assert result["checks"]["chat_agent"] == "missing"
    assert result["checks"]["cleanup_task"] == "missing"


def test_readiness_check_response_structure() -> None:
    """Test readiness check returns dict with required keys."""
    from app.api.health import readiness_check

    # Arrange: Mock Request with healthy dependencies
    mock_request = MagicMock()
    mock_request.app.state.vector_store = MagicMock()
    mock_request.app.state.session_store = MagicMock()
    mock_request.app.state.chat_agent = MagicMock()
    mock_request.app.state.cleanup_task = MagicMock()
    mock_request.app.state.cleanup_task.done.return_value = False

    # Act: Call readiness check
    result = readiness_check(mock_request)

    # Assert: Verify response structure
    assert isinstance(result, dict)
    assert "status" in result
    assert "checks" in result
    assert isinstance(result["checks"], dict)
    assert all(
        key in result["checks"]
        for key in ["vector_store", "session_store", "chat_agent", "cleanup_task"]
    )
