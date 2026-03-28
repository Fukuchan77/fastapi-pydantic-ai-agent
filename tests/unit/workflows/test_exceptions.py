"""Unit tests for RAG workflow exceptions.

Tests that workflow exception classes can be imported from their new location
and have the expected behavior for error classification and properties.
"""

from app.workflows.exceptions import RAGEvaluationError
from app.workflows.exceptions import RAGPermanentError
from app.workflows.exceptions import RAGRetrievalError
from app.workflows.exceptions import RAGSynthesisError
from app.workflows.exceptions import RAGTransientError
from app.workflows.exceptions import RAGWorkflowError


class TestRAGWorkflowError:
    """Tests for RAGWorkflowError base class."""

    def test_base_error_has_message(self) -> None:
        """RAGWorkflowError should store message attribute."""
        error = RAGWorkflowError("Test error")
        assert error.message == "Test error"
        assert str(error) == "Test error"

    def test_base_error_is_not_transient_by_default(self) -> None:
        """RAGWorkflowError should not be transient by default (safer)."""
        error = RAGWorkflowError("Test error")
        assert error.is_transient is False

    def test_is_error_transient_detects_rag_transient_error(self) -> None:
        """is_error_transient should return True for RAGTransientError."""
        transient = RAGTransientError("Temporary failure")
        assert RAGWorkflowError.is_error_transient(transient) is True

    def test_is_error_transient_detects_timeout_error(self) -> None:
        """is_error_transient should return True for TimeoutError."""
        timeout = TimeoutError("Operation timed out")
        assert RAGWorkflowError.is_error_transient(timeout) is True

    def test_is_error_transient_detects_connection_error(self) -> None:
        """is_error_transient should return True for ConnectionError."""
        connection = ConnectionError("Connection failed")
        assert RAGWorkflowError.is_error_transient(connection) is True

    def test_is_error_transient_detects_rate_limit_from_message(self) -> None:
        """is_error_transient should detect rate limit from error message."""
        error = ValueError("429 rate limit exceeded")
        assert RAGWorkflowError.is_error_transient(error) is True

    def test_is_error_transient_returns_false_for_permanent_errors(self) -> None:
        """is_error_transient should return False for non-transient errors."""
        error = ValueError("Invalid input")
        assert RAGWorkflowError.is_error_transient(error) is False


class TestRAGRetrievalError:
    """Tests for RAGRetrievalError."""

    def test_retrieval_error_with_query(self) -> None:
        """RAGRetrievalError should include query in string representation."""
        error = RAGRetrievalError("Retrieval failed", query="test query")
        assert error.query == "test query"
        assert "test query" in str(error)

    def test_retrieval_error_without_query(self) -> None:
        """RAGRetrievalError should work without query parameter."""
        error = RAGRetrievalError("Retrieval failed")
        assert error.query is None
        assert str(error) == "Retrieval failed"


class TestRAGEvaluationError:
    """Tests for RAGEvaluationError."""

    def test_evaluation_error_with_chunks_count(self) -> None:
        """RAGEvaluationError should store chunks_count."""
        error = RAGEvaluationError("Evaluation failed", chunks_count=5)
        assert error.chunks_count == 5

    def test_evaluation_error_without_chunks_count(self) -> None:
        """RAGEvaluationError should work without chunks_count."""
        error = RAGEvaluationError("Evaluation failed")
        assert error.chunks_count is None


class TestRAGSynthesisError:
    """Tests for RAGSynthesisError."""

    def test_synthesis_error_with_chunks_count(self) -> None:
        """RAGSynthesisError should store chunks_count."""
        error = RAGSynthesisError("Synthesis failed", chunks_count=3)
        assert error.chunks_count == 3

    def test_synthesis_error_without_chunks_count(self) -> None:
        """RAGSynthesisError should work without chunks_count."""
        error = RAGSynthesisError("Synthesis failed")
        assert error.chunks_count is None


class TestRAGTransientError:
    """Tests for RAGTransientError."""

    def test_transient_error_is_transient(self) -> None:
        """RAGTransientError should always be transient."""
        error = RAGTransientError("Temporary failure")
        assert error.is_transient is True
        assert error.is_permanent is False

    def test_transient_error_with_cause(self) -> None:
        """RAGTransientError should store underlying cause."""
        cause = TimeoutError("Original timeout")
        error = RAGTransientError("Transient error", cause=cause)
        assert error.cause is cause

    def test_transient_error_from_exception(self) -> None:
        """RAGTransientError.from_exception should wrap original exception."""
        original = ConnectionError("Connection failed")
        error = RAGTransientError.from_exception(original)
        assert error.cause is original
        assert "Connection failed" in error.message


class TestRAGPermanentError:
    """Tests for RAGPermanentError."""

    def test_permanent_error_is_not_transient(self) -> None:
        """RAGPermanentError should never be transient."""
        error = RAGPermanentError("Authentication failed")
        assert error.is_transient is False
        assert error.is_permanent is True

    def test_permanent_error_with_cause(self) -> None:
        """RAGPermanentError should store underlying cause."""
        cause = ValueError("Invalid API key")
        error = RAGPermanentError("Permanent error", cause=cause)
        assert error.cause is cause

    def test_permanent_error_from_exception(self) -> None:
        """RAGPermanentError.from_exception should wrap original exception."""
        original = ValueError("Invalid input")
        error = RAGPermanentError.from_exception(original)
        assert error.cause is original
        assert "Invalid input" in error.message
