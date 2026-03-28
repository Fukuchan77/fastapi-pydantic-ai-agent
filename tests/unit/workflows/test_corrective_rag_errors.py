"""Tests for RAG workflow error classification.

Task: Fix RAG Workflow error classification at app/workflows/corrective_rag.py:337
by introducing an explicit error hierarchy for better error handling.
"""

import pytest

from app.workflows.exceptions import RAGEvaluationError
from app.workflows.exceptions import RAGPermanentError
from app.workflows.exceptions import RAGRetrievalError
from app.workflows.exceptions import RAGSynthesisError
from app.workflows.exceptions import RAGTransientError
from app.workflows.exceptions import RAGWorkflowError


class TestRAGErrorHierarchy:
    """Test the RAG error hierarchy structure."""

    def test_base_rag_error_exists(self):
        """Base RAGWorkflowError should exist."""
        error = RAGWorkflowError("test error")
        assert str(error) == "test error"
        assert isinstance(error, Exception)

    def test_retrieval_error_inherits_from_base(self):
        """RAGRetrievalError should inherit from RAGWorkflowError."""
        error = RAGRetrievalError("retrieval failed")
        assert isinstance(error, RAGWorkflowError)
        assert isinstance(error, Exception)

    def test_evaluation_error_inherits_from_base(self):
        """RAGEvaluationError should inherit from RAGWorkflowError."""
        error = RAGEvaluationError("evaluation failed")
        assert isinstance(error, RAGWorkflowError)
        assert isinstance(error, Exception)

    def test_synthesis_error_inherits_from_base(self):
        """RAGSynthesisError should inherit from RAGWorkflowError."""
        error = RAGSynthesisError("synthesis failed")
        assert isinstance(error, RAGWorkflowError)
        assert isinstance(error, Exception)

    def test_transient_error_inherits_from_base(self):
        """RAGTransientError should inherit from RAGWorkflowError."""
        error = RAGTransientError("temporary failure")
        assert isinstance(error, RAGWorkflowError)
        assert isinstance(error, Exception)

    def test_permanent_error_inherits_from_base(self):
        """RAGPermanentError should inherit from RAGWorkflowError."""
        error = RAGPermanentError("permanent failure")
        assert isinstance(error, RAGWorkflowError)
        assert isinstance(error, Exception)


class TestRAGErrorClassification:
    """Test error classification as transient or permanent."""

    def test_transient_error_is_retryable(self):
        """Transient errors should be marked as retryable."""
        error = RAGTransientError("timeout")
        assert error.is_transient is True
        assert error.is_permanent is False

    def test_permanent_error_is_not_retryable(self):
        """Permanent errors should not be retryable."""
        error = RAGPermanentError("authentication failed")
        assert error.is_transient is False
        assert error.is_permanent is True

    def test_base_error_classification_defaults(self):
        """Base RAGWorkflowError should have sensible defaults."""
        error = RAGWorkflowError("generic error")
        # By default, assume not transient (safer to not retry)
        assert error.is_transient is False


class TestRAGErrorContext:
    """Test that errors carry useful context."""

    def test_retrieval_error_with_query(self):
        """RAGRetrievalError should store query context."""
        error = RAGRetrievalError("failed to retrieve", query="test query")
        assert error.query == "test query"
        assert "test query" in str(error) or "failed to retrieve" in str(error)

    def test_evaluation_error_with_chunks_count(self):
        """RAGEvaluationError should store chunk count context."""
        error = RAGEvaluationError("evaluation failed", chunks_count=5)
        assert error.chunks_count == 5

    def test_synthesis_error_with_chunks_count(self):
        """RAGSynthesisError should store chunk count context."""
        error = RAGSynthesisError("synthesis failed", chunks_count=3)
        assert error.chunks_count == 3

    def test_transient_error_with_cause(self):
        """Transient errors should store the underlying cause."""
        cause = TimeoutError("connection timeout")
        error = RAGTransientError("temporary failure", cause=cause)
        assert error.cause is cause
        assert isinstance(error.cause, TimeoutError)


class TestRAGErrorFactoryMethods:
    """Test factory methods for creating specific error types."""

    def test_create_transient_from_timeout(self):
        """Should create transient error from timeout exception."""
        timeout = TimeoutError("request timeout")
        error = RAGTransientError.from_exception(timeout)
        assert isinstance(error, RAGTransientError)
        assert error.cause is timeout

    def test_create_transient_from_rate_limit(self):
        """Should create transient error from rate limit exception."""
        rate_limit = Exception("429 rate limit exceeded")
        error = RAGTransientError.from_exception(rate_limit)
        assert isinstance(error, RAGTransientError)
        assert "rate limit" in str(error).lower() or "429" in str(error)

    def test_create_permanent_from_auth_error(self):
        """Should create permanent error from auth exception."""
        auth_error = Exception("401 unauthorized")
        error = RAGPermanentError.from_exception(auth_error)
        assert isinstance(error, RAGPermanentError)
        assert error.cause is auth_error

    def test_classify_error_as_transient_or_permanent(self):
        """Should classify unknown exceptions as transient or permanent."""
        # Timeout should be transient
        timeout = TimeoutError("timeout")
        assert RAGWorkflowError.is_error_transient(timeout) is True

        # Generic exception should be permanent (safer default)
        generic = Exception("unknown error")
        assert RAGWorkflowError.is_error_transient(generic) is False

        # Connection errors should be transient
        connection = ConnectionError("connection failed")
        assert RAGWorkflowError.is_error_transient(connection) is True


class TestRAGErrorIntegration:
    """Test error hierarchy integration with workflow."""

    def test_errors_can_be_caught_by_base_class(self):
        """All RAG errors should be catchable by RAGWorkflowError."""
        errors = [
            RAGRetrievalError("retrieval"),
            RAGEvaluationError("evaluation"),
            RAGSynthesisError("synthesis"),
            RAGTransientError("transient"),
            RAGPermanentError("permanent"),
        ]

        for error in errors:
            with pytest.raises(RAGWorkflowError):
                raise error

    def test_transient_errors_can_be_caught_separately(self):
        """Transient errors should be catchable for retry logic."""
        with pytest.raises(RAGTransientError) as exc_info:
            raise RAGTransientError("timeout")
        assert exc_info.value.is_transient is True

    def test_permanent_errors_can_be_caught_separately(self):
        """Permanent errors should be catchable to avoid retries."""
        with pytest.raises(RAGPermanentError) as exc_info:
            raise RAGPermanentError("auth failed")
        assert exc_info.value.is_permanent is True
