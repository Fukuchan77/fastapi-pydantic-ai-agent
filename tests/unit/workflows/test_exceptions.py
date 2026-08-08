"""Unit tests for RAG workflow exceptions.

Tests that workflow exception classes can be imported from their new location
and have the expected behavior for error classification and properties.
"""

import httpx
from pydantic_ai.exceptions import ModelHTTPError

from app.workflows.exceptions import DanglingCitationError
from app.workflows.exceptions import EmptyCitationError
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

    def test_base_error_is_not_transient_and_is_permanent_by_default(self) -> None:
        """The base class answers both properties from its own default (Req 7.1)."""
        error = RAGWorkflowError("Test error")
        assert error.is_transient is False
        assert error.is_permanent is True

    def test_is_error_transient_detects_rag_transient_error(self) -> None:
        """is_error_transient should return True for RAGTransientError."""
        transient = RAGTransientError("Temporary failure")
        assert RAGWorkflowError.is_error_transient(transient) is True

    def test_is_error_transient_detects_leaked_httpx_timeout_exception(self) -> None:
        """An unwrapped httpx.TimeoutException is transient defence-in-depth (Req 7.1)."""
        timeout = httpx.TimeoutException("Operation timed out")
        assert RAGWorkflowError.is_error_transient(timeout) is True

    def test_is_error_transient_detects_leaked_httpx_network_error(self) -> None:
        """An unwrapped httpx.NetworkError is transient defence-in-depth (Req 7.1)."""
        connection = httpx.ConnectError("Connection failed")
        assert RAGWorkflowError.is_error_transient(connection) is True

    def test_is_error_transient_ignores_builtin_timeout_and_connection_errors(self) -> None:
        """Builtin TimeoutError/ConnectionError carry no structured signal (Req 7.1)."""
        assert RAGWorkflowError.is_error_transient(TimeoutError("Operation timed out")) is False
        assert RAGWorkflowError.is_error_transient(ConnectionError("Connection failed")) is False

    def test_is_error_transient_classifies_rate_limit_by_status_code(self) -> None:
        """429 classifies as transient via `ModelHTTPError.status_code`, not message text."""
        error = ModelHTTPError(status_code=429, model_name="m", body="rate limit exceeded")
        assert RAGWorkflowError.is_error_transient(error) is True

    def test_is_error_transient_ignores_message_wording(self) -> None:
        """A message full of transient keywords does not flip a permanent status code."""
        error = ValueError("429 rate limit exceeded")
        assert RAGWorkflowError.is_error_transient(error) is False

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


class TestEmptyCitationError:
    """Tests for EmptyCitationError."""

    def test_empty_citation_error_inherits_from_base(self) -> None:
        """EmptyCitationError should extend RAGWorkflowError."""
        error = EmptyCitationError()
        assert isinstance(error, RAGWorkflowError)
        assert isinstance(error, Exception)

    def test_empty_citation_error_message_mentions_citations(self) -> None:
        """EmptyCitationError message should explain the missing-citation condition."""
        error = EmptyCitationError()
        assert "citation" in str(error).lower()


class TestDanglingCitationError:
    """Tests for DanglingCitationError."""

    def test_dangling_citation_error_inherits_from_base(self) -> None:
        """DanglingCitationError should extend RAGWorkflowError."""
        error = DanglingCitationError(unknown_ids={"memory::0099"}, known_ids={"memory::0000"})
        assert isinstance(error, RAGWorkflowError)
        assert isinstance(error, Exception)

    def test_dangling_citation_error_stores_unknown_and_known_ids(self) -> None:
        """DanglingCitationError should retain the unknown and known id sets."""
        error = DanglingCitationError(
            unknown_ids={"memory::0099", "memory::0100"},
            known_ids={"memory::0000", "memory::0001"},
        )
        assert error.unknown_ids == frozenset({"memory::0099", "memory::0100"})
        assert error.known_ids == frozenset({"memory::0000", "memory::0001"})

    def test_dangling_citation_error_message_lists_unknown_and_known_ids(self) -> None:
        """DanglingCitationError message should list every unknown id and the known set."""
        error = DanglingCitationError(unknown_ids={"memory::0099"}, known_ids={"memory::0000"})
        message = str(error)
        assert "memory::0099" in message
        assert "memory::0000" in message
