"""Unit tests for type-based transient/permanent error classification.

Drives `RAGWorkflowError.is_error_transient()`'s rewrite from a message-keyword
classifier to a status-code-table classifier keyed on `ModelHTTPError.status_code`
(Req 7.1, 7.2, 7.3). The classes above `TestNamedCategoriesStopRetryImmediately`
are the subset that drove task 7.1's implementation; that class and
`TestExhaustiveStatusBucketTableIsMessageInvariant` are task 7.2's formal
widening to the full 7.7 acceptance criteria: the three named non-retryable
categories (authentication, authorization, request-validation) and an
exhaustive, message-poisoned status-bucket table.
"""

import httpx
import pytest
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.exceptions import ModelHTTPError

from app.workflows.exceptions import RAGPermanentError
from app.workflows.exceptions import RAGRetrievalError
from app.workflows.exceptions import RAGTransientError
from app.workflows.exceptions import RAGWorkflowError


class TestStatusCodeClassification:
    """`ModelHTTPError.status_code` is the classifier's primary signal."""

    def test_message_wording_does_not_influence_classification(self) -> None:
        """A 401 whose message contains a transient keyword is not retried."""
        error = ModelHTTPError(status_code=401, model_name="m", body="connection timeout 503")
        assert RAGWorkflowError.is_error_transient(error) is False

    def test_429_is_transient(self) -> None:
        """429 (rate limit) is in the frozen transient set."""
        error = ModelHTTPError(status_code=429, model_name="m")
        assert RAGWorkflowError.is_error_transient(error) is True

    def test_500_502_503_504_are_transient(self) -> None:
        """The full 5xx upstream-unavailability bucket is transient."""
        for code in (500, 502, 503, 504):
            error = ModelHTTPError(status_code=code, model_name="m")
            assert RAGWorkflowError.is_error_transient(error) is True, code

    def test_408_409_are_transient(self) -> None:
        """Request timeout and conflict are in the frozen transient set."""
        for code in (408, 409):
            error = ModelHTTPError(status_code=code, model_name="m")
            assert RAGWorkflowError.is_error_transient(error) is True, code

    def test_401_403_400_422_are_permanent(self) -> None:
        """Auth, authz, and validation status codes default to permanent."""
        for code in (400, 401, 403, 422):
            error = ModelHTTPError(status_code=code, model_name="m")
            assert RAGWorkflowError.is_error_transient(error) is False, code

    def test_unrecognized_status_code_defaults_permanent(self) -> None:
        """Any status code outside the frozen set defaults to permanent."""
        error = ModelHTTPError(status_code=418, model_name="m")
        assert RAGWorkflowError.is_error_transient(error) is False


class TestFallbackExceptionGroupUnwrapping:
    """`FallbackExceptionGroup` is raised even for a chain of one model."""

    def test_unwraps_single_member_chain(self) -> None:
        """A chain-of-one group classifies by its single wrapped error."""
        group = FallbackExceptionGroup(
            "All models from FallbackModel failed",
            [ModelHTTPError(status_code=503, model_name="m")],
        )
        assert RAGWorkflowError.is_error_transient(group) is True

    def test_unwraps_multi_member_chain_by_final_error(self) -> None:
        """A multi-model chain classifies by the last (final) attempt's error."""
        group = FallbackExceptionGroup(
            "All models from FallbackModel failed",
            [
                ModelHTTPError(status_code=500, model_name="primary"),
                ModelHTTPError(status_code=401, model_name="fallback"),
            ],
        )
        assert RAGWorkflowError.is_error_transient(group) is False


class TestDefenceInDepthTypeArms:
    """Narrow type arms for project errors and leaked httpx errors."""

    def test_rag_transient_error_is_transient(self) -> None:
        """A RAGTransientError instance classifies as transient."""
        assert RAGWorkflowError.is_error_transient(RAGTransientError("boom")) is True

    def test_rag_permanent_error_is_not_transient(self) -> None:
        """A RAGPermanentError instance classifies as permanent."""
        assert RAGWorkflowError.is_error_transient(RAGPermanentError("boom")) is False

    def test_leaked_httpx_timeout_exception_is_transient(self) -> None:
        """An unwrapped httpx.TimeoutException classifies as transient."""
        error = httpx.TimeoutException("timed out")
        assert RAGWorkflowError.is_error_transient(error) is True

    def test_leaked_httpx_network_error_is_transient(self) -> None:
        """An unwrapped httpx.NetworkError subclass classifies as transient."""
        error = httpx.ConnectError("connection failed")  # a NetworkError subclass
        assert RAGWorkflowError.is_error_transient(error) is True

    def test_unrecognized_exception_type_defaults_permanent(self) -> None:
        """A bare ValueError with no structured signal defaults to permanent."""
        assert RAGWorkflowError.is_error_transient(ValueError("connection timeout 503")) is False


class TestNamedCategoriesStopRetryImmediately:
    """7.2's three named non-retryable categories, proven message-invariant per 7.7."""

    @pytest.mark.parametrize(
        "status_code",
        [401, 403, 400, 422],
        ids=["authentication", "authorization", "validation_400", "validation_422"],
    )
    def test_named_category_with_transient_keyword_message_is_not_retried(
        self, status_code: int
    ) -> None:
        """Auth/authz/validation stay permanent despite retry-suggestive wording."""
        error = ModelHTTPError(
            status_code=status_code,
            model_name="m",
            body="connection timeout, please retry - 503 upstream unavailable",
        )
        assert RAGWorkflowError.is_error_transient(error) is False


class TestExhaustiveStatusBucketTableIsMessageInvariant:
    """Every frozen-table status code maps as specified regardless of message wording."""

    @pytest.mark.parametrize(
        ("status_code", "expected_transient"),
        [
            (408, True),
            (409, True),
            (429, True),
            (500, True),
            (502, True),
            (503, True),
            (504, True),
            (400, False),
            (401, False),
            (403, False),
            (404, False),
            (405, False),
            (410, False),
            (418, False),
            (422, False),
            (451, False),
        ],
    )
    def test_status_bucket_with_poisoned_message(
        self, status_code: int, expected_transient: bool
    ) -> None:
        """Each bucket survives a message engineered to fool a keyword matcher."""
        poisoned_body = (
            "permanent failure, do not retry"
            if expected_transient
            else "connection timeout retry 503 rate limit unavailable"
        )
        error = ModelHTTPError(status_code=status_code, model_name="m", body=poisoned_body)
        assert RAGWorkflowError.is_error_transient(error) is expected_transient


class TestIsPermanentLiftedToBaseClass:
    """Every subclass must answer `is_permanent` instead of raising."""

    def test_base_class_is_permanent_by_default(self) -> None:
        """The base class answers is_permanent True by default (not is_transient)."""
        assert RAGWorkflowError("boom").is_permanent is True

    def test_subclass_with_no_override_answers_is_permanent(self) -> None:
        """RAGRetrievalError never overrode is_permanent before this task."""
        assert RAGRetrievalError("boom").is_permanent is True

    def test_transient_error_is_permanent_is_false(self) -> None:
        """RAGTransientError's is_permanent is False via the base's negation."""
        assert RAGTransientError("boom").is_permanent is False

    def test_permanent_error_is_permanent_is_true(self) -> None:
        """RAGPermanentError's is_permanent is True via the base's negation."""
        assert RAGPermanentError("boom").is_permanent is True
