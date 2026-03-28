"""RAG workflow exception hierarchy.

Domain-specific exceptions for the Corrective RAG workflow, providing
explicit error classification for better error handling and retry logic.
"""


class RAGWorkflowError(Exception):
    """Base exception for all RAG workflow errors.

    This is the base class for all RAG workflow-specific errors, providing
    a common interface for error handling and classification.

    Attributes:
        message: Human-readable error message
        is_transient: Whether this error is transient and can be retried
    """

    def __init__(self, message: str) -> None:
        """Initialize RAG workflow error.

        Args:
            message: Human-readable error message
        """
        super().__init__(message)
        self.message = message

    @property
    def is_transient(self) -> bool:
        """Whether this error is transient and should be retried.

        Returns:
            False by default (safer to not retry unless explicitly transient)
        """
        return False

    @staticmethod
    def is_error_transient(exception: Exception) -> bool:
        """Classify an exception as transient or permanent.

        This static method examines an exception and determines if it represents
        a transient error (timeout, rate limit, connection issue) that should be
        retried, or a permanent error that should not be retried.

        Args:
            exception: The exception to classify

        Returns:
            True if the error is transient and should be retried, False otherwise
        """
        # Check if it's already a RAGTransientError
        if isinstance(exception, RAGTransientError):
            return True

        # Check exception type - these are inherently transient
        if isinstance(exception, (TimeoutError, ConnectionError)):
            return True

        # Check error message for transient indicators
        error_msg = str(exception).lower()
        transient_indicators = [
            "timeout",
            "429",
            "rate limit",
            "503",
            "connection",
            "temporary",
            "unavailable",
        ]
        return any(indicator in error_msg for indicator in transient_indicators)


class RAGRetrievalError(RAGWorkflowError):
    """Exception raised during document retrieval from vector store.

    This error indicates that the retrieval step of the RAG workflow failed,
    typically due to vector store issues or query processing problems.

    Attributes:
        message: Human-readable error message
        query: The original query that failed to retrieve documents
    """

    def __init__(self, message: str, query: str | None = None) -> None:
        """Initialize retrieval error.

        Args:
            message: Human-readable error message
            query: Optional query that failed
        """
        super().__init__(message)
        self.query = query

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.query:
            return f"{self.message} (query: {self.query})"
        return self.message


class RAGEvaluationError(RAGWorkflowError):
    """Exception raised during relevance evaluation of retrieved chunks.

    This error indicates that the evaluation step (assessing relevance of
    retrieved documents) failed, typically due to LLM API issues.

    Attributes:
        message: Human-readable error message
        chunks_count: Number of chunks being evaluated when error occurred
    """

    def __init__(self, message: str, chunks_count: int | None = None) -> None:
        """Initialize evaluation error.

        Args:
            message: Human-readable error message
            chunks_count: Optional number of chunks being evaluated
        """
        super().__init__(message)
        self.chunks_count = chunks_count


class RAGSynthesisError(RAGWorkflowError):
    """Exception raised during answer synthesis from chunks.

    This error indicates that the synthesis step (generating final answer
    from relevant chunks) failed, typically due to LLM API issues.

    Attributes:
        message: Human-readable error message
        chunks_count: Number of chunks being synthesized when error occurred
    """

    def __init__(self, message: str, chunks_count: int | None = None) -> None:
        """Initialize synthesis error.

        Args:
            message: Human-readable error message
            chunks_count: Optional number of chunks being synthesized
        """
        super().__init__(message)
        self.chunks_count = chunks_count


class RAGTransientError(RAGWorkflowError):
    """Exception for transient errors that should be retried.

    This error indicates a temporary failure (timeout, rate limit, connection
    issue) that is likely to succeed on retry.

    Attributes:
        message: Human-readable error message
        cause: The underlying exception that caused this error
        is_transient: Always True for this error type
        is_permanent: Always False for this error type
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize transient error.

        Args:
            message: Human-readable error message
            cause: Optional underlying exception
        """
        super().__init__(message)
        self.cause = cause

    @property
    def is_transient(self) -> bool:
        """Transient errors should always be retried.

        Returns:
            True - this error type is always transient
        """
        return True

    @property
    def is_permanent(self) -> bool:
        """Transient errors are not permanent.

        Returns:
            False - this error type is never permanent
        """
        return False

    @classmethod
    def from_exception(cls, exception: Exception) -> "RAGTransientError":
        """Create a RAGTransientError from an existing exception.

        Args:
            exception: The underlying exception to wrap

        Returns:
            RAGTransientError wrapping the original exception
        """
        message = f"Transient error: {exception!s}"
        return cls(message, cause=exception)


class RAGPermanentError(RAGWorkflowError):
    """Exception for permanent errors that should not be retried.

    This error indicates a non-transient failure (authentication, invalid input,
    configuration error) that will not succeed on retry.

    Attributes:
        message: Human-readable error message
        cause: The underlying exception that caused this error
        is_transient: Always False for this error type
        is_permanent: Always True for this error type
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize permanent error.

        Args:
            message: Human-readable error message
            cause: Optional underlying exception
        """
        super().__init__(message)
        self.cause = cause

    @property
    def is_transient(self) -> bool:
        """Permanent errors should not be retried.

        Returns:
            False - this error type is never transient
        """
        return False

    @property
    def is_permanent(self) -> bool:
        """Permanent errors are not transient.

        Returns:
            True - this error type is always permanent
        """
        return True

    @classmethod
    def from_exception(cls, exception: Exception) -> "RAGPermanentError":
        """Create a RAGPermanentError from an existing exception.

        Args:
            exception: The underlying exception to wrap

        Returns:
            RAGPermanentError wrapping the original exception
        """
        message = f"Permanent error: {exception!s}"
        return cls(message, cause=exception)
