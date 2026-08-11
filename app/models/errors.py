"""Error response models for API endpoints."""

from pydantic import BaseModel
from pydantic import Field


class ErrorResponse(BaseModel):
    """Standard error response model.

    Used for HTTP error responses (401, 500, etc.) to provide a consistent
    error format across all endpoints. Every error response the service
    renders - regardless of status code or which layer raised it - uses this
    flat shape; see `app/api/errors.py` for the handlers that converge on it
    (Req 8.1, 8.4).

    Attributes:
        message: Human-readable error message
        code: Optional error code for programmatic error handling, drawn
            from the following documented, stable vocabulary (Req 8.3) so
            clients can branch on `code` instead of on body shape:

            - `UNAUTHORIZED` (401) - missing or invalid API key.
            - `FORBIDDEN` (403) - session-ownership rejection.
            - `NOT_FOUND` (404) - no route matches the request path. Raised
              by starlette's router, not by any route in this codebase.
            - `METHOD_NOT_ALLOWED` (405) - the path exists but not for this
              HTTP method; the response also carries an `Allow` header.
              Router-raised, as with 404.
            - `REQUEST_TOO_LARGE` (413) - request body exceeds the
              configured size limit (`app/middleware/request_size.py`).
            - `VALIDATION_ERROR` (422) - request-body validation failed;
              the message is generic and never echoes field-level detail
              (see `ValidationErrorDetail` below).
            - `RATE_LIMIT_EXCEEDED` (429) - the request-rate limit was hit.
            - `INTERNAL_ERROR` (500) - an unhandled exception.
            - `UPSTREAM_GROUNDING_FAILED` (502) - the RAG workflow could not
              produce a grounded, citable answer.
            - `DEPENDENCY_NOT_INITIALIZED` (503) - a required `app.state`
              singleton (`settings` or `llm_model`) was absent when a RAG
              query was handled.
            - `WORKFLOW_TIMEOUT` (504) - the RAG workflow exceeded its
              timeout.
    """

    message: str = Field(..., description="Human-readable error message")
    code: str | None = Field(default=None, description="Optional error code")


class ValidationErrorDetail(BaseModel):
    """Per-field validation error detail.

    Deliberately unused. Req 8.5 withholds internal detail - including
    request content - from every error body, and a 422 response built from
    this model would echo the client's own field names and values back to
    them. `app/api/errors.py::validation_exception_handler` returns a
    generic `ErrorResponse` instead, so this model documents the shape
    FastAPI's default 422 body used to carry rather than being constructed
    anywhere.

    Attributes:
        field: Name of the field that failed validation
        message: Human-readable error message
        type: Error type identifier (e.g., "value_error.email")
    """

    field: str = Field(..., description="Name of the field that failed validation")
    message: str = Field(..., description="Human-readable error message")
    type: str = Field(..., description="Error type identifier")
