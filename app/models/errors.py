"""Error response models for API endpoints."""

from pydantic import BaseModel
from pydantic import Field


class ErrorResponse(BaseModel):
    """Standard error response model.

    Used for HTTP error responses (401, 500, etc.) to provide a consistent
    error format across all endpoints.

    Attributes:
        message: Human-readable error message
        code: Optional error code for programmatic error handling
    """

    message: str = Field(..., description="Human-readable error message")
    code: str | None = Field(default=None, description="Optional error code")


class ValidationErrorDetail(BaseModel):
    """Detailed validation error information.

    Used in HTTP 422 responses to provide field-level validation error details.

    Attributes:
        field: Name of the field that failed validation
        message: Human-readable error message
        type: Error type identifier (e.g., "value_error.email")
    """

    field: str = Field(..., description="Name of the field that failed validation")
    message: str = Field(..., description="Human-readable error message")
    type: str = Field(..., description="Error type identifier")
