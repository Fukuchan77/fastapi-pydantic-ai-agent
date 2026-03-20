"""Unit tests for error response models."""

import pytest
from pydantic import ValidationError


def test_error_response_import_fails() -> None:
    """Test that ErrorResponse can be imported (RED: should fail initially)."""
    from app.models.errors import ErrorResponse

    # Test basic instantiation
    error = ErrorResponse(message="Test error")
    assert error.message == "Test error"
    assert error.code is None


def test_error_response_with_code() -> None:
    """Test ErrorResponse with optional code field."""
    from app.models.errors import ErrorResponse

    error = ErrorResponse(message="Test error", code="TEST_CODE")
    assert error.message == "Test error"
    assert error.code == "TEST_CODE"


def test_error_response_message_required() -> None:
    """Test that message field is required."""
    from app.models.errors import ErrorResponse

    with pytest.raises(ValidationError) as exc_info:
        ErrorResponse()  # type: ignore[call-arg]

    errors = exc_info.value.errors()
    assert any(e["loc"] == ("message",) and e["type"] == "missing" for e in errors)


def test_validation_error_detail_all_fields() -> None:
    """Test ValidationErrorDetail with all required fields."""
    from app.models.errors import ValidationErrorDetail

    detail = ValidationErrorDetail(
        field="email",
        message="Invalid email format",
        type="value_error.email",
    )
    assert detail.field == "email"
    assert detail.message == "Invalid email format"
    assert detail.type == "value_error.email"


def test_validation_error_detail_fields_required() -> None:
    """Test that all ValidationErrorDetail fields are required."""
    from app.models.errors import ValidationErrorDetail

    with pytest.raises(ValidationError) as exc_info:
        ValidationErrorDetail()  # type: ignore[call-arg]

    errors = exc_info.value.errors()
    assert len(errors) == 3  # field, message, type all missing
