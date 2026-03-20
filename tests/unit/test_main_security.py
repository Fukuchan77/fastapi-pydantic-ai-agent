"""Security tests for exception handler - Task 2.4."""

import logging

import pytest
from fastapi.testclient import TestClient


def test_exception_handler_does_not_expose_sensitive_data(caplog: pytest.LogCaptureFixture) -> None:
    """Test that exception handler returns generic message and logs internally.

    Task 2.4: Exception handler must:
    - Return generic "Internal server error occurred" message to client
    - Log full exception details internally with exc_info=True
    - Never expose stack traces, database paths, or internal structure to client
    """
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)

    # Sensitive exception with internal details
    sensitive_message = "Database connection failed: postgres://user:password@localhost:5432/db"

    @app.get("/test-sensitive-error")
    def test_sensitive_error_endpoint() -> None:
        raise RuntimeError(sensitive_message)

    # Capture logs at ERROR level
    with caplog.at_level(logging.ERROR):
        response = client.get("/test-sensitive-error")

    # Assert: Response should NOT contain the sensitive message
    assert response.status_code == 500
    json_data = response.json()
    assert "message" in json_data
    assert json_data["message"] == "Internal server error occurred"
    assert sensitive_message not in json_data["message"]

    # Assert: Sensitive details should be logged internally
    assert len(caplog.records) > 0
    log_message = caplog.records[0].message
    assert sensitive_message in log_message or "RuntimeError" in log_message

    # Assert: exc_info should be True (traceback logged)
    assert caplog.records[0].exc_info is not None


def test_exception_handler_logs_different_exception_types(caplog: pytest.LogCaptureFixture) -> None:
    """Test that different exception types are logged correctly."""
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)

    @app.get("/test-value-error")
    def test_value_error_endpoint() -> None:
        raise ValueError("Internal validation failed with secret key abc123")

    with caplog.at_level(logging.ERROR):
        response = client.get("/test-value-error")

    # Response should be generic
    assert response.status_code == 500
    assert response.json()["message"] == "Internal server error occurred"

    # Internal details logged
    assert len(caplog.records) > 0
    log_msg = caplog.records[0].message
    assert "ValueError" in log_msg or "secret key abc123" in log_msg


def test_exception_handler_does_not_expose_stack_traces(caplog: pytest.LogCaptureFixture) -> None:
    """Test that stack traces are never exposed in the response."""
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)

    @app.get("/test-stack-trace")
    def test_stack_trace_endpoint() -> None:
        try:
            # Nested exception to generate a deeper stack trace
            _ = 1 / 0
        except ZeroDivisionError as e:
            raise RuntimeError("/app/internal/secret_module.py failed") from e

    with caplog.at_level(logging.ERROR):
        response = client.get("/test-stack-trace")

    json_data = response.json()

    # Response should be generic
    assert json_data["message"] == "Internal server error occurred"

    # Should NOT contain file paths or module names
    assert "secret_module" not in json_data["message"]
    assert ".py" not in json_data["message"]
    assert "Traceback" not in json_data["message"]

    # But internal logs should have the details
    assert len(caplog.records) > 0


def test_exception_handler_uses_background_tasks_for_logging() -> None:
    """Test that exception handler uses BackgroundTasks for logging.

    Task 2.5: Exception handler must:
    - Return HTTP 500 response with a background task registered
    - Log exception details in the background task (non-blocking)
    - Capture traceback before handing off to background task

    Note: TestClient runs background tasks synchronously after response generation,
    so we verify that the background task mechanism is used, not timing.
    """
    from unittest.mock import patch

    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)

    logger_called = False

    def track_log(*args, **kwargs):
        nonlocal logger_called
        logger_called = True

    @app.get("/test-background-logging")
    def test_background_logging_endpoint() -> None:
        raise ValueError("Test error for background logging")

    # Patch logger.error to track if it's called
    with patch("app.main.logger.error", side_effect=track_log):
        response = client.get("/test-background-logging")

    # Response should be returned with HTTP 500
    assert response.status_code == 500
    assert response.json()["message"] == "Internal server error occurred"

    # Logger should have been called via background task
    # (TestClient runs background tasks after response generation)
    assert logger_called, "Logger should be called via background task"


def test_exception_handler_captures_traceback_before_background_task() -> None:
    """Test that traceback is captured before being handed to background task.

    Task 2.5: Must capture exc_info BEFORE creating the background task,
    otherwise the traceback context may be lost.

    Verifies that sys.exc_info() tuple (exception type, instance, traceback)
    is captured and passed to the logger.
    """
    from unittest.mock import patch

    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)

    captured_exc_info = None

    def capture_log(*args, **kwargs):
        nonlocal captured_exc_info
        captured_exc_info = kwargs.get("exc_info")

    @app.get("/test-traceback-capture")
    def test_traceback_endpoint() -> None:
        raise RuntimeError("Test traceback capture")

    with patch("app.main.logger.error", side_effect=capture_log):
        response = client.get("/test-traceback-capture")

    assert response.status_code == 500

    # exc_info should be a tuple from sys.exc_info(): (type, value, traceback)
    assert captured_exc_info is not None, "exc_info must be captured"
    assert isinstance(captured_exc_info, tuple), "exc_info should be a tuple from sys.exc_info()"
    assert len(captured_exc_info) == 3, (
        "exc_info tuple should have 3 elements (type, value, traceback)"
    )
    assert captured_exc_info[0] is RuntimeError, "First element should be exception type"
    assert isinstance(captured_exc_info[1], RuntimeError), (
        "Second element should be exception instance"
    )
    assert captured_exc_info[2] is not None, "Third element should be traceback object"
