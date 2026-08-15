"""Unit tests for CORS wildcard warning at startup.

This test ensures that a warning is logged when CORS_ORIGINS contains "*"
to prevent accidental production misconfiguration.
"""

import logging

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app


def _build_settings(**overrides: object) -> Settings:
    """Build a valid Settings instance directly, without touching os.environ."""
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "openai:gpt-4",
        "llm_api_key": "test-llm-key-12345",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _has_cors_wildcard_warning(records: list[logging.LogRecord]) -> bool:
    """Return True if any warning record mentions CORS and a wildcard."""
    return any(
        "cors" in record.message.lower()
        and ("*" in record.message or "wildcard" in record.message.lower())
        for record in records
        if record.levelno == logging.WARNING
    )


def test_cors_wildcard_logs_warning_at_startup(caplog: pytest.LogCaptureFixture) -> None:
    """Test that a warning is logged when CORS_ORIGINS contains wildcard.

    Prevent accidental production misconfiguration by warning
    when CORS allows all origins.
    """
    app = create_app(settings=_build_settings(cors_origins="*"), model=TestModel())

    # TestClient triggers lifespan startup on context manager entry
    with caplog.at_level(logging.WARNING), TestClient(app):
        pass

    assert _has_cors_wildcard_warning(caplog.records), (
        f"Expected CORS wildcard warning in logs. Found: {[r.message for r in caplog.records]}"
    )


def test_cors_specific_origins_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that NO warning is logged when CORS has specific origins.

    Warning should only appear for wildcard "*", not for
    legitimate specific origins.
    """
    app = create_app(
        settings=_build_settings(
            cors_origins="https://example.com,https://app.example.com",
        ),
        model=TestModel(),
    )

    with caplog.at_level(logging.WARNING), TestClient(app):
        pass  # Lifespan runs on context manager entry

    assert not _has_cors_wildcard_warning(caplog.records), (
        "Expected NO CORS wildcard warning for specific origins. "
        f"Found: {[r.message for r in caplog.records]}"
    )


def test_cors_wildcard_in_list_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that warning is logged even when wildcard is in a list with other origins.

    Having "*" anywhere in the origins list is a security risk.
    """
    app = create_app(
        settings=_build_settings(
            cors_origins="https://example.com,*,https://app.example.com",
        ),
        model=TestModel(),
    )

    with caplog.at_level(logging.WARNING), TestClient(app):
        pass  # Lifespan runs on context manager entry

    assert _has_cors_wildcard_warning(caplog.records), (
        f"Expected CORS wildcard warning in logs. Found: {[r.message for r in caplog.records]}"
    )
