"""Unit tests for app/observability.py."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from pydantic import ValidationError

from app.config import Settings
from app.observability import configure_logfire


class TestConfigureLogfire:
    """Tests for configure_logfire() function."""

    @patch("app.observability.logfire.configure")
    @patch("app.observability.logfire.instrument_pydantic_ai")
    def test_configure_logfire_with_token(
        self,
        mock_instrument_pydantic: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """Test configure_logfire() calls logfire.configure() when token is provided."""
        # Arrange
        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-llm-key-12345"),
            logfire_token=SecretStr("test-logfire-token"),
            logfire_service_name="test-service",
        )

        # Act
        configure_logfire(settings)

        # Assert
        mock_configure.assert_called_once_with(
            token="test-logfire-token",  # noqa: S106
            service_name="test-service",
        )
        mock_instrument_pydantic.assert_called_once()

    @patch("app.observability.logfire.configure")
    @patch("app.observability.logfire.instrument_pydantic_ai")
    def test_configure_logfire_without_token(
        self,
        mock_instrument_pydantic: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """Test configure_logfire() skips logfire.configure() when token is None."""
        # Arrange
        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-llm-key-12345"),
            logfire_token=None,
            logfire_service_name="test-service",
        )

        # Act
        configure_logfire(settings)

        # Assert
        mock_configure.assert_not_called()
        mock_instrument_pydantic.assert_called_once()

    def test_configure_logfire_empty_token_raises_validation_error(
        self,
    ) -> None:
        """Test that empty logfire_token raises ValidationError during Settings construction."""
        # Arrange & Act & Assert
        # Task 16.13 added validation that rejects empty or whitespace-only tokens
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("test-llm-key-12345"),
                logfire_token=SecretStr(""),  # Empty string should be rejected
                logfire_service_name="test-service",
            )

        # Verify the error is about logfire_token field
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("logfire_token",) for error in errors)
        # Check that the error message mentions "cannot be empty"
        error_messages = [str(error.get("ctx", {}).get("error", "")) for error in errors]
        assert any("cannot be empty" in msg for msg in error_messages)
