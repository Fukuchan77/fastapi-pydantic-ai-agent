"""Unit tests for app/observability.py."""

from unittest.mock import MagicMock
from unittest.mock import patch

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
            api_key="test-key",
            llm_model="openai:gpt-4o",
            llm_api_key="test-llm-key",
            logfire_token="test-logfire-token",  # noqa: S106
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
            api_key="test-key",
            llm_model="openai:gpt-4o",
            llm_api_key="test-llm-key",
            logfire_token=None,
            logfire_service_name="test-service",
        )

        # Act
        configure_logfire(settings)

        # Assert
        mock_configure.assert_not_called()
        mock_instrument_pydantic.assert_called_once()

    @patch("app.observability.logfire.configure")
    @patch("app.observability.logfire.instrument_pydantic_ai")
    def test_configure_logfire_empty_token(
        self,
        mock_instrument_pydantic: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """Test configure_logfire() skips logfire.configure() when token is empty string."""
        # Arrange
        settings = Settings(
            api_key="test-key",
            llm_model="openai:gpt-4o",
            llm_api_key="test-llm-key",
            logfire_token="",
            logfire_service_name="test-service",
        )

        # Act
        configure_logfire(settings)

        # Assert
        mock_configure.assert_not_called()
        mock_instrument_pydantic.assert_called_once()
