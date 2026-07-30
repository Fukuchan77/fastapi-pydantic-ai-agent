"""Unit tests for app/observability.py."""

import logging
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from logfire import ScrubbingOptions
from pydantic import SecretStr
from pydantic import ValidationError

from app.config import Settings
from app.observability import configure_logfire


DEFAULT_SCRUBBING = ScrubbingOptions(extra_patterns=["prompt", "tool_input", "tool_output"])


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
            scrubbing=DEFAULT_SCRUBBING,
        )
        mock_instrument_pydantic.assert_called_once()

    @patch("app.observability.logfire.configure")
    @patch("app.observability.logfire.instrument_pydantic_ai")
    def test_configure_logfire_without_token(
        self,
        mock_instrument_pydantic: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """Test configure_logfire() still enables scrubbing when token is None."""
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
        mock_configure.assert_called_once_with(scrubbing=DEFAULT_SCRUBBING)
        mock_instrument_pydantic.assert_called_once()

    def test_configure_logfire_empty_token_raises_validation_error(
        self,
    ) -> None:
        """Test that empty logfire_token raises ValidationError during Settings construction."""
        # Arrange & Act & Assert
        # added validation that rejects empty or whitespace-only tokens
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


class TestConfigureLogfireScrubbing:
    """Tests for Req 7.1/7.2/7.3: scrubbing default, opt-out, and fail-soft init."""

    def _settings(self, **overrides: object) -> Settings:
        return Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-llm-key-12345"),
            logfire_token=None,
            **overrides,  # type: ignore[arg-type]
        )

    @patch("app.observability.logfire.configure")
    @patch("app.observability.logfire.instrument_pydantic_ai")
    def test_scrubbing_enabled_by_default_covers_prompt_and_tool_payloads(
        self,
        mock_instrument_pydantic: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """Req 7.1: scrubbing is enabled by default with prompt/tool_input/tool_output."""
        settings = self._settings()

        configure_logfire(settings)

        _, kwargs = mock_configure.call_args
        scrubbing = kwargs["scrubbing"]
        assert isinstance(scrubbing, ScrubbingOptions)
        assert set(scrubbing.extra_patterns) == {"prompt", "tool_input", "tool_output"}

    @patch("app.observability.logfire.configure")
    @patch("app.observability.logfire.instrument_pydantic_ai")
    def test_log_sensitive_payloads_disables_scrubbing_and_warns(
        self,
        mock_instrument_pydantic: MagicMock,
        mock_configure: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Req 7.2: log_sensitive_payloads=True disables scrubbing and emits an audit warning."""
        settings = self._settings(log_sensitive_payloads=True)

        with caplog.at_level(logging.WARNING, logger="app.observability"):
            configure_logfire(settings)

        mock_configure.assert_called_once_with(scrubbing=False)
        assert any(
            "log_sensitive_payloads" in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        )

    @patch("app.observability.logfire.instrument_pydantic_ai")
    @patch("app.observability.logfire.configure", side_effect=RuntimeError("boom"))
    def test_initialization_exception_is_swallowed_and_does_not_raise(
        self,
        mock_configure: MagicMock,
        mock_instrument_pydantic: MagicMock,
    ) -> None:
        """Req 7.3: an exception during observability init never blocks app startup."""
        settings = self._settings()

        configure_logfire(settings)  # must not raise

    @patch("app.observability.logfire.configure")
    @patch(
        "app.observability.logfire.instrument_pydantic_ai",
        side_effect=RuntimeError("boom"),
    )
    def test_instrumentation_exception_is_also_swallowed(
        self,
        mock_instrument_pydantic: MagicMock,
        mock_configure: MagicMock,
    ) -> None:
        """Req 7.3: a failure in instrument_pydantic_ai() also must not raise."""
        settings = self._settings()

        configure_logfire(settings)  # must not raise
