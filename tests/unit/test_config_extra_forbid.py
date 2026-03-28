"""Tests for Settings extra='forbid' validation.

Task 16.11: Verify Settings rejects unknown environment variables.
"""

import pytest
from pydantic import HttpUrl
from pydantic import SecretStr
from pydantic import ValidationError

from app.config import Settings
from app.config import get_settings


class TestSettingsExtraForbid:
    """Test Settings rejects extra environment variables."""

    def test_rejects_unknown_field_in_constructor(self):
        """Settings should reject unknown fields passed directly with extra='forbid'."""
        # Should raise ValidationError when passing extra fields to constructor
        with pytest.raises(
            ValidationError,
            match=r"Extra inputs are not permitted|extra fields not permitted",
        ):
            Settings(
                api_key=SecretStr("secure-test-key-1234567890"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("sk-test-key-1234567890"),
                unknown_field="some-value",  # Extra field not defined in Settings
            )

    def test_rejects_typo_in_field_name(self):
        """Settings should reject typos in field names when passed to constructor."""
        # Should raise ValidationError when field name is misspelled
        with pytest.raises(
            ValidationError,
            match=r"Extra inputs are not permitted|extra fields not permitted",
        ):
            Settings(
                api_key=SecretStr("secure-test-key-1234567890"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("sk-test-key-1234567890"),
                llm_base_ulr="http://localhost:11434",  # Typo: ULR instead of URL
            )

    def test_accepts_all_defined_fields_via_constructor(self):
        """Settings should accept all properly named fields via constructor."""
        # Should not raise - all fields are defined in Settings
        settings = Settings(
            api_key=SecretStr("secure-test-key-1234567890"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("sk-test-key-1234567890"),
            llm_base_url=HttpUrl("http://localhost:11434"),
            max_output_retries=5,
            app_env="development",
            enable_mock_tools=True,
            http_timeout=60.0,
            http_connect_timeout=10.0,
            llm_retry_max_attempts=5,
            llm_retry_base_delay=2.0,
            cors_origins=["http://localhost:3000"],
            logfire_token=SecretStr("lf_test_token_1234567890"),
            logfire_service_name="test-service",
        )
        assert settings.api_key.get_secret_value() == "secure-test-key-1234567890"
        assert settings.llm_model == "openai:gpt-4o"

    def test_rejects_multiple_unknown_fields(self):
        """Settings should reject when multiple unknown fields are provided."""
        # Should raise ValidationError
        with pytest.raises(
            ValidationError,
            match=r"Extra inputs are not permitted|extra fields not permitted",
        ):
            Settings(
                api_key=SecretStr("secure-test-key-1234567890"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("sk-test-key-1234567890"),
                random_field_1="value1",  # Unknown field
                random_field_2="value2",  # Unknown field
                another_unknown="value3",  # Unknown field
            )

    def test_environment_variables_load_correctly(self, monkeypatch):
        """Environment variables should still load correctly with extra='forbid'.

        Note: pydantic-settings only reads environment variables for defined fields,
        so extra environment variables are never passed to the model validation.
        This is expected behavior - extra='forbid' applies to model initialization,
        not to which environment variables exist in the environment.
        """
        # Clear cache to ensure fresh Settings instance
        get_settings.cache_clear()

        # Set environment variables
        monkeypatch.setenv("API_KEY", "secure-test-key-1234567890")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test-key-1234567890")

        # Set an unknown environment variable (should be ignored by pydantic-settings)
        monkeypatch.setenv("UNKNOWN_VARIABLE", "some-value")

        # Should work fine - pydantic-settings only reads defined fields
        settings = get_settings()
        assert settings.api_key.get_secret_value() == "secure-test-key-1234567890"
        assert settings.llm_model == "openai:gpt-4o"

        # Clean up
        get_settings.cache_clear()
