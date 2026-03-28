"""Test SecretStr protection for sensitive configuration fields.

Task 16.7: Strengthen secret field protection with SecretStr.
Tests that api_key, llm_api_key, and logfire_token use SecretStr type
and are properly protected from exposure in repr() and logging.
"""

from typing import get_args
from typing import get_origin

import pytest
from pydantic import SecretStr
from pydantic import ValidationError

from app.config import Settings


class TestSecretStrFieldTypes:
    """Test that sensitive fields use SecretStr type."""

    def test_api_key_is_secret_str(self) -> None:
        """Test that api_key field uses SecretStr type."""
        # Get the type annotation for api_key
        field_info = Settings.model_fields["api_key"]
        field_type = field_info.annotation

        # api_key should be SecretStr, not str
        assert field_type == SecretStr, f"api_key should be SecretStr type, got {field_type}"

    def test_llm_api_key_is_optional_secret_str(self) -> None:
        """Test that llm_api_key field uses SecretStr | None type."""
        # Get the type annotation for llm_api_key
        field_info = Settings.model_fields["llm_api_key"]
        field_type = field_info.annotation

        # llm_api_key should be SecretStr | None
        # Check if it's a Union type
        origin = get_origin(field_type)
        if origin is not None:  # It's a generic type (Union)
            args = get_args(field_type)
            assert SecretStr in args, (
                f"llm_api_key should include SecretStr in Union, got {field_type}"
            )
            assert type(None) in args, f"llm_api_key should include None in Union, got {field_type}"
        else:
            pytest.fail(f"llm_api_key should be SecretStr | None, got {field_type}")

    def test_logfire_token_is_optional_secret_str(self) -> None:
        """Test that logfire_token field uses SecretStr | None type."""
        # Get the type annotation for logfire_token
        field_info = Settings.model_fields["logfire_token"]
        field_type = field_info.annotation

        # logfire_token should be SecretStr | None
        origin = get_origin(field_type)
        if origin is not None:  # It's a generic type (Union)
            args = get_args(field_type)
            assert SecretStr in args, (
                f"logfire_token should include SecretStr in Union, got {field_type}"
            )
            assert type(None) in args, (
                f"logfire_token should include None in Union, got {field_type}"
            )
        else:
            pytest.fail(f"logfire_token should be SecretStr | None, got {field_type}")


class TestSecretStrReprProtection:
    """Test that SecretStr values are not exposed in repr()."""

    def test_api_key_not_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that api_key value is not exposed in repr()."""
        # Set up environment with valid values
        monkeypatch.setenv("API_KEY", "test-api-key-secure-16chars")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

        settings = Settings()
        repr_str = repr(settings)

        # The actual secret value should NOT appear in repr
        assert "test-api-key-secure-16chars" not in repr_str, (
            "api_key secret value should not be exposed in repr()"
        )
        # SecretStr should show as '**********' or similar
        assert "api_key" in repr_str or "**********" in repr_str

    def test_llm_api_key_not_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that llm_api_key value is not exposed in repr()."""
        monkeypatch.setenv("API_KEY", "test-api-key-secure-16chars")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test-llm-key-secure-value-123")

        settings = Settings()
        repr_str = repr(settings)

        # The actual secret value should NOT appear in repr
        assert "sk-test-llm-key-secure-value-123" not in repr_str, (
            "llm_api_key secret value should not be exposed in repr()"
        )

    def test_logfire_token_not_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that logfire_token value is not exposed in repr()."""
        monkeypatch.setenv("API_KEY", "test-api-key-secure-16chars")
        monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
        monkeypatch.setenv("LOGFIRE_TOKEN", "logfire-token-secure-value-123")

        settings = Settings()
        repr_str = repr(settings)

        # The actual secret value should NOT appear in repr
        assert "logfire-token-secure-value-123" not in repr_str, (
            "logfire_token secret value should not be exposed in repr()"
        )


class TestSecretStrGetSecretValue:
    """Test that .get_secret_value() retrieves the actual string."""

    def test_api_key_get_secret_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that api_key.get_secret_value() returns the actual string."""
        test_key = "test-api-key-secure-16chars"
        monkeypatch.setenv("API_KEY", test_key)
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

        settings = Settings()

        # Should be able to get the actual value using .get_secret_value()
        assert hasattr(settings.api_key, "get_secret_value"), (
            "api_key should have .get_secret_value() method"
        )
        assert settings.api_key.get_secret_value() == test_key, (
            "api_key.get_secret_value() should return the actual string"
        )

    def test_llm_api_key_get_secret_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that llm_api_key.get_secret_value() returns the actual string."""
        test_key = "sk-test-llm-key-secure-value-123"
        monkeypatch.setenv("API_KEY", "test-api-key-secure-16chars")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", test_key)

        settings = Settings()

        # llm_api_key is optional, so check it's not None first
        assert settings.llm_api_key is not None
        assert hasattr(settings.llm_api_key, "get_secret_value"), (
            "llm_api_key should have .get_secret_value() method"
        )
        assert settings.llm_api_key.get_secret_value() == test_key, (
            "llm_api_key.get_secret_value() should return the actual string"
        )

    def test_logfire_token_get_secret_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that logfire_token.get_secret_value() returns the actual string."""
        test_token = "logfire-token-secure-value-123"  # noqa: S105
        monkeypatch.setenv("API_KEY", "test-api-key-secure-16chars")
        monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
        monkeypatch.setenv("LOGFIRE_TOKEN", test_token)

        settings = Settings()

        # logfire_token is optional, so check it's not None first
        assert settings.logfire_token is not None
        assert hasattr(settings.logfire_token, "get_secret_value"), (
            "logfire_token should have .get_secret_value() method"
        )
        assert settings.logfire_token.get_secret_value() == test_token, (
            "logfire_token.get_secret_value() should return the actual string"
        )


class TestSecretStrEnvironmentLoading:
    """Test that environment variables are correctly loaded into SecretStr."""

    def test_api_key_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that API_KEY environment variable loads into SecretStr."""
        test_key = "env-api-key-secure-value-16chars"
        monkeypatch.setenv("API_KEY", test_key)
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

        settings = Settings()

        # Should load as SecretStr
        assert isinstance(settings.api_key, SecretStr), "api_key should be SecretStr instance"
        assert settings.api_key.get_secret_value() == test_key

    def test_llm_api_key_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that LLM_API_KEY environment variable loads into SecretStr."""
        test_key = "env-llm-key-secure-value-16chars"
        monkeypatch.setenv("API_KEY", "test-api-key-secure-16chars")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", test_key)

        settings = Settings()

        # Should load as SecretStr
        assert isinstance(settings.llm_api_key, SecretStr), (
            "llm_api_key should be SecretStr instance"
        )
        assert settings.llm_api_key.get_secret_value() == test_key

    def test_logfire_token_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that LOGFIRE_TOKEN environment variable loads into SecretStr."""
        test_token = "env-logfire-token-secure-value"  # noqa: S105
        monkeypatch.setenv("API_KEY", "test-api-key-secure-16chars")
        monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
        monkeypatch.setenv("LOGFIRE_TOKEN", test_token)

        settings = Settings()

        # Should load as SecretStr
        assert isinstance(settings.logfire_token, SecretStr), (
            "logfire_token should be SecretStr instance"
        )
        assert settings.logfire_token.get_secret_value() == test_token

    def test_optional_fields_can_be_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that optional SecretStr fields can be None."""
        monkeypatch.setenv("API_KEY", "test-api-key-secure-16chars")
        monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
        # Explicitly unset LLM_API_KEY and LOGFIRE_TOKEN to ensure they're None
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)

        settings = Settings()

        # These optional fields should be None
        assert settings.llm_api_key is None, "llm_api_key should be None when not set"
        assert settings.logfire_token is None, "logfire_token should be None when not set"


class TestSecretStrValidation:
    """Test that validation still works with SecretStr."""

    def test_api_key_min_length_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that api_key minimum length validation works with SecretStr."""
        monkeypatch.setenv("API_KEY", "short")  # Less than 16 characters
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

        # Should raise validation error for length
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_msg = str(exc_info.value)
        assert "at least 16 characters" in error_msg.lower()

    def test_api_key_placeholder_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that api_key placeholder validation works with SecretStr."""
        monkeypatch.setenv("API_KEY", "your-api-key-here")  # Placeholder
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")

        # Should raise validation error for placeholder
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_msg = str(exc_info.value)
        assert "placeholder" in error_msg.lower()

    def test_llm_api_key_min_length_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that llm_api_key minimum length validation works with SecretStr."""
        monkeypatch.setenv("API_KEY", "test-api-key-secure-16chars")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "short")  # Less than 16 characters

        # Should raise validation error for length
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_msg = str(exc_info.value)
        assert "at least 16 characters" in error_msg.lower()

    def test_logfire_token_min_length_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that logfire_token minimum length validation works with SecretStr."""
        monkeypatch.setenv("API_KEY", "test-api-key-secure-16chars")
        monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
        monkeypatch.setenv("LOGFIRE_TOKEN", "short")  # Less than 16 characters

        # Should raise validation error for length
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_msg = str(exc_info.value)
        assert "at least 16 characters" in error_msg.lower()
