"""Unit tests for llm_api_key minimum length validation (Task 16.12)."""

import pytest
from pydantic import SecretStr
from pydantic import ValidationError

from app.config import Settings


class TestLLMAPIKeyMinimumLength:
    """Test llm_api_key minimum length validation."""

    def test_llm_api_key_none_is_allowed(self):
        """llm_api_key=None should be allowed (optional for Ollama)."""
        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="ollama:llama2",
            llm_api_key=None,
        )
        assert settings.llm_api_key is None

    def test_llm_api_key_empty_string_rejected(self):
        """Empty string llm_api_key should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr(""),
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("llm_api_key",)
        assert "empty" in errors[0]["msg"].lower()

    def test_llm_api_key_whitespace_only_rejected(self):
        """Whitespace-only llm_api_key should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("   "),
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("llm_api_key",)
        assert "empty" in errors[0]["msg"].lower()

    def test_llm_api_key_too_short_rejected(self):
        """llm_api_key shorter than 16 characters should be rejected."""
        short_keys = [
            "abc",  # 3 chars
            "test-key",  # 8 chars (also a placeholder, will be caught by placeholder check)
            "sk-short123",  # 12 chars
            "a" * 15,  # 15 chars
        ]

        for short_key in short_keys:
            with pytest.raises(ValidationError) as exc_info:
                Settings(
                    api_key=SecretStr("test-api-key-12345"),
                    llm_model="openai:gpt-4o",
                    llm_api_key=SecretStr(short_key),
                )

            errors = exc_info.value.errors()
            assert len(errors) == 1
            assert errors[0]["loc"] == ("llm_api_key",)
            # Error message will contain either "16 characters" (length check)
            # or "placeholder" (if caught by placeholder check first)
            error_msg = errors[0]["msg"]
            assert "16 characters" in error_msg or "placeholder" in error_msg

    def test_llm_api_key_minimum_16_chars_accepted(self):
        """llm_api_key with exactly 16 characters should be accepted."""
        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("a" * 16),  # Exactly 16 chars
        )
        assert settings.llm_api_key.get_secret_value() == "a" * 16

    def test_llm_api_key_long_key_accepted(self):
        """llm_api_key longer than 16 characters should be accepted."""
        long_keys = [
            "sk-" + "a" * 48,  # OpenAI format (51 chars)
            "b" * 32,  # 32 chars
            "anthropic-key-1234567890",  # 25 chars
        ]

        for long_key in long_keys:
            settings = Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr(long_key),
            )
            assert settings.llm_api_key.get_secret_value() == long_key

    def test_llm_api_key_with_surrounding_whitespace_stripped(self):
        """llm_api_key with surrounding whitespace should be stripped and validated."""
        # Valid key with whitespace should be stripped and accepted
        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("  valid-key-123456  "),  # 18 chars after strip
        )
        assert (
            settings.llm_api_key.get_secret_value() == "  valid-key-123456  "
        )  # Original value preserved

    def test_llm_api_key_placeholder_values_rejected(self):
        """Common placeholder values should be rejected even if 16+ chars."""
        # Test placeholders that are actually in the validator's list
        actual_placeholders = [
            "your-api-key-here",  # 19 chars
            "api-key-here",  # 12 chars
        ]

        for placeholder in actual_placeholders:
            with pytest.raises(ValidationError) as exc_info:
                Settings(
                    api_key=SecretStr("test-api-key-12345"),
                    llm_model="openai:gpt-4o",
                    llm_api_key=SecretStr(placeholder),
                )

            errors = exc_info.value.errors()
            assert len(errors) == 1
            assert errors[0]["loc"] == ("llm_api_key",)
            assert "placeholder" in errors[0]["msg"].lower()
