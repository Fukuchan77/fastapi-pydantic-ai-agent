"""Unit tests for build_model provider name case normalization (Task 22.3)."""

from unittest.mock import patch

from pydantic import SecretStr

from app.agents.chat_agent import build_model
from app.config import Settings


class TestBuildModelCaseNormalization:
    """Test suite for Task 22.3: provider name case normalization."""

    def test_build_model_normalizes_uppercase_ollama_provider(self, monkeypatch) -> None:
        """build_model should normalize uppercase 'Ollama' to lowercase 'ollama'.

        Bug: When llm_model="Ollama:granite3.3:latest" (capital O), the provider_name
        is not normalized, so the check `elif provider_name == "ollama"` at line 50
        fails, and the default Ollama base URL is not set.

        Expected: provider_name should be normalized to lowercase after split,
        so "Ollama:model" is treated the same as "ollama:model".
        """
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="Ollama:granite3.3:latest",  # Capital O
        )

        with patch("app.agents.chat_agent.LiteLLMModel") as mock_model:
            build_model(settings)
            # Should normalize to "ollama/granite3.3:latest" and set default base URL
            mock_model.assert_called_once_with(
                model_name="ollama/granite3.3:latest",
                api_key=None,
                settings={"litellm_api_base": "http://localhost:11434"},
            )

    def test_build_model_normalizes_mixed_case_openai_provider(self) -> None:
        """build_model should normalize mixed-case 'OpenAI' to lowercase 'openai'."""
        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="OpenAI:gpt-4o",  # Mixed case
            llm_api_key=SecretStr("test-openai-key-16"),
        )

        with patch("app.agents.chat_agent.LiteLLMModel") as mock_model:
            build_model(settings)
            # Should normalize to "openai/gpt-4o"
            mock_model.assert_called_once_with(
                model_name="openai/gpt-4o",
                api_key="test-openai-key-16",
                settings=None,
            )

    def test_build_model_normalizes_anthropic_uppercase(self) -> None:
        """build_model should normalize 'ANTHROPIC' to lowercase 'anthropic'."""
        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="ANTHROPIC:claude-3-5-sonnet-20241022",  # All uppercase
            llm_api_key=SecretStr("test-anthropic-key"),
        )

        with patch("app.agents.chat_agent.LiteLLMModel") as mock_model:
            build_model(settings)
            # Should normalize to "anthropic/claude-3-5-sonnet-20241022"
            mock_model.assert_called_once_with(
                model_name="anthropic/claude-3-5-sonnet-20241022",
                api_key="test-anthropic-key",
                settings=None,
            )
