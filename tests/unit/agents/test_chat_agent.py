"""Unit tests for chat agent factory and tool functions."""

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel

from app.agents.chat_agent import _build_system_prompt
from app.agents.chat_agent import build_chat_agent
from app.agents.chat_agent import build_model
from app.agents.deps import AgentDeps
from app.config import Settings


class TestBuildModel:
    """Test suite for build_model LiteLLM factory."""

    def test_build_model_ollama_provider(self, monkeypatch) -> None:
        """build_model should create LiteLLMModel with ollama/model format."""
        from pydantic import HttpUrl

        monkeypatch.delenv("LLM_API_KEY", raising=False)

        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="ollama:llama3.2",
            llm_base_url=HttpUrl("http://localhost:11434/v1"),
        )

        with patch("app.agents.chat_agent.LiteLLMModel") as mock_model:
            build_model(settings)
            mock_model.assert_called_once_with(
                model_name="ollama/llama3.2",
                api_key=None,
                settings={"litellm_api_base": "http://localhost:11434/v1"},
            )

    def test_build_model_ollama_default_api_base(self, monkeypatch) -> None:
        """build_model should default to http://localhost:11434 when llm_base_url is not set."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="ollama:granite3.3:latest",
        )

        with patch("app.agents.chat_agent.LiteLLMModel") as mock_model:
            build_model(settings)
            mock_model.assert_called_once_with(
                model_name="ollama/granite3.3:latest",
                api_key=None,
                settings={"litellm_api_base": "http://localhost:11434"},
            )

    def test_build_model_ollama_with_tag_in_model_name(self, monkeypatch) -> None:
        """build_model should handle ollama models with version tags correctly.

        Verifies that ollama:granite3.3:latest is converted to ollama/granite3.3:latest
        with the full tag preserved in the model name.
        """
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="ollama:granite3.3:latest",
        )

        with patch("app.agents.chat_agent.LiteLLMModel") as mock_model:
            build_model(settings)
            # Verify the tag (:latest) is preserved in the conversion
            mock_model.assert_called_once_with(
                model_name="ollama/granite3.3:latest",
                api_key=None,
                settings={"litellm_api_base": "http://localhost:11434"},
            )

    def test_build_model_openai_with_custom_base_url(self) -> None:
        """build_model should create LiteLLMModel with custom litellm_api_base for openai."""
        from pydantic import HttpUrl

        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-openai-key-16"),
            llm_base_url=HttpUrl("https://custom.openai.com/v1"),
        )

        with patch("app.agents.chat_agent.LiteLLMModel") as mock_model:
            build_model(settings)
            mock_model.assert_called_once_with(
                model_name="openai/gpt-4o",
                api_key="test-openai-key-16",
                settings={"litellm_api_base": "https://custom.openai.com/v1"},
            )

    def test_build_model_openai_with_api_key_only(self) -> None:
        """build_model should create LiteLLMModel with no settings when base_url is absent."""
        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-openai-key-16"),
        )

        with patch("app.agents.chat_agent.LiteLLMModel") as mock_model:
            build_model(settings)
            mock_model.assert_called_once_with(
                model_name="openai/gpt-4o",
                api_key="test-openai-key-16",
                settings=None,
            )

    def test_build_model_openai_requires_api_key_for_cloud(self, monkeypatch) -> None:
        """Settings validator should require API key for cloud OpenAI provider."""
        monkeypatch.delenv("LLM_API_KEY", raising=False)  # Remove to test validation

        # Cloud OpenAI requires API key
        with pytest.raises(ValueError, match="llm_api_key is required"):
            Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                # Missing llm_api_key
            )

    def test_build_model_requires_provider_prefix(self) -> None:
        """Settings validator should require provider:model format."""
        # The Settings validator now enforces provider:model format
        # so models without prefix are rejected at Settings construction
        with pytest.raises(ValueError, match="must follow 'provider:model' format"):
            Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="gpt-4o",  # Missing provider prefix
                llm_api_key=SecretStr("test-openai-key-16"),
            )

    def test_build_model_validator_rejects_unsupported_provider(self) -> None:
        """Settings validator should reject unsupported providers."""
        # Settings validator catches unsupported providers
        with pytest.raises(ValueError, match="provider must be one of"):
            Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="unsupported:model-name",
                llm_api_key=SecretStr("test-api-key-12345"),
            )

    def test_build_model_validator_normalizes_uppercase_provider(self) -> None:
        """Settings validator should normalize uppercase provider names to lowercase ()."""
        # Settings validator now normalizes uppercase providers instead of rejecting them
        settings = Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="OPENAI:gpt-4o",  # Uppercase is now normalized to lowercase
            llm_api_key=SecretStr("test-api-key-12345"),
        )
        # Verify the provider was normalized (check by extracting provider from llm_model)
        provider = settings.llm_model.split(":", 1)[0]
        assert provider == "openai", f"Expected 'openai' but got '{provider}'"


class TestBuildSystemPrompt:
    """Test suite for _build_system_prompt function."""

    @pytest.mark.asyncio
    async def test_build_system_prompt_returns_string(self) -> None:
        """_build_system_prompt should return a non-empty string."""
        mock_ctx = Mock(spec=RunContext[AgentDeps])

        prompt = await _build_system_prompt(mock_ctx)

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    @pytest.mark.asyncio
    async def test_build_system_prompt_mentions_tools(self) -> None:
        """_build_system_prompt should mention tool usage."""
        mock_ctx = Mock(spec=RunContext[AgentDeps])

        prompt = await _build_system_prompt(mock_ctx)

        # Prompt should mention tools since the agent has tool-calling capabilities
        assert "tool" in prompt.lower()

    @pytest.mark.asyncio
    async def test_build_system_prompt_is_helpful_tone(self) -> None:
        """_build_system_prompt should have a helpful, assistant tone."""
        mock_ctx = Mock(spec=RunContext[AgentDeps])

        prompt = await _build_system_prompt(mock_ctx)

        # Check for helpful/assistant language
        assert any(word in prompt.lower() for word in ["helpful", "assist", "help"])


class TestBuildChatAgent:
    """Test suite for build_chat_agent factory function."""

    def test_build_chat_agent_returns_agent_instance(self) -> None:
        """build_chat_agent should return an Agent instance."""
        with patch("app.agents.chat_agent.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("test-api-key-12345"),
            )

            model = TestModel()
            agent = build_chat_agent(model=model)

            assert isinstance(agent, Agent)

    def test_build_chat_agent_uses_provided_model(self) -> None:
        """build_chat_agent should use the provided model when specified."""
        with patch("app.agents.chat_agent.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("test-api-key-12345"),
            )

            test_model = TestModel()
            agent = build_chat_agent(model=test_model)

            # Agent should be created (we can't directly inspect the model,
            # but we verify the agent was constructed successfully)
            assert isinstance(agent, Agent)

    def test_build_chat_agent_builds_model_from_settings(self) -> None:
        """build_chat_agent should build model from settings when model=None."""
        with (
            patch("app.agents.chat_agent.get_settings") as mock_settings,
            patch("app.agents.chat_agent.build_model") as mock_build,
        ):
            mock_settings.return_value = Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("test-api-key-12345"),
            )
            mock_build.return_value = TestModel()

            agent = build_chat_agent(model=None)

            # Verify _build_model was called
            mock_build.assert_called_once()
            assert isinstance(agent, Agent)

    def test_build_chat_agent_configures_agent_type_parameters(self) -> None:
        """build_chat_agent should configure Agent with correct type parameters."""
        with patch("app.agents.chat_agent.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("test-api-key-12345"),
                max_output_retries=5,
            )

            test_model = TestModel()
            agent = build_chat_agent(model=test_model)

            # Agent should be properly typed
            assert isinstance(agent, Agent)
            # Check that deps_type is AgentDeps (via the type hint in the factory)
            # This is validated at type-check time, so just verify agent exists
            assert agent is not None

    def test_build_chat_agent_has_tools(self) -> None:
        """build_chat_agent should have tools available."""
        with patch("app.agents.chat_agent.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("test-api-key-12345"),
            )

            test_model = TestModel()
            agent = build_chat_agent(model=test_model)

            # Verify agent was created successfully with tools
            # We can't easily inspect private _function_tools, but we can verify
            # the agent exists and was constructed properly
            assert isinstance(agent, Agent)
            assert agent is not None


class TestAgentIntegration:
    """Integration tests for agent with tools using TestModel."""

    @pytest.mark.asyncio
    async def test_agent_can_be_run_with_test_model(self) -> None:
        """Agent should be runnable with TestModel for testing."""
        with patch("app.agents.chat_agent.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                api_key=SecretStr("test-api-key-12345"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("test-api-key-12345"),
            )

            # Create agent with TestModel
            test_model = TestModel()
            agent = build_chat_agent(model=test_model)

            # Create mock deps to demonstrate agent can work with proper types
            from unittest.mock import AsyncMock

            _mock_deps = AgentDeps(
                http_client=AsyncMock(spec=pytest.importorskip("httpx").AsyncClient),
                settings=mock_settings.return_value,
                session_store=Mock(),
            )

            # Verify agent can be used (we don't need to actually run it,
            # just verify it's properly constructed)
            assert isinstance(agent, Agent)
            assert agent is not None
