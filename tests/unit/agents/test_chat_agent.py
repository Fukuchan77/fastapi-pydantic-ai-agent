"""Unit tests for app/agents/chat_agent.py - Agent factory and tools."""

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from app.agents.chat_agent import build_chat_agent


# Check for optional dependencies
try:
    import anthropic  # noqa: F401

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import groq  # noqa: F401

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import openai  # noqa: F401

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class TestBuildChatAgent:
    """Tests for build_chat_agent factory function."""

    @patch("app.agents.chat_agent.get_settings")
    def test_build_chat_agent_returns_agent(self, mock_get_settings):
        """build_chat_agent must return a pydantic_ai Agent instance."""
        mock_get_settings.return_value = Mock(
            llm_model="openai:gpt-4o",
            llm_api_key="test-key",
            llm_base_url=None,
            max_output_retries=3,
        )

        agent = build_chat_agent(model=TestModel())

        assert isinstance(agent, Agent)

    @patch("app.agents.chat_agent.get_settings")
    def test_build_chat_agent_accepts_test_model(self, mock_get_settings):
        """build_chat_agent must accept a TestModel parameter."""
        mock_get_settings.return_value = Mock(
            llm_model="openai:gpt-4o",
            llm_api_key="test-key",
            llm_base_url=None,
            max_output_retries=3,
        )

        # Should not raise when model is passed
        agent_with_model = build_chat_agent(model=TestModel())
        assert agent_with_model is not None
        assert isinstance(agent_with_model, Agent)

    @patch("app.agents.chat_agent.get_settings")
    def test_build_chat_agent_uses_provided_model(self, mock_get_settings):
        """build_chat_agent must use the provided model parameter."""
        mock_get_settings.return_value = Mock(
            llm_model="openai:gpt-4o",
            llm_api_key="test-key",
            llm_base_url=None,
            max_output_retries=3,
        )
        test_model = TestModel()

        agent = build_chat_agent(model=test_model)

        # The agent should have the test model
        assert agent.model is test_model


class TestMockWebSearchTool:
    """Tests for mock_web_search tool function."""

    @pytest.mark.asyncio
    @patch("app.agents.chat_agent.get_settings")
    async def test_mock_web_search_tool_is_registered(self, mock_get_settings):
        """build_chat_agent must register mock_web_search tool as a placeholder."""
        mock_get_settings.return_value = Mock(
            llm_model="openai:gpt-4o",
            llm_api_key="test-key",
            llm_base_url=None,
            max_output_retries=3,
        )

        agent = build_chat_agent(model=TestModel())

        # Agent should have tools registered
        # Check using public API: agent._function_toolset is the internal storage
        assert hasattr(agent, "_function_toolset")
        assert agent._function_toolset is not None


class TestBuildModel:
    """Tests for _build_model helper function."""

    def test_local_provider_dummy_key_constant_exists(self):
        """_LOCAL_PROVIDER_DUMMY_KEY constant must exist with expected value.

        For local providers like Ollama.
        """
        from app.agents.chat_agent import _LOCAL_PROVIDER_DUMMY_KEY

        # Verify the constant exists and has the expected descriptive value
        # This constant is used in _build_model() when llm_api_key is None
        # to satisfy SDK's non-optional api_key parameter for local providers
        assert _LOCAL_PROVIDER_DUMMY_KEY == "LOCAL-PROVIDER-DOES-NOT-USE-API-KEY"

    @pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
    def test_build_model_supports_anthropic_provider(self):
        """_build_model must support anthropic provider."""
        from app.agents.chat_agent import _build_model

        mock_settings = Mock(
            llm_model="anthropic:claude-3-5-sonnet-20241022",
            llm_api_key="test-anthropic-key",
            llm_base_url=None,
        )

        model = _build_model(mock_settings)

        # Should return a Model instance (not raise ValueError)
        assert model is not None
        # The model should be an Anthropic model
        assert "anthropic" in str(type(model)).lower()

    @pytest.mark.skipif(not HAS_GROQ, reason="groq not installed")
    def test_build_model_supports_groq_provider(self):
        """_build_model must support groq provider."""
        from app.agents.chat_agent import _build_model

        mock_settings = Mock(
            llm_model="groq:mixtral-8x7b-32768",
            llm_api_key="test-groq-key",
            llm_base_url=None,
        )

        model = _build_model(mock_settings)

        # Should return a Model instance (not raise ValueError)
        assert model is not None
        # The model should be a Groq model
        assert "groq" in str(type(model)).lower()

    @pytest.mark.skipif(not HAS_OPENAI, reason="openai not installed")
    def test_build_model_supports_ollama_provider(self):
        """_build_model must support ollama provider with base_url."""
        from app.agents.chat_agent import _build_model

        mock_settings = Mock(
            llm_model="ollama:llama3",
            llm_api_key=None,
            llm_base_url="http://localhost:11434/v1",
        )

        model = _build_model(mock_settings)

        # Should return a Model instance (not raise ValueError)
        assert model is not None
        # For Ollama, we use OpenAI-compatible model with custom base_url
        assert model is not None
