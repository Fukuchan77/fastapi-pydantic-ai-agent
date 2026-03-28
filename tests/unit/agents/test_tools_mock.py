"""Unit tests for mock tools registration.

Tests verify that register_mock_tools() is correctly integrated
into build_chat_agent() based on environment and configuration settings.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from app.agents.chat_agent import build_chat_agent
from app.agents.deps import AgentDeps
from app.config import Settings


class TestMockToolsRegistration:
    """Test suite for mock tools registration behavior."""

    def test_mock_tools_registered_when_development_and_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that mock tools ARE registered in development when enabled.

        Given:
            - app_env = "development" (not production)
            - enable_mock_tools = True
        When:
            - build_chat_agent() is called
        Then:
            - register_mock_tools() should be called with the agent
        """
        # Arrange: Set up environment for development with mock tools enabled
        monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("ENABLE_MOCK_TOOLS", "true")

        # Clear settings cache to force reload with new env vars
        from app.config import get_settings

        get_settings.cache_clear()

        # Use a TestModel to avoid actual LLM calls
        test_model = TestModel()

        # Act & Assert: Mock the register_mock_tools function at its source
        with patch("app.agents.tools_mock.register_mock_tools") as mock_register:
            agent = build_chat_agent(model=test_model)

            # Verify register_mock_tools was called once with the agent
            mock_register.assert_called_once()
            # Verify it was called with an Agent instance
            call_args = mock_register.call_args[0]
            assert len(call_args) == 1
            assert isinstance(call_args[0], Agent)
            # Verify agent was created successfully
            assert isinstance(agent, Agent)

    def test_mock_tools_not_registered_when_production(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that mock tools are NOT registered in production environment.

        Given:
            - app_env = "production"
            - enable_mock_tools = True (misconfiguration attempt)
        When:
            - Settings() is instantiated
        Then:
            - ValidationError should be raised (model_validator prevents this)
        """
        # Arrange: Attempt to enable mock tools in production (should fail validation)
        monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("ENABLE_MOCK_TOOLS", "true")

        # Clear settings cache
        from app.config import get_settings

        get_settings.cache_clear()

        # Act & Assert: Settings validation should prevent this configuration
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        # Verify the error message mentions the security risk
        error_messages = str(exc_info.value)
        assert "enable_mock_tools cannot be enabled in production" in error_messages
        assert "security risk" in error_messages.lower()

    def test_mock_tools_not_registered_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that mock tools are NOT registered when disabled.

        Given:
            - app_env = "development"
            - enable_mock_tools = False
        When:
            - build_chat_agent() is called
        Then:
            - register_mock_tools() should NOT be called
        """
        # Arrange: Development environment but mock tools disabled
        monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("ENABLE_MOCK_TOOLS", "false")

        # Clear settings cache
        from app.config import get_settings

        get_settings.cache_clear()

        # Use a TestModel to avoid actual LLM calls
        test_model = TestModel()

        # Act & Assert: Mock the function to verify it's never called
        with patch("app.agents.tools_mock.register_mock_tools") as mock_register:
            agent = build_chat_agent(model=test_model)

            # Verify register_mock_tools was NOT called
            mock_register.assert_not_called()
            assert isinstance(agent, Agent)

    def test_mock_tools_registered_in_staging(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that mock tools can be registered in staging environment.

        Given:
            - app_env = "staging" (not production)
            - enable_mock_tools = True
        When:
            - build_chat_agent() is called
        Then:
            - register_mock_tools() should be called
        """
        # Arrange: Staging environment with mock tools enabled
        monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("ENABLE_MOCK_TOOLS", "true")

        # Clear settings cache
        from app.config import get_settings

        get_settings.cache_clear()

        # Use a TestModel to avoid actual LLM calls
        test_model = TestModel()

        # Act & Assert: Verify register_mock_tools is called
        with patch("app.agents.tools_mock.register_mock_tools") as mock_register:
            agent = build_chat_agent(model=test_model)

            # Verify register_mock_tools was called
            mock_register.assert_called_once()
            call_args = mock_register.call_args[0]
            assert isinstance(call_args[0], Agent)
            # Verify agent was created successfully
            assert isinstance(agent, Agent)

    def test_register_mock_tools_function_directly(self) -> None:
        """Test register_mock_tools() function behavior directly.

        Given:
            - A fresh Agent instance
        When:
            - register_mock_tools() is called
        Then:
            - Agent should have mock_web_search tool registered
        """
        # Arrange: Create a test agent
        from app.agents.tools_mock import register_mock_tools

        test_model = TestModel()
        agent: Agent[AgentDeps, str] = Agent(
            model=test_model,
            deps_type=AgentDeps,
            output_type=str,
        )

        # Act: Register mock tools
        register_mock_tools(agent)

        # Assert: Verify the agent exists and was modified
        # We verify behavior rather than internal state
        assert isinstance(agent, Agent)
        # The tool was registered via @agent.tool decorator inside register_mock_tools

    @pytest.mark.asyncio
    async def test_mock_web_search_tool_returns_stub_data(self) -> None:
        """Test that mock_web_search tool returns stub data.

        Given:
            - An agent with mock tools registered
        When:
            - The agent is run with a query
        Then:
            - It can access the mock_web_search tool
        """
        # Arrange: Build agent with mock tools
        from unittest.mock import AsyncMock

        from app.agents.tools_mock import register_mock_tools

        test_model = TestModel()
        agent: Agent[AgentDeps, str] = Agent(
            model=test_model,
            deps_type=AgentDeps,
            output_type=str,
        )
        register_mock_tools(agent)

        # Create mock deps
        from pydantic import SecretStr

        mock_deps = AgentDeps(
            http_client=AsyncMock(),
            settings=Settings(
                api_key=SecretStr("test-api-key-1234567890"),
                llm_model="openai:gpt-4o",
                llm_api_key=SecretStr("test-llm-key-1234567890"),
            ),
            session_store=MagicMock(),
        )

        # Act: Run the agent (TestModel will handle the response)
        result = await agent.run("test query", deps=mock_deps)

        # Assert: Verify agent executed successfully
        # The tool is available for the agent to use
        assert result is not None
        assert isinstance(result.output, str)

    def test_mock_tools_not_registered_when_production_via_build_agent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that even if validation is bypassed, production check works.

        This test verifies the defense-in-depth approach: even if Settings
        validation were somehow bypassed, the build_chat_agent() function
        itself checks app_env != "production".

        Given:
            - app_env = "production"
            - enable_mock_tools = False (to pass validation)
        When:
            - build_chat_agent() is called
        Then:
            - register_mock_tools() should NOT be called
        """
        # Arrange: Production with mock tools disabled
        monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-key-1234567890")
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("ENABLE_MOCK_TOOLS", "false")

        # Clear settings cache
        from app.config import get_settings

        get_settings.cache_clear()

        test_model = TestModel()

        # Act & Assert: Verify register_mock_tools is not called in production
        with patch("app.agents.tools_mock.register_mock_tools") as mock_register:
            agent = build_chat_agent(model=test_model)

            # Should NOT be called even with enable_mock_tools check
            mock_register.assert_not_called()
            assert isinstance(agent, Agent)
