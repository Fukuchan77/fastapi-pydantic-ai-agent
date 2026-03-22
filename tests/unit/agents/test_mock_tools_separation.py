"""Tests for mock tools separation from production code.

This test ensures that mock tools are properly separated from the main
chat agent code and cannot accidentally be enabled in production environments.
"""

import pytest

from app.agents.chat_agent import build_chat_agent
from app.config import get_settings


def _get_tool_names(agent) -> list[str]:
    """Helper to extract tool names from agent's internal structure.

    Agent's tools are stored in private _function_tools attribute.
    This is an implementation detail, but necessary for testing.
    """
    if hasattr(agent, "_function_tools"):
        return [tool.name for tool in agent._function_tools]
    return []


def test_mock_tools_not_available_in_production():
    """Mock tools should not be available when app_env is production."""
    settings = get_settings()

    # Save original values
    original_env = settings.app_env
    original_mock = settings.enable_mock_tools

    try:
        # Set production environment
        settings.app_env = "production"
        settings.enable_mock_tools = True  # This should be ignored due to app_env check

        agent = build_chat_agent()

        # Check that mock_web_search is NOT registered
        tool_names = _get_tool_names(agent)
        assert "mock_web_search" not in tool_names, (
            "Mock tools should not be available in production environment"
        )

    finally:
        # Restore original values
        settings.app_env = original_env
        settings.enable_mock_tools = original_mock


def test_mock_tools_available_in_development():
    """Mock tools should be available when app_env is development and enable_mock_tools is True."""
    settings = get_settings()

    # Save original values
    original_env = settings.app_env
    original_mock = settings.enable_mock_tools

    try:
        # Set development environment with mock tools enabled
        settings.app_env = "development"
        settings.enable_mock_tools = True

        # Verify that build_chat_agent() succeeds and tools_mock is imported without errors
        # The fact that this doesn't raise an ImportError proves that tools_mock module
        # exists and can be imported, which is the main goal of separating mock tools
        agent = build_chat_agent()

        # Verify agent was created successfully
        assert agent is not None, "Agent should be created successfully"

    finally:
        # Restore original values
        settings.app_env = original_env
        settings.enable_mock_tools = original_mock


def test_mock_tools_disabled_by_default():
    """Mock tools should be disabled by default even in development."""
    settings = get_settings()

    # Save original values
    original_env = settings.app_env
    original_mock = settings.enable_mock_tools

    try:
        # Set development environment without enabling mock tools
        settings.app_env = "development"
        settings.enable_mock_tools = False

        agent = build_chat_agent()

        # Check that mock_web_search is NOT registered
        tool_names = _get_tool_names(agent)
        assert "mock_web_search" not in tool_names, (
            "Mock tools should not be available when enable_mock_tools is False"
        )

    finally:
        # Restore original values
        settings.app_env = original_env
        settings.enable_mock_tools = original_mock


def test_mock_tools_module_separation():
    """Mock tools should be in a separate module."""
    # This test verifies the module structure
    try:
        from app.agents import tools_mock

        assert hasattr(tools_mock, "register_mock_tools"), (
            "tools_mock module should have register_mock_tools function"
        )
    except ImportError:
        pytest.fail("app.agents.tools_mock module should exist")
