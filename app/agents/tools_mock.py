"""Mock tools for development and testing only.

⚠️ WARNING: This module should NEVER be imported in production environments.
The tools in this module return stub data and do not perform actual operations.
"""

import logging

from pydantic_ai import Agent
from pydantic_ai import RunContext

from app.agents.deps import AgentDeps


logger = logging.getLogger(__name__)


def register_mock_tools(agent: Agent[AgentDeps, str]) -> None:
    """Register mock tools to the agent for development/testing.

    ⚠️ WARNING: This function should only be called in non-production environments.
    The registered tools return stub data and do not perform actual operations.

    Args:
        agent: The PydanticAI Agent instance to register tools to.
    """

    @agent.tool
    async def mock_web_search(ctx: RunContext[AgentDeps], query: str) -> str:
        """Mock web search tool - placeholder that returns stub data.

        ⚠️ WARNING: This is a MOCK implementation for development only!
        It does NOT perform actual web searches and returns stub data.

        Args:
            ctx: RunContext providing access to AgentDeps (http_client, settings).
            query: The search query string.

        Returns:
            Mock search results as a formatted string (not real search data).
        """
        logger.warning(
            "Mock web search tool called with query: %s. "
            "This returns stub data only. Replace with real search API for production.",
            query[:100],
        )

        # Stub implementation - returns mock data only
        return (
            f"Mock search results for '{query}':\n"
            f"1. Example result about {query}\n"
            f"2. Another relevant link about {query}\n"
            f"3. More information on {query}"
        )

    logger.info("Mock tools registered to agent (development mode)")
