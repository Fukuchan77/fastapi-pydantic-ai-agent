"""Pydantic Logfire observability configuration."""

import logfire

from app.config import Settings


def configure_logfire(settings: Settings) -> None:
    """Configure Pydantic Logfire for AI-native observability.

    Initializes Logfire with the provided settings and instruments Pydantic AI
    for automatic tracing of agent runs, tool calls, and token usage.

    Requirements:
        - 8.1: Auto-instrument all Pydantic AI agent runs
        - 8.2: Record token usage and cost metadata
        - 8.3: Emit spans for tool invocations

    Args:
        settings: Application settings containing Logfire configuration
    """
    # Only configure logfire if token is provided and non-empty
    if settings.logfire_token:
        logfire.configure(
            token=settings.logfire_token,
            service_name=settings.logfire_service_name,
        )

    # Always instrument Pydantic AI regardless of token (for local dev)
    logfire.instrument_pydantic_ai()
