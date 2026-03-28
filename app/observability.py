"""Pydantic Logfire observability configuration."""

import logfire

from app.config import Settings


def configure_logfire(settings: Settings) -> None:
    """Configure Pydantic Logfire for AI-native observability.

    Initializes Logfire with the provided settings and instruments Pydantic AI
    for automatic tracing of agent runs, tool calls, and token usage.

    When logfire_token is provided, Logfire is configured with the token and
    service name for remote logging. When logfire_token is None, Pydantic AI
    instrumentation is still enabled for local development (traces are emitted
    but not sent to Logfire cloud).

    Requirements:
        - 8.1: Auto-instrument all Pydantic AI agent runs
        - 8.2: Record token usage and cost metadata
        - 8.3: Emit spans for tool invocations

    Args:
        settings: Application settings containing Logfire configuration
            (logfire_token and logfire_service_name)

    Example:
        >>> from app.config import get_settings
        >>> settings = get_settings()
        >>> configure_logfire(settings)
        # Pydantic AI is now instrumented for observability
    """
    # Only configure logfire if token is provided and non-empty
    # Task 16.7: Extract secret value from SecretStr
    if settings.logfire_token:
        logfire.configure(
            token=settings.logfire_token.get_secret_value(),
            service_name=settings.logfire_service_name,
        )

    # Always instrument Pydantic AI regardless of token (for local dev)
    logfire.instrument_pydantic_ai()
