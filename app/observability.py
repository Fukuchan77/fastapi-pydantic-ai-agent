"""Pydantic Logfire observability configuration."""

import logging
from typing import Any
from typing import Literal

import logfire
from logfire import ScrubbingOptions

from app.config import Settings


logger = logging.getLogger(__name__)

_SCRUBBED_PATTERNS = ["prompt", "tool_input", "tool_output"]


def configure_logfire(settings: Settings) -> None:
    """Configure Pydantic Logfire for AI-native observability.

    Initializes Logfire with the provided settings and instruments Pydantic AI
    for automatic tracing of agent runs, tool calls, and token usage.

    When logfire_token is provided, Logfire is configured with the token and
    service name for remote logging. When logfire_token is None, Pydantic AI
    instrumentation is still enabled for local development (traces are emitted
    but not sent to Logfire cloud).

    Scrubbing of `prompt`/`tool_input`/`tool_output` payloads is enabled by
    default (Req 7.1). Setting `log_sensitive_payloads=True` disables it and
    emits an audit warning (Req 7.2). Any failure during initialization is
    caught and logged so it never blocks application startup (Req 7.3).

    Requirements:
        - 7.1: Enable Logfire scrubbing for prompt/tool payloads by default
        - 7.2: Emit an audit warning when scrubbing is disabled
        - 7.3: Never let an observability init failure block startup
        - 8.1: Auto-instrument all Pydantic AI agent runs
        - 8.2: Record token usage and cost metadata
        - 8.3: Emit spans for tool invocations

    Args:
        settings: Application settings containing Logfire configuration
            (logfire_token, logfire_service_name, log_sensitive_payloads)

    Example:
        >>> from app.config import get_settings
        >>> settings = get_settings()
        >>> configure_logfire(settings)
        # Pydantic AI is now instrumented for observability
    """
    try:
        scrubbing: ScrubbingOptions | Literal[False]
        if settings.log_sensitive_payloads:
            scrubbing = False
            logger.warning(
                "AUDIT: log_sensitive_payloads is enabled - Logfire scrubbing of "
                "prompt/tool_input/tool_output payloads is disabled."
            )
        else:
            scrubbing = ScrubbingOptions(extra_patterns=_SCRUBBED_PATTERNS)

        configure_kwargs: dict[str, Any] = {"scrubbing": scrubbing}
        if settings.logfire_token:
            configure_kwargs["token"] = settings.logfire_token.get_secret_value()
            configure_kwargs["service_name"] = settings.logfire_service_name

        logfire.configure(**configure_kwargs)

        # Always instrument Pydantic AI regardless of token (for local dev)
        logfire.instrument_pydantic_ai()
    except Exception:
        logger.exception("Failed to initialize Logfire observability; continuing without it")
