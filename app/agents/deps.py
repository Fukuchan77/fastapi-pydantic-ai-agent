"""Agent dependencies - RunContext dependencies for Pydantic AI agents."""

from dataclasses import dataclass

import httpx
from fastapi import Request

from app.config import Settings
from app.stores.session_store import SessionStore


@dataclass
class AgentDeps:
    """Dependencies injected into agent tools via RunContext[AgentDeps].

    This dataclass is the generic type parameter for RunContext in all
    agent tool functions, providing access to shared resources.

    Attributes:
        http_client: Shared async HTTP client for external API calls.
        settings: Application configuration settings.
        session_store: Session history persistence backend.
    """

    http_client: httpx.AsyncClient
    settings: Settings
    session_store: SessionStore


async def get_agent_deps(request: Request) -> AgentDeps:
    """FastAPI dependency factory that constructs AgentDeps from app.state.

    Args:
        request: The FastAPI request object with app.state populated by lifespan.

    Returns:
        AgentDeps instance with shared resources from app.state.
    """
    return AgentDeps(
        http_client=request.app.state.http_client,
        settings=request.app.state.settings,
        session_store=request.app.state.session_store,
    )
