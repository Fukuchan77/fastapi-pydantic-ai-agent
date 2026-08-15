"""Agent dependencies - RunContext dependencies for Pydantic AI agents."""

from dataclasses import dataclass
from dataclasses import field

import httpx
from fastapi import Request

from app.agents.guardrails import AuditTrail
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
        principal: Stable identifier of the calling principal, for audit
            attribution (Req 4.7). Populated by `bind_principal()` at each
            agent route's entry point, not by `get_agent_deps()`: resolving
            it inside the dependency would require importing
            `app.deps.auth`, and that import closes a cycle
            (`app.deps` -> `app.deps.workflow` -> `app.workflows.corrective_rag`
            -> `app.agents.chat_agent` -> `app.agents.deps`). Stays `None`
            for any non-agent caller that builds deps directly.
        audit: Sink refused/denied/budget-blocked tool attempts are recorded
            to by `run_guarded()` (Req 4.7).
    """

    http_client: httpx.AsyncClient
    settings: Settings
    session_store: SessionStore
    principal: str | None = None
    audit: AuditTrail = field(default_factory=AuditTrail)


async def get_agent_deps(request: Request) -> AgentDeps:
    """FastAPI dependency factory that constructs AgentDeps from app.state.

    Leaves `principal` unset; agent routes bind it with `bind_principal()`
    once `verify_api_key` has resolved the caller.

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


def bind_principal(deps: AgentDeps, principal_id: str) -> AgentDeps:
    """Attach the authenticated caller's id to a per-request `AgentDeps`.

    Every `AuditRecord` a guarded run produces (Req 4.7) is attributed through
    this field, so an agent route must bind it before running the agent -
    otherwise the audit trail records *what* was refused but never *for whom*,
    which is most of its value once more than one API key exists.

    Mutating in place is safe: `get_agent_deps()` constructs a fresh instance
    per request, so nothing is shared between callers.

    Args:
        deps: The per-request dependencies to bind onto.
        principal_id: `Principal.id` of the authenticated caller.

    Returns:
        AgentDeps: The same instance, with `principal` set.
    """
    deps.principal = principal_id
    return deps
