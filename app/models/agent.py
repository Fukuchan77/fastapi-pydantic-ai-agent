"""Request and response models for agent endpoints."""

from pydantic import BaseModel
from pydantic import Field

from app.agents.guardrails import AuditRecord
from app.agents.guardrails import StopReason


class ChatRequest(BaseModel):
    """Request model for chat endpoint.

    Attributes:
        message: User message to send to the agent (1-32000 chars).
        session_id: Server-issued session ID from a previous response, for
            conversation continuity. If omitted, the server mints a new one
            (Req 11.1). Presenting a session_id bound to another principal
            is rejected with 403 (Req 11.2).
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="User message to send to the agent",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity. If omitted, the "
        "server issues a new one; presenting another principal's session_id "
        "is rejected with 403.",
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint.

    Attributes:
        reply: Agent's response to the user message.
        session_id: The server-issued session ID for this conversation - the
            minted id for a new conversation, or the same id echoed back
            when continuing one (Req 11.1).
        tool_calls_made: Number of tool calls executed during this conversation turn.
    """

    reply: str = Field(description="Agent's response to the user message")
    session_id: str | None = Field(
        description="Server-issued session ID for this conversation (Req 11.1)"
    )
    tool_calls_made: int = Field(
        description="Number of tool calls executed during this conversation turn"
    )
    stop_reason: StopReason = Field(
        default="completed",
        description="Why the guarded agent run stopped (Req 4.3)",
    )
    audit: list[AuditRecord] = Field(
        default_factory=list,
        description="Refused, denied, or budget-blocked tool attempts recorded during "
        "this turn (Req 4.7)",
    )
