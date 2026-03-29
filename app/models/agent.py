"""Request and response models for agent endpoints."""

from pydantic import BaseModel
from pydantic import Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint.

    Attributes:
        message: User message to send to the agent (1-32000 chars).
        session_id: Optional session ID for conversation continuity.
            If None, a stateless single-turn conversation is performed.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=32_000,
        description="User message to send to the agent",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity. If omitted, stateless.",
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint.

    Attributes:
        reply: Agent's response to the user message.
        session_id: Session ID if session was used, None for stateless conversations.
        tool_calls_made: Number of tool calls executed during this conversation turn.
    """

    reply: str = Field(description="Agent's response to the user message")
    session_id: str | None = Field(
        description="Session ID if session was used, None for stateless conversations"
    )
    tool_calls_made: int = Field(
        description="Number of tool calls executed during this conversation turn"
    )
