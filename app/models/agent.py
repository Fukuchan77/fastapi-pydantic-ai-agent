"""Request and response models for agent endpoints."""

from pydantic import BaseModel
from pydantic import Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=32_000)
    session_id: str | None = None


class ChatOutput(BaseModel):
    """Output model for agent execution."""

    reply: str
    tool_calls_made: int = 0


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    reply: str
    session_id: str | None
    tool_calls_made: int
