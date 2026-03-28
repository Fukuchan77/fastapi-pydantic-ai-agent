"""Agent API routes and SSE streaming adapter.

This module provides FastAPI routes for the Pydantic AI chat agent,
including both standard request/response and Server-Sent Events (SSE)
streaming endpoints.
"""

import asyncio
import json
import logging
from typing import Protocol
from typing import runtime_checkable

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import ToolCallPart

from app.agents.deps import AgentDeps
from app.agents.deps import get_agent_deps
from app.deps.auth import verify_api_key
from app.models.agent import ChatRequest
from app.models.agent import ChatResponse


logger = logging.getLogger(__name__)


@runtime_checkable
class StreamAdapter(Protocol):
    r"""Protocol for SSE stream adapters.

    Defines the interface for formatting Server-Sent Events (SSE) in different
    protocols (standard SSE, Vercel AI Data Stream, AG-UI, etc.).
    """

    def format_event(self, event_type: str, content: str) -> str:
        r"""Format an SSE event with the given type and content.

        Args:
            event_type: Type of the event (e.g., "delta", "done", "error").
            content: Content payload for the event.

        Returns:
            Formatted SSE event string.
        """
        ...

    def format_done(self) -> str:
        """Format a terminal "done" event to signal stream completion.

        Returns:
            Formatted SSE event string for stream completion.
        """
        ...

    def format_error(self, message: str) -> str:
        """Format an error event with the given message.

        Args:
            message: Error message to include in the event.

        Returns:
            Formatted SSE event string for error.
        """
        ...


class DefaultSSEAdapter:
    r"""Default SSE adapter that emits standard JSON events.

    Produces SSE events in the format:
        data: {"type": "event_type", "content": "content"}\n\n

    This format is compatible with standard EventSource API clients.
    """

    def format_event(self, event_type: str, content: str) -> str:
        r"""Format an SSE event with the given type and content.

        Task 16.23: Added JSON serialization error handling to prevent
        crashes from unserializable content.

        Args:
            event_type: Type of the event (e.g., "delta", "done", "error").
            content: Content payload for the event.

        Returns:
            Formatted SSE event string in the format:
                 {"type": "...", "content": "..."}\n\n
        """
        try:
            payload = {"type": event_type, "content": content}
            return f"data: {json.dumps(payload)}\n\n"
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize SSE event: %s", e, exc_info=True)
            error_payload = {"type": "error", "content": "Serialization failed"}
            return f"data: {json.dumps(error_payload)}\n\n"

    def format_done(self) -> str:
        """Format a terminal "done" event to signal stream completion.

        Returns:
            Formatted SSE event string with type="done" and empty content.
        """
        return self.format_event("done", "")

    def format_error(self, message: str) -> str:
        """Format an error event with the given message.

        Args:
            message: Error message to include in the event.

        Returns:
            Formatted SSE event string with type="error" and the error message.
        """
        return self.format_event("error", message)


# Create router for agent endpoints
router = APIRouter(tags=["agent"])


@router.post("/agent/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    req: Request,
    deps: AgentDeps = Depends(get_agent_deps),  # noqa: B008
    _: None = Depends(verify_api_key),
) -> ChatResponse:
    """Handle chat requests with the Pydantic AI agent.

    This endpoint loads session history if a session_id is provided,
    runs the agent with the user's message, saves the updated history,
    and returns the agent's response.

    Args:
        request: ChatRequest with message and optional session_id.
        req: FastAPI Request object for accessing app.state.
        deps: AgentDeps with session_store and other dependencies.
        _: Authentication dependency (validates X-API-Key header).

    Returns:
        ChatResponse with the agent's reply, session_id, and tool call count.
    """
    # Load session history if session_id provided
    history = []
    if request.session_id:
        history = await deps.session_store.get_history(request.session_id)

    # Get the chat agent from app.state
    chat_agent = req.app.state.chat_agent

    # Run the agent with message and history
    result = await chat_agent.run(
        request.message,
        deps=deps,
        message_history=history,
    )

    # Save updated message history back to session store
    if request.session_id:
        await deps.session_store.save_history(
            request.session_id,
            result.all_messages(),
        )

    # Return response
    # Count tool calls from message history
    # Task 16.31: Count ToolCallPart instances in ModelResponse messages
    tool_calls_made = sum(
        1
        for m in result.all_messages()
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    )

    # Extract reply from result - handle both str output and Pydantic model output
    # When output_type is a Pydantic model with 'reply' field, use result.data.reply
    # Otherwise use result.output directly (str or other simple types)
    if hasattr(result, "data") and hasattr(result.data, "reply"):
        reply = result.data.reply
    else:
        # FunctionModel and simple str outputs use result.output
        # Task 16.32: Simplified - all Python objects have __str__, so no need for conditional
        reply = str(result.output)

    return ChatResponse(
        reply=reply,
        session_id=request.session_id,
        tool_calls_made=tool_calls_made,
    )


@router.post("/agent/stream")
async def stream_agent(
    request: ChatRequest,
    req: Request,
    deps: AgentDeps = Depends(get_agent_deps),  # noqa: B008
    _: None = Depends(verify_api_key),
) -> StreamingResponse:
    """Stream chat responses from the Pydantic AI agent via Server-Sent Events.

    This endpoint loads session history if a session_id is provided,
    runs the agent with streaming enabled, emits SSE events as tokens
    are generated, saves the updated history, and sends a terminal done event.

    Args:
        request: ChatRequest with message and optional session_id.
        req: FastAPI Request object for accessing app.state.
        deps: AgentDeps with session_store and other dependencies.
        _: Authentication dependency (validates X-API-Key header).

    Returns:
        StreamingResponse with text/event-stream media type.
    """
    from collections.abc import AsyncIterator

    from fastapi.responses import StreamingResponse

    adapter = DefaultSSEAdapter()

    async def generate() -> AsyncIterator[str]:
        """Generate SSE events from agent stream.

        Task 16.22: Added comprehensive error handling to distinguish
        different error types and prevent leaking internal details to clients.
        """
        try:
            # Load session history if session_id provided
            history = []
            if request.session_id:
                history = await deps.session_store.get_history(request.session_id)

            # Get the chat agent from app.state
            chat_agent = req.app.state.chat_agent

            # Run agent with streaming enabled
            async with chat_agent.run_stream(
                request.message,
                deps=deps,
                message_history=history,
            ) as result:
                # Stream deltas as they arrive
                async for delta in result.stream_text(delta=True):
                    yield adapter.format_event("delta", delta)

                # Collect all messages for history saving
                # Must be done inside the context manager while result is still valid
                all_messages = result.all_messages()

            # MEDIUM FIX: Save session BEFORE emitting done event
            # This ensures clients are only notified of completion if the session
            # was saved successfully. If save fails, the client won't receive a done event.
            if request.session_id:
                try:
                    await deps.session_store.save_history(
                        request.session_id,
                        all_messages,
                    )
                except ValueError as e:
                    # Validation errors (e.g., message count exceeded) - log and notify client
                    logger.warning(
                        "Failed to save session history for session %s: %s",
                        request.session_id,
                        e,
                    )
                    # Emit error event instead of done event
                    yield adapter.format_error(f"Failed to save session: {e}")
                    return  # Don't emit done event
                except Exception as e:
                    # Unexpected errors during save - log and notify client
                    logger.error(
                        "Unexpected error saving session history for session %s: %s",
                        request.session_id,
                        e,
                        exc_info=True,
                    )
                    # Emit error event instead of done event
                    yield adapter.format_error("Failed to save session")
                    return  # Don't emit done event

            # Emit terminal done event (only if save succeeded or no session_id)
            yield adapter.format_done()

        except asyncio.CancelledError:
            # Task 16.22: Client disconnected - log but don't send error event
            logger.info("Stream cancelled by client for message: %s", request.message[:50])
            raise
        except ValueError as e:
            # Task 16.22: Validation errors - safe to expose message
            logger.warning("Validation error in stream: %s", e)
            yield adapter.format_error("Invalid request parameters")
        except Exception as e:
            # Task 16.22: Unexpected errors - log full details, return generic message
            logger.error(
                "Unexpected error in agent stream: %s",
                e,
                exc_info=True,
                extra={"user_message": request.message[:100]},
            )
            yield adapter.format_error("An unexpected error occurred")

    return StreamingResponse(generate(), media_type="text/event-stream")
