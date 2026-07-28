"""Agent API routes and SSE streaming endpoint.

This module provides FastAPI routes for the Pydantic AI chat agent,
including both standard request/response and Server-Sent Events (SSE)
streaming endpoints. The SSE wire contract is the typed 5-event union
defined in `app.patterns.sse`; stream lifecycle hardening is owned by
`app.api.v1._stream`.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import ToolCallPart

from app.agents.chat_agent import ChatOutput
from app.agents.deps import AgentDeps
from app.agents.deps import get_agent_deps
from app.api.v1._stream import event_source
from app.deps.auth import verify_api_key
from app.models.agent import ChatRequest
from app.models.agent import ChatResponse


logger = logging.getLogger(__name__)

_HEARTBEAT_COMMENT = ": heartbeat\n\n"

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
    # Count ToolCallPart instances in ModelResponse messages
    tool_calls_made = sum(
        1
        for m in result.all_messages()
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    )

    # NativeOutput-capable models (Req 10.2) produce a ChatOutput instance;
    # other models produce plain str (Req 10.3) - see build_chat_agent().
    reply = result.output.reply if isinstance(result.output, ChatOutput) else str(result.output)

    return ChatResponse(
        reply=reply,
        session_id=request.session_id,
        tool_calls_made=tool_calls_made,
    )


async def _with_heartbeat(
    agen: AsyncIterator[str],
    interval: float,
) -> AsyncIterator[str]:
    """Interleave SSE heartbeat comments while `agen` is idle (Req 2.8).

    Uses `asyncio.wait()` (never `asyncio.wait_for()`) around the pending
    upstream call, so a heartbeat tick only checks readiness — it never
    cancels the in-flight event `_stream.py` is still producing.

    Args:
        agen: The SSE wire-text generator to wrap (e.g. from `event_source`).
        interval: Seconds to wait for the next value before emitting a heartbeat.

    Yields:
        Values from `agen` in order, interspersed with heartbeat comments.
    """
    next_task: asyncio.Task[str] | None = None
    try:
        while True:
            if next_task is None:
                next_task = asyncio.ensure_future(agen.__anext__())
            done, _pending = await asyncio.wait({next_task}, timeout=interval)
            if not done:
                yield _HEARTBEAT_COMMENT
                continue
            try:
                yield next_task.result()
            except StopAsyncIteration:
                return
            finally:
                next_task = None
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()


@router.post("/agent/stream")
async def stream_agent(
    request: ChatRequest,
    req: Request,
    deps: AgentDeps = Depends(get_agent_deps),  # noqa: B008
    _: None = Depends(verify_api_key),
) -> StreamingResponse:
    """Stream chat responses from the Pydantic AI agent via Server-Sent Events.

    Emits the typed 5-event union (`step_started`/`tool_called`/`token`/
    `completed`/`error`) defined in `app.patterns.sse`. Session history is
    loaded and saved by the event source in `app.api.v1._stream`; this route
    only wires headers and the idle heartbeat.

    Args:
        request: ChatRequest with message and optional session_id.
        req: FastAPI Request object for accessing app.state.
        deps: AgentDeps with session_store and other dependencies.
        _: Authentication dependency (validates X-API-Key header).

    Returns:
        StreamingResponse with text/event-stream media type.
    """
    settings = req.app.state.settings
    chat_agent = req.app.state.chat_agent
    wire_stream = event_source(req, chat_agent, request, deps, settings)
    return StreamingResponse(
        _with_heartbeat(wire_stream, settings.sse_heartbeat_interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
