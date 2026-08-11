"""Event-source generator for the agent SSE stream endpoint.

This module owns the agent-stream lifecycle guarantees: the `sse_max_events`
cap, `request.is_disconnected()` polling with resource release, a terminal
`error` event on exception, `asyncio.CancelledError` re-raise after cleanup,
closing the underlying async generator in a `finally` block, and a
`sse_send_timeout` timeout around producing each individual event so no
single stuck step (LLM call or tool execution) can hang the stream forever.

It does not own the SSE wire codec (`app.patterns.sse`) or usage/tool policy.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Protocol
from typing import cast
from typing import runtime_checkable

from fastapi import Request
from pydantic_ai import Agent
from pydantic_ai import NativeOutput
from pydantic_ai import RunUsage
from pydantic_ai import UsageLimitExceeded
from pydantic_ai import UsageLimits
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import PartDeltaEvent
from pydantic_ai.messages import PartStartEvent
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import TextPartDelta

from app.agents.chat_agent import ChatOutput
from app.agents.deps import AgentDeps
from app.agents.guardrails import GuardrailStopError
from app.agents.guardrails import build_guarded_toolset
from app.agents.guardrails import classify_usage_limit_exceeded
from app.config import Settings
from app.models.agent import ChatRequest
from app.patterns.sse import Completed
from app.patterns.sse import Error
from app.patterns.sse import SSEEvent
from app.patterns.sse import StepStarted
from app.patterns.sse import Token
from app.patterns.sse import ToolCalled
from app.patterns.sse import to_sse


logger = logging.getLogger(__name__)

_ARGS_SUMMARY_MAX_LEN = 200


@runtime_checkable
class _DisconnectAware(Protocol):
    """Structural interface for the disconnect check `_run_with_lifecycle_guards` needs.

    Narrower than `fastapi.Request` so lifecycle-guard unit tests can pass a
    lightweight fake instead of constructing a real ASGI request.
    """

    async def is_disconnected(self) -> bool: ...


def _summarize_tool_args(args: str | dict[str, object] | None) -> str:
    """Render a tool call's arguments as a short, truncated summary string.

    Args:
        args: The raw tool call arguments (JSON string, dict, or None).

    Returns:
        A summary string, truncated to `_ARGS_SUMMARY_MAX_LEN` characters.
    """
    if args is None:
        return ""
    text = args if isinstance(args, str) else json.dumps(args, default=str)
    if len(text) > _ARGS_SUMMARY_MAX_LEN:
        return text[:_ARGS_SUMMARY_MAX_LEN] + "...(truncated)"
    return text


async def _agent_event_stream(
    chat_agent: Agent[AgentDeps, str | ChatOutput],
    chat_request: ChatRequest,
    deps: AgentDeps,
    history: Sequence[ModelMessage],
    settings: Settings,
    usage: RunUsage | None = None,
) -> AsyncGenerator[SSEEvent]:
    """Drive `Agent.iter()` and map pydantic-ai's graph nodes/events onto the typed SSE union.

    Mapping: `ModelRequestNode` -> `StepStarted`; `PartDeltaEvent(TextPartDelta)`
    -> `Token`; `FunctionToolCallEvent` -> `ToolCalled`; the `End` node -> session
    save (if a session_id was provided) followed by `Completed`.

    When `chat_agent.output_type` is `NativeOutput` (Req 10.2), the model's
    streamed text *is* the raw JSON envelope, not user-facing text - text
    deltas are suppressed during the model-request node, and the parsed
    `ChatOutput.reply` is emitted as a single `Token` at the `End` node
    instead, so raw JSON never reaches the client.

    The run is wrapped in the same guardrails as the non-streaming chat path
    (Req 4.1/9.4: native `UsageLimits`, including `tool_calls_limit`; Req
    4.4-4.6: tool allow-list/approval/budget checks) via
    `build_guarded_toolset`/`agent.override`. A `GuardrailStopError`/
    `UsageLimitExceeded` raised here propagates out of this generator and is
    turned into a terminal `Error` SSE event by `_run_with_lifecycle_guards`.

    Per ADR-1 (same mechanism as `run_guarded`'s non-streaming path), `usage`
    is a caller-owned `RunUsage` that pydantic-ai mutates in place as the run
    progresses, so its counters stay readable after a `UsageLimitExceeded`
    raise even though the exception itself carries none - the caller passes
    the same instance to `_run_with_lifecycle_guards` so its consumer-side
    exception handler, which sits outside this generator's task, can read it.

    Args:
        chat_agent: The Pydantic AI chat agent to run.
        chat_request: The incoming chat request (message + optional session_id).
        deps: Agent dependencies (session store, settings, http client).
        history: Prior conversation history to seed the run with.
        settings: Application settings (usage_request_limit/usage_total_tokens_limit/
            usage_tool_calls_limit).
        usage: Caller-owned `RunUsage` passed through to `Agent.iter()`. A
            fresh instance is created when omitted (e.g. by call sites that
            don't need to read counters after the run).

    Yields:
        Typed SSE events as the agent run progresses.
    """
    is_native_output = isinstance(chat_agent.output_type, NativeOutput)
    run_usage = usage if usage is not None else RunUsage()

    limits = UsageLimits(
        request_limit=settings.usage_request_limit,
        total_tokens_limit=settings.usage_total_tokens_limit,
        tool_calls_limit=settings.usage_tool_calls_limit,
    )
    guarded_toolset = build_guarded_toolset(chat_agent, limits=limits, audit=deps.audit)

    # See `run_guarded`'s docstring: `override(toolsets=...)` alone does not
    # replace tools registered via `@agent.tool` - `tools=[]` empties that
    # slot while `guarded_toolset` (which already wraps a snapshot of the
    # original combined toolsets) is installed as the sole toolset.
    with chat_agent.override(tools=[], toolsets=[guarded_toolset]):
        async with chat_agent.iter(
            chat_request.message,
            deps=deps,
            message_history=list(history),
            usage_limits=limits,
            usage=run_usage,
        ) as agent_run:
            async for node in agent_run:
                if Agent.is_model_request_node(node):
                    yield StepStarted()
                    async with node.stream(agent_run.ctx) as request_stream:
                        async for event in request_stream:
                            if is_native_output:
                                # The streamed text is the raw JSON envelope, not
                                # user-facing text - drain without emitting; the
                                # parsed reply is emitted once at the End node.
                                continue
                            is_text_start = isinstance(event, PartStartEvent) and isinstance(
                                event.part, TextPart
                            )
                            if is_text_start and event.part.content:
                                yield Token(content=event.part.content)
                            elif (
                                isinstance(event, PartDeltaEvent)
                                and isinstance(event.delta, TextPartDelta)
                                and event.delta.content_delta
                            ):
                                yield Token(content=event.delta.content_delta)
                elif Agent.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                yield ToolCalled(
                                    name=event.part.tool_name,
                                    args_summary=_summarize_tool_args(event.part.args),
                                )
                elif Agent.is_end_node(node):
                    if is_native_output:
                        final_output = node.data.output
                        if isinstance(final_output, ChatOutput) and final_output.reply:
                            yield Token(content=final_output.reply)
                    if chat_request.session_id:
                        try:
                            await deps.session_store.save_history(
                                chat_request.session_id,
                                agent_run.all_messages(),
                            )
                        except Exception:
                            # `save_history` never raises for reaching the
                            # per-session cap (Req 3.4: it trims instead), and
                            # `stream_agent` has already authorized
                            # `session_id` before this generator runs, so a
                            # malformed id can't reach here either. This
                            # branch is a defensive catch-all for genuinely
                            # unexpected failures (e.g. a backend I/O error),
                            # not a capacity or validation path.
                            logger.error(
                                "Unexpected error saving session history for session %s",
                                chat_request.session_id,
                                exc_info=True,
                            )
                            yield Error(message="Failed to save session")
                            return
                    yield Completed()


class _QueueDone:
    """Unique sentinel type for the producer-done marker.

    A bare `object()` sentinel types as plain `object`, which absorbs
    `SSEEvent` in the queue's union and defeats `is`-narrowing after the
    `item is _QUEUE_DONE` check below. A dedicated type keeps `item` narrowed
    to `SSEEvent` in the non-sentinel branch.
    """


_QUEUE_DONE = _QueueDone()


async def _drive_to_queue(
    agen: AsyncGenerator[SSEEvent],
    queue: asyncio.Queue[SSEEvent | _QueueDone],
) -> None:
    """Run `agen` to completion in a single dedicated task, forwarding each event.

    `Agent.iter()` holds an anyio cancel scope open across many `yield`
    points; anyio requires that scope to be entered and exited by the *same*
    task. Driving `agen.__anext__()` from a fresh `asyncio.wait_for()` call on
    every loop iteration (as an earlier version of this function did) spawns a
    new task per call and violates that invariant, raising "Attempted to exit
    cancel scope in a different task than it was entered in". Running `agen`
    to completion inside one persistent task — this one — and communicating
    with the consumer only through a plain `asyncio.Queue` (which has no
    task-affinity requirement) avoids the problem entirely. `agen.aclose()`
    is called here so cleanup also runs inside `agen`'s own task.

    Args:
        agen: The raw async generator of typed SSE events to drive.
        queue: Queue events are forwarded to; a sentinel is put on exit.
    """
    try:
        async for event in agen:
            await queue.put(event)
    finally:
        await agen.aclose()
        await queue.put(_QUEUE_DONE)


async def _run_with_lifecycle_guards(
    request: _DisconnectAware,
    agen: AsyncGenerator[SSEEvent],
    settings: Settings,
    *,
    usage: RunUsage | None = None,
    message_length: int = 0,
) -> AsyncIterator[str]:
    """Enforce the SSE stream's lifecycle guarantees around a raw event source.

    Owns: the `sse_max_events` cap, `request.is_disconnected()` polling,
    the terminal `error` event on exception, `CancelledError` re-raise after
    cleanup, closing `agen` (via `_drive_to_queue`'s `finally`), and the
    per-event `sse_send_timeout`.

    Args:
        request: The incoming request (only `is_disconnected()` is used).
        agen: The raw async generator of typed SSE events to guard.
        settings: Application settings (sse_max_events/sse_send_timeout).
        usage: The same caller-owned `RunUsage` passed into the event source
            driving `agen` (e.g. `_agent_event_stream`), if any. `agen` runs
            in its own dedicated task (see `_drive_to_queue`); this handler
            runs in the caller's task instead, outside that task and outside
            the `Agent.iter()` context manager it held open. Reading the same
            `RunUsage` instance here (mutated in place by pydantic-ai as the
            run progressed) is what lets a `UsageLimitExceeded` here report
            the same requests/tool_calls/total_tokens detail as the
            non-streaming path's `run_guarded()` can (Req 9.4).
        message_length: Length of the user's message, logged as metadata
            instead of the message content itself (never log raw user input).

    Yields:
        SSE wire-format strings ready to send to the client.
    """
    queue: asyncio.Queue[SSEEvent | _QueueDone] = asyncio.Queue()
    producer = asyncio.ensure_future(_drive_to_queue(agen, queue))
    # Yield once so `producer` starts running (enters its try/finally) before
    # anything below can cancel it — cancelling a task that never started its
    # coroutine body would skip `_drive_to_queue`'s `finally: agen.aclose()`.
    await asyncio.sleep(0)
    event_count = 0
    try:
        while True:
            if await request.is_disconnected():
                logger.info("Client disconnected; stopping agent stream")
                return
            try:
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=settings.sse_send_timeout,
                )
            except TimeoutError:
                logger.error(
                    "Agent stream timed out after %ds waiting for the next event",
                    settings.sse_send_timeout,
                )
                yield to_sse(Error(message="Stream timed out"))
                return
            if item is _QUEUE_DONE:
                if producer.cancelled():
                    # The event source's own run was cancelled (not just our
                    # polling of it) — treat this the same as being cancelled
                    # ourselves: no error event, just re-raise after cleanup.
                    raise asyncio.CancelledError
                exc = producer.exception()
                if exc is not None:
                    if isinstance(exc, GuardrailStopError):
                        logger.warning("Agent stream tool call refused: %s", exc.stop_reason)
                        yield to_sse(Error(message=f"Tool call refused: {exc.stop_reason}"))
                    elif isinstance(exc, UsageLimitExceeded):
                        stop_reason = classify_usage_limit_exceeded(exc)
                        logger.warning("Agent stream usage limit exceeded: %s", stop_reason)
                        detail = (
                            f" (requests={usage.requests}, tool_calls={usage.tool_calls}, "
                            f"total_tokens={usage.total_tokens})"
                            if usage is not None
                            else ""
                        )
                        yield to_sse(Error(message=f"Usage limit exceeded: {stop_reason}{detail}"))
                    else:
                        logger.error(
                            "Unexpected error in agent stream",
                            exc_info=exc,
                            extra={"message_length": message_length},
                        )
                        yield to_sse(Error(message="An unexpected error occurred"))
                return
            event_count += 1
            # `item is _QUEUE_DONE` above returns/raises on every path, so by
            # construction `item` is always `SSEEvent` here - but that
            # exhaustiveness spans a `while True:` loop plus nested
            # if/elif/else/raise, which ty's flow narrowing doesn't follow.
            # `cast` documents the invariant instead of suppressing the check.
            yield to_sse(cast(SSEEvent, item))
            if event_count >= settings.sse_max_events:
                logger.warning(
                    "SSE event cap (%d) reached; stopping stream",
                    settings.sse_max_events,
                )
                return
    except asyncio.CancelledError:
        logger.info(
            "Agent stream cancelled by client (message_length=%d)",
            message_length,
        )
        raise
    finally:
        if not producer.done():
            producer.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await producer


async def event_source(
    request: Request,
    chat_agent: Agent[AgentDeps, str | ChatOutput],
    chat_request: ChatRequest,
    deps: AgentDeps,
    settings: Settings,
) -> AsyncIterator[str]:
    """Build the agent event stream and guard its lifecycle, yielding SSE wire text.

    Owns a single caller-owned `RunUsage` for the whole request, shared
    between the event source (`_agent_event_stream`, which runs in its own
    dedicated task) and the lifecycle guard (`_run_with_lifecycle_guards`,
    whose `UsageLimitExceeded` handling runs outside that task) so both see
    the same mutated-in-place counters (Req 9.4).

    Args:
        request: The incoming FastAPI request (for disconnect polling).
        chat_agent: The Pydantic AI chat agent to run.
        chat_request: The incoming chat request (message + optional session_id).
        deps: Agent dependencies (session store, settings, http client).
        settings: Application settings (sse_max_events/sse_send_timeout).

    Yields:
        SSE wire-format strings ready to send to the client.
    """
    history: list[ModelMessage] = []
    if chat_request.session_id:
        history = await deps.session_store.get_history(chat_request.session_id)

    usage = RunUsage()
    agen = _agent_event_stream(chat_agent, chat_request, deps, history, settings, usage=usage)
    async for wire in _run_with_lifecycle_guards(
        request,
        agen,
        settings,
        usage=usage,
        message_length=len(chat_request.message),
    ):
        yield wire
