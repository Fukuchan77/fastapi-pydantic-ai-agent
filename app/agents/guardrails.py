"""Agent execution guardrails: usage limits, tool allow-list, approval, and audit.

`run_guarded()` wraps a single `agent.run()` call with:
- native `UsageLimits` (token/request budgets, Req 4.1),
- a tool allow-list check (Req 4.4),
- an optional approval hook invoked before a guarded tool executes (Req 4.5),
- a token-budget check before executing any tool (Req 4.6, "side effects" per
  the spec clarification means every registered tool - Req 15's read-only flag
  is deferred),
- and an audit trail of every refused/denied/budget-blocked attempt (Req 4.7).

All three checks are enforced by wrapping the agent's own toolsets in
`_GuardedToolset.call_tool()` for the duration of the run, via
`agent.override(tools=[], toolsets=[guarded])` - the tool remains visible to
the model (so a scripted/hallucinated call to it is still dispatched to
`call_tool()` for auditing) rather than being hidden from its schema.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai import CombinedToolset
from pydantic_ai import RunContext
from pydantic_ai import UsageLimitExceeded
from pydantic_ai import UsageLimits
from pydantic_ai.messages import ModelMessage
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset


StopReason = Literal["completed", "max_iterations", "budget_exceeded", "denied", "disallowed_tool"]
"""Closed vocabulary for why a guarded agent run stopped (Req 4.3)."""

ApprovalHook = Callable[[str, dict[str, Any]], Awaitable[bool]]
"""Invoked as `approval_hook(tool_name, tool_args)` before a guarded tool call executes."""

_ARGS_SUMMARY_MAX_LEN = 200


class AuditRecord(BaseModel):
    """A single refused, denied, or budget-blocked tool-call attempt (Req 4.7)."""

    tool_name: str
    stop_reason: StopReason
    detail: str = ""


class AuditTrail:
    """In-memory, append-only sink of `AuditRecord`s for a single agent run/session."""

    def __init__(self) -> None:
        """Start with an empty trail."""
        self.entries: list[AuditRecord] = []

    def record(self, entry: AuditRecord) -> None:
        """Append a refused/denied/budget-blocked attempt to the trail.

        Args:
            entry: The audit record to append.
        """
        self.entries.append(entry)


@dataclass
class GuardedResult[OutputT]:
    """The outcome of a `run_guarded()` call."""

    output: OutputT | None
    stop_reason: StopReason
    audit: list[AuditRecord]
    messages: list[ModelMessage] = field(default_factory=list)


class GuardrailStopError(Exception):
    """Raised from `_GuardedToolset.call_tool` to abort the run.

    Public so callers driving `Agent.iter()` directly (e.g. the SSE stream
    generator in `app.api.v1._stream`) can catch it and map it to a
    user-facing signal instead of treating it as an unexpected error.
    """

    def __init__(self, stop_reason: StopReason, tool_name: str) -> None:
        """Carry the stop reason and refused tool name up to the caller.

        Args:
            stop_reason: Why the run is being aborted.
            tool_name: The tool call that triggered the abort.
        """
        self.stop_reason = stop_reason
        self.tool_name = tool_name
        super().__init__(f"Guarded run stopped ({stop_reason}) on tool {tool_name!r}")


def _summarize_tool_args(tool_args: dict[str, Any]) -> str:
    """Render tool-call arguments as a short, truncated summary for the audit trail.

    Args:
        tool_args: The raw tool call arguments.

    Returns:
        A summary string, truncated to `_ARGS_SUMMARY_MAX_LEN` characters.
    """
    text = str(tool_args)
    if len(text) > _ARGS_SUMMARY_MAX_LEN:
        return text[:_ARGS_SUMMARY_MAX_LEN] + "...(truncated)"
    return text


@dataclass
class _GuardedToolset[DepsT](WrapperToolset[DepsT]):
    """Wraps the agent's combined toolset to enforce allow-list/approval/budget checks."""

    allowed_tools: set[str] | None
    approval_hook: ApprovalHook | None
    limits: UsageLimits
    audit: AuditTrail

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[DepsT],
        tool: ToolsetTool[DepsT],
    ) -> Any:  # noqa: ANN401 - matches AbstractToolset.call_tool's own `-> Any` signature
        """Enforce allow-list, approval, and pre-side-effect budget checks before delegating.

        Args:
            name: The tool name the model requested.
            tool_args: The tool call's arguments.
            ctx: The current run context (used to read accumulated token usage).
            tool: The tool definition resolved by the wrapped toolset.

        Returns:
            The wrapped toolset's result when every check passes.

        Raises:
            GuardrailStopError: When the allow-list, approval, or budget check fails.
        """
        if self.allowed_tools is not None and name not in self.allowed_tools:
            self.audit.record(
                AuditRecord(
                    tool_name=name,
                    stop_reason="disallowed_tool",
                    detail=_summarize_tool_args(tool_args),
                )
            )
            raise GuardrailStopError("disallowed_tool", name)

        if self.approval_hook is not None and not await self.approval_hook(name, tool_args):
            self.audit.record(
                AuditRecord(
                    tool_name=name,
                    stop_reason="denied",
                    detail=_summarize_tool_args(tool_args),
                )
            )
            raise GuardrailStopError("denied", name)

        total_tokens_limit = self.limits.total_tokens_limit
        if total_tokens_limit is not None and ctx.usage.total_tokens >= total_tokens_limit:
            self.audit.record(
                AuditRecord(
                    tool_name=name,
                    stop_reason="budget_exceeded",
                    detail=_summarize_tool_args(tool_args),
                )
            )
            raise GuardrailStopError("budget_exceeded", name)

        return await super().call_tool(name, tool_args, ctx, tool)


def build_guarded_toolset[DepsT](
    agent: Agent[DepsT, Any],
    *,
    limits: UsageLimits,
    audit: AuditTrail,
    allowed_tools: set[str] | None = None,
    approval_hook: ApprovalHook | None = None,
) -> AbstractToolset[DepsT]:
    """Build the guarded toolset wrapping `agent`'s own toolsets.

    Shared by `run_guarded()` and callers that drive `Agent.iter()` directly
    (e.g. the SSE stream generator), so both entry points enforce the same
    allow-list/approval/budget checks via `agent.override(toolsets=[...])`.

    Args:
        agent: The agent whose registered toolsets should be wrapped.
        limits: Native `UsageLimits`; `total_tokens_limit` gates the
            pre-tool-call budget check (Req 4.6).
        audit: Sink audit records are appended to.
        allowed_tools: Tool names permitted to execute. `None` disables the
            allow-list check entirely.
        approval_hook: Awaited with `(tool_name, tool_args)` before a tool
            executes.

    Returns:
        A toolset suitable for `agent.override(toolsets=[...])`.
    """
    return _GuardedToolset(
        CombinedToolset(list(agent.toolsets)),
        allowed_tools=allowed_tools,
        approval_hook=approval_hook,
        limits=limits,
        audit=audit,
    )


def classify_usage_limit_exceeded(exc: UsageLimitExceeded) -> StopReason:
    """Map a native `UsageLimitExceeded` to the closed `StopReason` vocabulary.

    Request-count and tool-call-count limits describe the agentic loop running
    too many iterations; token limits describe exhausting the token budget.

    Args:
        exc: The exception raised by `UsageLimits` enforcement.

    Returns:
        `"max_iterations"` for request/tool-call limits, `"budget_exceeded"` for
        token limits.
    """
    message = str(exc)
    if "request_limit" in message or "tool_calls_limit" in message:
        return "max_iterations"
    return "budget_exceeded"


async def run_guarded[DepsT, OutputT](
    agent: Agent[DepsT, OutputT],
    user_prompt: str,
    *,
    deps: DepsT,
    message_history: Sequence[ModelMessage] | None = None,
    limits: UsageLimits,
    allowed_tools: set[str] | None = None,
    approval_hook: ApprovalHook | None = None,
    audit: AuditTrail | None = None,
) -> GuardedResult[OutputT]:
    """Run `agent` with usage limits, a tool allow-list, approval, and an audit trail.

    Args:
        agent: The Pydantic AI agent to run.
        user_prompt: The user's message.
        deps: Agent dependencies to inject.
        message_history: Prior conversation history to seed the run with.
        limits: Native `UsageLimits` applied to the run (Req 4.1); its
            `total_tokens_limit` also gates the pre-tool-call budget check (Req 4.6).
        allowed_tools: Tool names permitted to execute. `None` disables the
            allow-list check entirely (no tool-name restriction).
        approval_hook: Awaited with `(tool_name, tool_args)` before a tool
            executes; a `False` result stops the run with `stop_reason="denied"`.
        audit: Sink audit records are appended to. A fresh `AuditTrail` is
            created when omitted.

    Returns:
        GuardedResult carrying the final output and every exchanged message
        on `stop_reason="completed"`; `output`/`messages` stay empty for any
        stopped run (Req 4.7's audit trail - not session history - is the
        record of a refused/denied/budget-blocked turn).
    """
    audit = audit if audit is not None else AuditTrail()
    guarded_toolset = build_guarded_toolset(
        agent,
        limits=limits,
        audit=audit,
        allowed_tools=allowed_tools,
        approval_hook=approval_hook,
    )

    try:
        # `agent.toolsets` always re-includes the agent's own function
        # toolset (tools registered via `@agent.tool`) regardless of
        # `override(toolsets=...)` - only `override(tools=...)` replaces it.
        # Emptying `tools` while installing `guarded_toolset` (which already
        # wraps a snapshot of the *original* combined toolsets, captured
        # above) as the sole toolset avoids double-registering every
        # directly-registered tool under two names.
        with agent.override(tools=[], toolsets=[guarded_toolset]):
            result = await agent.run(
                user_prompt,
                deps=deps,
                message_history=list(message_history) if message_history else None,
                usage_limits=limits,
            )
    except GuardrailStopError as exc:
        return GuardedResult(output=None, stop_reason=exc.stop_reason, audit=list(audit.entries))
    except UsageLimitExceeded as exc:
        return GuardedResult(
            output=None,
            stop_reason=classify_usage_limit_exceeded(exc),
            audit=list(audit.entries),
        )

    return GuardedResult(
        output=result.output,
        stop_reason="completed",
        audit=list(audit.entries),
        messages=list(result.all_messages()),
    )
