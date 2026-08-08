"""Unit tests for agent execution guardrails (Req 4.1-4.7).

Per plan.md's AgentGuardrails Testability note, `run_guarded()` is exercised
directly against `mock_web_search` (registered via `register_mock_tools`,
dev-only per `app.agents.tools_mock`) with a `TestModel` scripted to call it,
so the disallowed_tool/denied/budget_exceeded branches don't depend on Req 15's
deferred real-tool work.
"""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai import RunUsage
from pydantic_ai import UsageLimits
from pydantic_ai.models.test import TestModel

from app.agents.deps import AgentDeps
from app.agents.guardrails import AuditRecord
from app.agents.guardrails import AuditTrail
from app.agents.guardrails import GuardrailStopError
from app.agents.guardrails import build_guarded_toolset
from app.agents.guardrails import classify_usage_limit_exceeded
from app.agents.guardrails import run_guarded
from app.agents.tools_mock import register_mock_tools
from app.stores.session_store import SessionStore
from tests.conftest import build_test_settings


def _build_agent(model: TestModel) -> Agent[AgentDeps, str]:
    """Build a bare agent with the dev-only mock tool registered."""
    agent: Agent[AgentDeps, str] = Agent(model=model, deps_type=AgentDeps, output_type=str)
    register_mock_tools(agent)
    return agent


def _build_deps() -> AgentDeps:
    return AgentDeps(
        http_client=Mock(spec=httpx.AsyncClient),
        settings=build_test_settings(),
        session_store=Mock(spec=SessionStore),
    )


class TestRunGuardedCompleted:
    """Req 4.3: a run with no refused/denied/budget-blocked tool call completes normally."""

    @pytest.mark.asyncio
    async def test_completes_with_no_tool_calls(self) -> None:
        """A run with no tool calls completes with the model's output."""
        model = TestModel(call_tools=[], custom_output_text="hello there")
        agent = _build_agent(model)

        result = await run_guarded(
            agent,
            "hi",
            deps=_build_deps(),
            limits=UsageLimits(),
        )

        assert result.stop_reason == "completed"
        assert result.output == "hello there"
        assert result.audit == []

    @pytest.mark.asyncio
    async def test_completes_when_allowed_tool_is_called(self) -> None:
        """An allowed tool call executes normally and the run still completes."""
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)

        result = await run_guarded(
            agent,
            "search for something",
            deps=_build_deps(),
            limits=UsageLimits(),
            allowed_tools={"mock_web_search"},
        )

        assert result.stop_reason == "completed"
        assert result.audit == []

    @pytest.mark.asyncio
    async def test_no_allow_list_permits_any_registered_tool(self) -> None:
        """allowed_tools=None (the default) disables the allow-list check entirely."""
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)

        result = await run_guarded(
            agent,
            "search for something",
            deps=_build_deps(),
            limits=UsageLimits(),
        )

        assert result.stop_reason == "completed"
        assert result.audit == []


class TestRunGuardedDisallowedTool:
    """Req 4.4: a tool absent from allowed_tools is refused, audited, and stops the run."""

    @pytest.mark.asyncio
    async def test_stops_with_disallowed_tool(self) -> None:
        """A tool call outside allowed_tools stops the run and is audited."""
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)

        result = await run_guarded(
            agent,
            "search for something",
            deps=_build_deps(),
            limits=UsageLimits(),
            allowed_tools=set(),
        )

        assert result.stop_reason == "disallowed_tool"
        assert result.output is None
        assert len(result.audit) == 1
        assert result.audit[0].tool_name == "mock_web_search"
        assert result.audit[0].stop_reason == "disallowed_tool"

    @pytest.mark.asyncio
    async def test_records_attempt_on_shared_audit_sink(self) -> None:
        """The caller's own AuditTrail (e.g. AgentDeps.audit) accumulates the attempt."""
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)
        audit = AuditTrail()

        result = await run_guarded(
            agent,
            "search for something",
            deps=_build_deps(),
            limits=UsageLimits(),
            allowed_tools={"some_other_tool"},
            audit=audit,
        )

        assert result.stop_reason == "disallowed_tool"
        assert audit.entries == result.audit
        assert len(audit.entries) == 1


class TestRunGuardedApprovalHook:
    """Req 4.5: an approval hook is invoked before a guarded tool executes."""

    @pytest.mark.asyncio
    async def test_denied_when_hook_refuses(self) -> None:
        """A refusing approval hook stops the run with stop_reason=denied."""
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)
        approval_hook = AsyncMock(return_value=False)

        result = await run_guarded(
            agent,
            "search for something",
            deps=_build_deps(),
            limits=UsageLimits(),
            allowed_tools={"mock_web_search"},
            approval_hook=approval_hook,
        )

        assert result.stop_reason == "denied"
        assert result.output is None
        assert len(result.audit) == 1
        assert result.audit[0].stop_reason == "denied"
        approval_hook.assert_awaited_once()
        assert approval_hook.call_args[0][0] == "mock_web_search"

    @pytest.mark.asyncio
    async def test_completes_when_hook_approves(self) -> None:
        """An approving approval hook lets the tool call through and the run completes."""
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)
        approval_hook = AsyncMock(return_value=True)

        result = await run_guarded(
            agent,
            "search for something",
            deps=_build_deps(),
            limits=UsageLimits(),
            allowed_tools={"mock_web_search"},
            approval_hook=approval_hook,
        )

        assert result.stop_reason == "completed"
        assert result.audit == []
        approval_hook.assert_awaited_once()


class TestRunGuardedBudgetExceeded:
    """Req 4.6: the token budget is checked before executing any tool with side effects."""

    @pytest.mark.asyncio
    async def test_call_tool_stops_when_usage_already_at_limit(self) -> None:
        """Direct unit test of the guarded toolset's `call_tool()`.

        A full `run_guarded()` round-trip can't isolate this branch: pydantic-ai's
        own `UsageLimits.check_tokens()` re-checks `total_tokens_limit` immediately
        after the model response that requests the tool call - before our
        toolset's `call_tool()` ever runs - so for a shared threshold the native
        check always raises first (also correctly classified as
        stop_reason="budget_exceeded", see TestRunGuardedMaxIterations and
        TestClassifyUsageLimitExceeded), leaving no room for ours to fire.
        Calling `call_tool()` directly with an already-elevated `RunContext.usage`
        exercises the actual unit under test instead.
        """
        from pydantic_ai import RunContext
        from pydantic_ai import RunUsage

        model = TestModel(call_tools=[])
        agent = _build_agent(model)
        audit = AuditTrail()
        limits = UsageLimits(total_tokens_limit=10)
        guarded = build_guarded_toolset(
            agent,
            limits=limits,
            audit=audit,
            allowed_tools={"mock_web_search"},
        )
        deps = _build_deps()
        ctx = RunContext(deps=deps, model=model, usage=RunUsage(input_tokens=20, output_tokens=0))
        tools = await guarded.get_tools(ctx)
        tool = tools["mock_web_search"]

        with pytest.raises(GuardrailStopError) as exc_info:
            await guarded.call_tool("mock_web_search", {"query": "x"}, ctx, tool)

        assert exc_info.value.stop_reason == "budget_exceeded"
        assert len(audit.entries) == 1
        assert audit.entries[0].stop_reason == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_no_budget_check_when_total_tokens_limit_unset(self) -> None:
        """total_tokens_limit=None disables the pre-tool-call budget check entirely."""
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)

        result = await run_guarded(
            agent,
            "search for something",
            deps=_build_deps(),
            limits=UsageLimits(total_tokens_limit=None),
            allowed_tools={"mock_web_search"},
        )

        assert result.stop_reason == "completed"


class TestRunGuardedMaxIterations:
    """Req 4.1/4.3: native UsageLimits enforcement maps to max_iterations."""

    @pytest.mark.asyncio
    async def test_request_limit_exceeded_maps_to_max_iterations(self) -> None:
        """A tool call forces a 2nd model request; request_limit=1 blocks it."""
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)

        result = await run_guarded(
            agent,
            "search for something",
            deps=_build_deps(),
            limits=UsageLimits(request_limit=1),
            allowed_tools={"mock_web_search"},
        )

        assert result.stop_reason == "max_iterations"
        assert result.output is None


class TestRunGuardedUsageLimitAudit:
    """Req 7.4/7.5/9.6: a native UsageLimitExceeded records audit detail too.

    Before this task, `run_guarded`'s `except UsageLimitExceeded` returned an
    empty `audit` list - the native check raises before `_GuardedToolset`
    ever runs, so nothing was ever recorded for the most common budget stop.
    The detail must be derived from *our own* `limits` crossed with the
    caller-owned `usage` snapshot (ADR-1), never from `exc`'s message text.
    """

    @pytest.mark.asyncio
    async def test_request_limit_stop_is_recorded_with_derived_detail(self) -> None:
        """A request_limit stop appends one audit entry naming requests=1/1."""
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)

        result = await run_guarded(
            agent,
            "search for something",
            deps=_build_deps(),
            limits=UsageLimits(request_limit=1),
            allowed_tools={"mock_web_search"},
        )

        assert result.stop_reason == "max_iterations"
        assert len(result.audit) == 1
        entry = result.audit[0]
        assert entry.stop_reason == "max_iterations"
        assert "requests=1/1" in entry.detail

    @pytest.mark.asyncio
    async def test_detail_is_derived_from_snapshot_not_exception_message(self) -> None:
        """The recorded detail reflects our limits x usage, not exc's wording.

        Proven by mismatching the two: a total_tokens_limit low enough to
        never be reached by TestModel's tiny fixture responses, together with
        a request_limit=1 stop. If the detail were built from `str(exc)` it
        would read "request_limit" (the exception's own wording); the
        snapshot-derived detail instead names the counter that actually
        crossed its configured value.
        """
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)

        result = await run_guarded(
            agent,
            "search for something",
            deps=_build_deps(),
            limits=UsageLimits(request_limit=1, total_tokens_limit=1_000_000),
            allowed_tools={"mock_web_search"},
        )

        assert result.stop_reason == "max_iterations"
        entry = result.audit[0]
        assert "requests=1/1" in entry.detail
        assert "total_tokens" not in entry.detail

    @pytest.mark.asyncio
    async def test_no_crossed_limit_falls_back_to_snapshot_only_detail(self) -> None:
        """A synthetic raise with no configured limit still yields a snapshot detail.

        Exercises `_usage_limit_detail`'s defensive fallback directly - the
        library never actually raises without a configured limit crossed, so
        this path is otherwise unreachable through `run_guarded` itself.
        """
        from app.agents.guardrails import _usage_limit_detail

        detail = _usage_limit_detail(UsageLimits(request_limit=None), RunUsage(requests=3))

        assert "requests=3" in detail
        assert "no configured limit matched" in detail


class TestClassifyUsageLimitExceeded:
    """Direct unit coverage of the UsageLimitExceeded -> StopReason mapping."""

    def test_request_limit_message_maps_to_max_iterations(self) -> None:
        """A request_limit message classifies as max_iterations."""
        from pydantic_ai import UsageLimitExceeded

        exc = UsageLimitExceeded("The next request would exceed the request_limit of 1")

        assert classify_usage_limit_exceeded(exc) == "max_iterations"

    def test_tool_calls_limit_message_maps_to_max_iterations(self) -> None:
        """A tool_calls_limit message classifies as max_iterations."""
        from pydantic_ai import UsageLimitExceeded

        exc = UsageLimitExceeded("The next tool call(s) would exceed the tool_calls_limit of 1")

        assert classify_usage_limit_exceeded(exc) == "max_iterations"

    def test_token_limit_message_maps_to_budget_exceeded(self) -> None:
        """A total_tokens_limit message classifies as budget_exceeded."""
        from pydantic_ai import UsageLimitExceeded

        exc = UsageLimitExceeded("Exceeded the total_tokens_limit of 10")

        assert classify_usage_limit_exceeded(exc) == "budget_exceeded"

    def test_unrecognized_message_hits_the_documented_default_branch(self) -> None:
        """Req 7.5: an unrecognised message still resolves to one documented default.

        The dependency exposes no attribute distinguishing an iteration limit
        from a token limit (R1); a message naming neither `request_limit` nor
        `tool_calls_limit` must still classify deterministically rather than
        raise or return an undeclared value (7.6).
        """
        from pydantic_ai import UsageLimitExceeded

        exc = UsageLimitExceeded("some future limit_kind the library has not shipped yet")

        assert classify_usage_limit_exceeded(exc) == "budget_exceeded"


class TestRunGuardedUsageObject:
    """Req 9.4/14.2 (ADR-1): a caller-owned RunUsage is passed into agent.run().

    The native exception carries no counters, so `run_guarded` must own the
    `RunUsage` instance itself - the library mutates it in place - to make
    accurate post-stop counters readable (closes the 14.2 verification
    finding that `UsageLimits.tool_calls_limit` was never set/observable).
    """

    @pytest.mark.asyncio
    async def test_passes_caller_owned_usage_object_to_agent_run(self) -> None:
        """A completed run still passes a real, populated RunUsage to agent.run()."""
        model = TestModel(call_tools=[], custom_output_text="hello there")
        agent = _build_agent(model)

        with patch.object(agent, "run", wraps=agent.run) as spy_run:
            result = await run_guarded(
                agent,
                "hi",
                deps=_build_deps(),
                limits=UsageLimits(),
            )

        assert result.stop_reason == "completed"
        spy_run.assert_awaited_once()
        _, kwargs = spy_run.call_args
        usage = kwargs.get("usage")
        assert isinstance(usage, RunUsage)
        assert usage.requests >= 1

    @pytest.mark.asyncio
    async def test_usage_object_is_mutated_in_place_even_on_a_budget_stop(self) -> None:
        """The passed-in RunUsage reflects real counters even when the native check raises."""
        model = TestModel(call_tools=["mock_web_search"])
        agent = _build_agent(model)

        with patch.object(agent, "run", wraps=agent.run) as spy_run:
            result = await run_guarded(
                agent,
                "search for something",
                deps=_build_deps(),
                limits=UsageLimits(request_limit=1),
                allowed_tools={"mock_web_search"},
            )

        assert result.stop_reason == "max_iterations"
        spy_run.assert_awaited_once()
        _, kwargs = spy_run.call_args
        usage = kwargs.get("usage")
        assert isinstance(usage, RunUsage)
        assert usage.requests >= 1


class TestAuditTrail:
    """Basic contract of the in-memory audit sink."""

    def test_starts_empty(self) -> None:
        """A fresh AuditTrail has no entries."""
        assert AuditTrail().entries == []

    def test_record_appends(self) -> None:
        """record() appends the given entry to .entries."""
        trail = AuditTrail()
        entry = AuditRecord(tool_name="t", stop_reason="denied")

        trail.record(entry)

        assert trail.entries == [entry]
