"""Integration tests for the pydantic-ai node/event -> typed SSE union mapping.

Exercises `_agent_event_stream()` (app/api/v1/_stream.py) against a real
`Agent` driven by `FunctionModel`, per Task 5's `Agent.iter()`-based mapping:
`ModelRequestNode` -> StepStarted, `PartDeltaEvent(TextPartDelta)` -> Token,
`FunctionToolCallEvent` -> ToolCalled, the `End` node -> session save (if a
session_id was given) then Completed.
"""

from collections.abc import AsyncIterator
from collections.abc import Sequence
from unittest.mock import patch

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai import RunUsage
from pydantic_ai import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import DeltaToolCall
from pydantic_ai.models.function import FunctionModel

from app.agents.deps import AgentDeps
from app.api.v1._stream import _agent_event_stream
from app.api.v1._stream import _run_with_lifecycle_guards
from app.models.agent import ChatRequest
from app.patterns.sse import Completed
from app.patterns.sse import Error
from app.patterns.sse import StepStarted
from app.patterns.sse import Token
from app.patterns.sse import ToolCalled
from app.patterns.sse import parse_sse_events
from app.stores.session_store import InMemorySessionStore
from tests.conftest import build_test_settings


async def _text_only_stream(
    messages: list[ModelMessage], agent_info: AgentInfo
) -> AsyncIterator[str]:
    for chunk in ["Hel", "lo, ", "world"]:
        yield chunk


def _build_agent_deps(**session_store_overrides: object) -> AgentDeps:
    return AgentDeps(
        http_client=httpx.AsyncClient(),
        settings=build_test_settings(),
        session_store=InMemorySessionStore(**session_store_overrides),
    )


class TestTextOnlyRun:
    """A plain text response maps to StepStarted, Token(s), Completed."""

    @pytest.mark.asyncio
    async def test_emits_step_started_tokens_then_completed(self) -> None:
        """No tool calls involved: single model step, then completion."""
        agent: Agent[AgentDeps, str] = Agent(
            model=FunctionModel(stream_function=_text_only_stream),
            deps_type=AgentDeps,
            output_type=str,
        )
        deps = _build_agent_deps()
        chat_request = ChatRequest(message="Hi")

        events = [
            e
            async for e in _agent_event_stream(
                agent, chat_request, deps, history=[], settings=deps.settings
            )
        ]

        assert events[0] == StepStarted()
        assert events[-1] == Completed()
        tokens = [e for e in events if isinstance(e, Token)]
        assert "".join(t.content for t in tokens) == "Hello, world"

    @pytest.mark.asyncio
    async def test_no_session_id_skips_save(self) -> None:
        """Without a session_id, save_history is never called."""
        agent: Agent[AgentDeps, str] = Agent(
            model=FunctionModel(stream_function=_text_only_stream),
            deps_type=AgentDeps,
            output_type=str,
        )
        deps = _build_agent_deps()
        chat_request = ChatRequest(message="Hi")

        [
            e
            async for e in _agent_event_stream(
                agent, chat_request, deps, history=[], settings=deps.settings
            )
        ]

        assert await deps.session_store.get_history("anything") == []


async def _tool_call_then_text_stream(
    messages: list[ModelMessage], agent_info: AgentInfo
) -> AsyncIterator[object]:
    """Issue one tool call, then answer in text once its return is in history.

    Shared by `TestToolCallRun` and `TestUsageLimitEnforcement` - any test
    needing a stream that forces a second model request (e.g. to trip
    `request_limit`) reuses this rather than a near-duplicate.
    """
    has_tool_return = any(
        isinstance(m, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in m.parts)
        for m in messages
    )
    if not has_tool_return:
        yield {
            0: DeltaToolCall(
                name="get_weather",
                json_args='{"city": "Paris"}',
                tool_call_id="call_1",
            )
        }
    else:
        for chunk in ["It's ", "sunny."]:
            yield chunk


def _build_tool_calling_agent() -> Agent[AgentDeps, str]:
    """Build the agent driven by `_tool_call_then_text_stream`, `get_weather` registered."""
    agent: Agent[AgentDeps, str] = Agent(
        model=FunctionModel(stream_function=_tool_call_then_text_stream),
        deps_type=AgentDeps,
        output_type=str,
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f"Weather in {city}: sunny"

    return agent


class TestToolCallRun:
    """A tool-calling response maps ToolCalled between two StepStarted steps."""

    @pytest.mark.asyncio
    async def test_emits_tool_called_between_two_model_steps(self) -> None:
        """ToolCalled carries the tool name and a non-raw args summary."""
        agent = _build_tool_calling_agent()
        deps = _build_agent_deps()
        chat_request = ChatRequest(message="What's the weather in Paris?")

        events = [
            e
            async for e in _agent_event_stream(
                agent, chat_request, deps, history=[], settings=deps.settings
            )
        ]

        step_started_count = sum(1 for e in events if isinstance(e, StepStarted))
        assert step_started_count == 2

        tool_called = [e for e in events if isinstance(e, ToolCalled)]
        assert len(tool_called) == 1
        assert tool_called[0].name == "get_weather"
        assert "Paris" in tool_called[0].args_summary

        assert events[-1] == Completed()
        tokens = [e for e in events if isinstance(e, Token)]
        assert "".join(t.content for t in tokens) == "It's sunny."


class TestSessionSaveBeforeCompleted:
    """Session history is saved before Completed, and never after a save failure."""

    @pytest.mark.asyncio
    async def test_successful_save_is_followed_by_completed(self) -> None:
        """A successful save_history() results in a normal Completed event."""
        agent: Agent[AgentDeps, str] = Agent(
            model=FunctionModel(stream_function=_text_only_stream),
            deps_type=AgentDeps,
            output_type=str,
        )
        deps = _build_agent_deps()
        chat_request = ChatRequest(message="Hi", session_id="sess-1")

        events = [
            e
            async for e in _agent_event_stream(
                agent, chat_request, deps, history=[], settings=deps.settings
            )
        ]

        assert events[-1] == Completed()
        saved = await deps.session_store.get_history("sess-1")
        assert len(saved) > 0

    @pytest.mark.asyncio
    async def test_save_failure_yields_error_not_completed(self) -> None:
        """A save_history() failure yields a terminal Error, never a Completed event."""

        class _FailingSessionStore(InMemorySessionStore):
            async def save_history(self, session_id: str, messages: Sequence[ModelMessage]) -> None:
                raise RuntimeError("simulated session store backend failure")

        agent: Agent[AgentDeps, str] = Agent(
            model=FunctionModel(stream_function=_text_only_stream),
            deps_type=AgentDeps,
            output_type=str,
        )
        deps = AgentDeps(
            http_client=httpx.AsyncClient(),
            settings=build_test_settings(),
            session_store=_FailingSessionStore(),
        )
        chat_request = ChatRequest(message="Hi", session_id="sess-1")

        events = [
            e
            async for e in _agent_event_stream(
                agent, chat_request, deps, history=[], settings=deps.settings
            )
        ]

        assert events[-1] == Error(message="Failed to save session")
        assert Completed() not in events


class _FakeRequest:
    """Minimal stand-in for fastapi.Request exposing only is_disconnected()."""

    async def is_disconnected(self) -> bool:
        return False


class TestUsageLimitEnforcement:
    """Req 9.4: the stream path enforces and reports usage limits like the chat path.

    Mirrors `tests/unit/agents/test_guardrails.py::TestRunGuardedUsageObject` and
    `tests/unit/api/v1/test_agent_endpoints.py::test_chat_sets_tool_calls_limit_from_settings`
    for the `Agent.iter()`-driven stream path (Task 6.1's non-streaming
    counterpart is `run_guarded()`; this is `_agent_event_stream()`).
    """

    @pytest.mark.asyncio
    async def test_tool_calls_limit_reaches_agent_iter(self) -> None:
        """The stream's UsageLimits carries tool_calls_limit, matching the chat path."""
        agent = _build_tool_calling_agent()
        deps = _build_agent_deps()
        chat_request = ChatRequest(message="What's the weather in Paris?")

        with patch.object(agent, "iter", wraps=agent.iter) as spy_iter:
            [
                e
                async for e in _agent_event_stream(
                    agent, chat_request, deps, history=[], settings=deps.settings
                )
            ]

        spy_iter.assert_called_once()
        _, kwargs = spy_iter.call_args
        assert kwargs["usage_limits"].tool_calls_limit == deps.settings.usage_tool_calls_limit

    @pytest.mark.asyncio
    async def test_passes_caller_owned_usage_object_to_agent_iter(self) -> None:
        """A completed run passes a real, populated RunUsage to agent.iter()."""
        agent: Agent[AgentDeps, str] = Agent(
            model=FunctionModel(stream_function=_text_only_stream),
            deps_type=AgentDeps,
            output_type=str,
        )
        deps = _build_agent_deps()
        chat_request = ChatRequest(message="Hi")

        with patch.object(agent, "iter", wraps=agent.iter) as spy_iter:
            [
                e
                async for e in _agent_event_stream(
                    agent, chat_request, deps, history=[], settings=deps.settings
                )
            ]

        spy_iter.assert_called_once()
        _, kwargs = spy_iter.call_args
        usage = kwargs["usage"]
        assert isinstance(usage, RunUsage)
        assert usage.requests >= 1

    @pytest.mark.asyncio
    async def test_usage_object_is_mutated_in_place_even_on_a_budget_stop(self) -> None:
        """A caller-supplied RunUsage reflects real counters even when the native check raises.

        Mirrors `test_guardrails.py`'s identically-named test for `run_guarded()`.
        """
        agent = _build_tool_calling_agent()
        deps = _build_agent_deps()
        settings = build_test_settings(usage_request_limit=1)
        chat_request = ChatRequest(message="What's the weather in Paris?")
        usage = RunUsage()

        with pytest.raises(UsageLimitExceeded):
            async for _ in _agent_event_stream(
                agent, chat_request, deps, history=[], settings=settings, usage=usage
            ):
                pass

        assert usage.requests >= 1

    @pytest.mark.asyncio
    async def test_usage_limit_exceeded_reports_the_mutated_snapshot_across_the_task_boundary(
        self,
    ) -> None:
        """The consumer, running outside the producer's task, sees the same live RunUsage.

        `_agent_event_stream` (the producer) runs inside `_drive_to_queue`'s own
        task; `_run_with_lifecycle_guards` (the consumer) reads
        `producer.exception()` from the caller's task instead. This proves the
        `usage` object plumbed into both carries real, mutated counters across
        that task boundary, not just that it's syntactically wired through.
        """
        agent = _build_tool_calling_agent()
        deps = _build_agent_deps()
        settings = build_test_settings(usage_request_limit=1)
        chat_request = ChatRequest(message="What's the weather in Paris?")
        usage = RunUsage()

        agen = _agent_event_stream(
            agent, chat_request, deps, history=[], settings=settings, usage=usage
        )
        wires = [
            w async for w in _run_with_lifecycle_guards(_FakeRequest(), agen, settings, usage=usage)
        ]

        parsed = parse_sse_events("".join(wires))
        error = parsed[-1]
        assert isinstance(error, Error)
        assert "max_iterations" in error.message
        assert usage.requests >= 1
        assert f"requests={usage.requests}" in error.message
        assert f"tool_calls={usage.tool_calls}" in error.message
        assert f"total_tokens={usage.total_tokens}" in error.message
