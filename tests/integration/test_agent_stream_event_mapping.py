"""Integration tests for the pydantic-ai node/event -> typed SSE union mapping.

Exercises `_agent_event_stream()` (app/api/v1/_stream.py) against a real
`Agent` driven by `FunctionModel`, per Task 5's `Agent.iter()`-based mapping:
`ModelRequestNode` -> StepStarted, `PartDeltaEvent(TextPartDelta)` -> Token,
`FunctionToolCallEvent` -> ToolCalled, the `End` node -> session save (if a
session_id was given) then Completed.
"""

from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import DeltaToolCall
from pydantic_ai.models.function import FunctionModel

from app.agents.deps import AgentDeps
from app.api.v1._stream import _agent_event_stream
from app.models.agent import ChatRequest
from app.patterns.sse import Completed
from app.patterns.sse import Error
from app.patterns.sse import StepStarted
from app.patterns.sse import Token
from app.patterns.sse import ToolCalled
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


class TestToolCallRun:
    """A tool-calling response maps ToolCalled between two StepStarted steps."""

    @staticmethod
    async def _tool_call_then_text_stream(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[object]:
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

    def _build_agent(self) -> Agent[AgentDeps, str]:
        agent: Agent[AgentDeps, str] = Agent(
            model=FunctionModel(stream_function=self._tool_call_then_text_stream),
            deps_type=AgentDeps,
            output_type=str,
        )

        @agent.tool_plain
        def get_weather(city: str) -> str:
            return f"Weather in {city}: sunny"

        return agent

    @pytest.mark.asyncio
    async def test_emits_tool_called_between_two_model_steps(self) -> None:
        """ToolCalled carries the tool name and a non-raw args summary."""
        agent = self._build_agent()
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
            async def save_history(self, session_id: str, messages: list[ModelMessage]) -> None:
                raise ValueError("Too many messages")

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
