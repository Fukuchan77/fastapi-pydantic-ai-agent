"""Integration tests: NativeOutput streaming does not leak raw JSON (Task 7.3).

Extended boundary per Req 10.2. When `build_chat_agent` wraps output in
`NativeOutput(ChatOutput)`, the model's streamed text *is* the JSON envelope
(e.g. `{"reply": "Hel`, `lo"}`).
`_agent_event_stream()` must not forward those raw fragments as `Token`
events - instead it emits the parsed `ChatOutput.reply` once, at the `End`
node, before `Completed`.
"""

from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile

from app.agents.chat_agent import build_chat_agent
from app.agents.deps import AgentDeps
from app.api.v1._stream import _agent_event_stream
from app.models.agent import ChatRequest
from app.patterns.sse import Completed
from app.patterns.sse import Token
from app.stores.session_store import InMemorySessionStore
from tests.conftest import build_test_settings


async def _native_json_stream(
    messages: list[ModelMessage], agent_info: AgentInfo
) -> AsyncIterator[str]:
    for chunk in ['{"repl', 'y": "Hello, world"}']:
        yield chunk


def _build_agent_deps() -> AgentDeps:
    return AgentDeps(
        http_client=httpx.AsyncClient(),
        settings=build_test_settings(),
        session_store=InMemorySessionStore(),
    )


class TestNativeOutputStreamDoesNotLeakJson:
    """The raw JSON envelope never reaches the client as streamed text."""

    @pytest.mark.asyncio
    async def test_emits_parsed_reply_once_instead_of_raw_json_deltas(self) -> None:
        """Only the parsed reply text is emitted, as a single Token before Completed."""
        agent = build_chat_agent(
            model=FunctionModel(
                stream_function=_native_json_stream,
                profile=ModelProfile(supports_json_schema_output=True),
            ),
            settings=build_test_settings(),
        )
        deps = _build_agent_deps()
        chat_request = ChatRequest(message="Hi")

        events = [e async for e in _agent_event_stream(agent, chat_request, deps, history=[])]

        tokens = [e for e in events if isinstance(e, Token)]
        assert tokens == [Token(content="Hello, world")]
        assert events[-1] == Completed()

    @pytest.mark.asyncio
    async def test_no_token_contains_raw_json_syntax(self) -> None:
        """No individual Token's content contains the JSON envelope's syntax."""
        agent = build_chat_agent(
            model=FunctionModel(
                stream_function=_native_json_stream,
                profile=ModelProfile(supports_json_schema_output=True),
            ),
            settings=build_test_settings(),
        )
        deps = _build_agent_deps()
        chat_request = ChatRequest(message="Hi")

        events = [e async for e in _agent_event_stream(agent, chat_request, deps, history=[])]

        tokens = [e for e in events if isinstance(e, Token)]
        assert all('"reply"' not in t.content and "{" not in t.content for t in tokens)
