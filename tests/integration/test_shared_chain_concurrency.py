"""Integration test: chat and RAG runs share one `FallbackModel` chain under concurrent entry.

Task 5.13, Req 3.2, 3.11. `FallbackModel.__aenter__`/`__aexit__` are
reference-counted (`_enter_lock` + `_entered_count`, both private) so a
single chain instance can be entered by more than one concurrent agent run.
Before Req 3.1-3.3's fix, RAG never read the injected `app.state.llm_model`
at all, so this refcounted path was only ever exercised by the chat agent
alone - never concurrently by two agents sharing the *same* chain object.

Deliberately injects a real `FallbackModel` (a chain of one is enough - the
refcounting lives on the wrapper, not on how many members it has), not the
plain `FunctionModel` the default `client` test fixture resolves to: that
would make this test pass while covering none of the refcount risk it
exists to cover (Req 3.11).
"""

import asyncio

import pytest
from httpx import ASGITransport
from httpx import AsyncClient
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile

from app.main import create_app
from tests.conftest import build_test_settings


@pytest.mark.asyncio
async def test_chat_and_rag_runs_succeed_concurrently_against_shared_chain() -> None:
    """A chat turn and a RAG query, run concurrently against one shared chain, must both succeed.

    They must also genuinely overlap in time, not merely both succeed in
    sequence. A two-party `asyncio.Event` barrier inside the shared model function
    proves the overlap deterministically: whichever run's model call
    arrives first blocks until the second run's model call has also
    started, so neither run's model call can complete before both are
    in flight simultaneously.
    """
    entered_count = 0
    entered_lock = asyncio.Lock()
    both_entered = asyncio.Event()

    async def shared_model_function(messages: list, agent_info: AgentInfo) -> ModelResponse:
        """Answer either agent, but only once a second concurrent caller has also arrived.

        Blocking until that happens is what proves the two runs overlapped.
        """
        nonlocal entered_count
        async with entered_lock:
            entered_count += 1
            if entered_count >= 2:
                both_entered.set()

        await asyncio.wait_for(both_entered.wait(), timeout=5.0)

        if agent_info.output_tools:
            tool = agent_info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool.name, {"sufficient": True, "rationale": "relevant"})]
            )
        return ModelResponse(parts=[TextPart(content="shared chain answer")])

    chain = FallbackModel(
        FunctionModel(
            shared_model_function,
            profile=ModelProfile(supports_json_schema_output=False),
        )
    )
    settings = build_test_settings()
    app = create_app(settings=settings, model=chain)
    auth_headers = {"X-API-Key": "test-api-key-12345"}

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert app.state.llm_model is chain

        await app.state.vector_store.add_documents(["FastAPI is a Python web framework."])

        chat_task = client.post(
            "/v1/agent/chat",
            json={"message": "Hello there"},
            headers=auth_headers,
        )
        rag_task = client.post(
            "/v1/rag/query",
            json={"query": "What is FastAPI?"},
            headers=auth_headers,
        )

        chat_response, rag_response = await asyncio.gather(chat_task, rag_task)

    assert entered_count >= 2, (
        "Both runs must have reached the shared model function - otherwise "
        "this test proves nothing about concurrent entry."
    )
    assert chat_response.status_code == 200, chat_response.text
    assert rag_response.status_code == 200, rag_response.text

    rag_body = rag_response.json()
    assert rag_body["context_found"] is True
    assert rag_body["citations"]
