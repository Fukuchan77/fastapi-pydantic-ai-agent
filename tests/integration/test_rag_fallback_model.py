"""Integration test: RAG queries fail over across a `FallbackModel` chain.

Task 5.12, Req 3.5, 3.10. Named after `test_lifespan_llm_fallback.py`'s
lifespan-level fallback coverage; this test exercises the RAG path's own
fallback behaviour, which required `app.state.llm_model` to reach
`get_rag_workflow` at all (Req 3.1-3.3) - before that fix `/v1/rag/query`
always built a single model with no failover regardless of what chain
`app.state.llm_model` held.
"""

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow
from tests.conftest import build_test_settings


@pytest.mark.asyncio
async def test_rag_query_succeeds_when_primary_model_always_fails() -> None:
    """A RAG query must succeed via the fallback member when the primary always fails.

    Always raises a failover-eligible `ModelHTTPError` (Req 3.5), and the
    structured `RelevanceVerdict` output must validate under that fallback
    member (Req 3.10) - confirming `FallbackModel.profile` is never touched
    on the RAG path (Req 3.6), since that call would raise.
    """
    primary_call_count = 0

    def always_failing_primary(messages: list, agent_info: AgentInfo) -> ModelResponse:
        """Simulate a primary provider that is always down."""
        nonlocal primary_call_count
        primary_call_count += 1
        raise ModelHTTPError(status_code=503, model_name="primary-always-down")

    def healthy_fallback(messages: list, agent_info: AgentInfo) -> ModelResponse:
        """Answer either the eval agent's structured tool call or the synth agent's plain text.

        Discriminates the same way `tests/conftest.py`'s `simple_llm_function`
        does - by `agent_info.output_tools`, never by prompt text.
        """
        if agent_info.output_tools:
            tool = agent_info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool.name, {"sufficient": True, "rationale": "relevant"})]
            )
        return ModelResponse(parts=[TextPart(content="synthesized answer from fallback")])

    chain = FallbackModel(
        FunctionModel(always_failing_primary),
        FunctionModel(healthy_fallback),
    )

    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(
        [
            "Python is a programming language.",
            "FastAPI is a web framework for Python.",
        ]
    )

    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=build_test_settings(),
        llm_model=chain,
    )

    result = await workflow.run(query="What is FastAPI?")

    assert primary_call_count > 0, (
        "The primary member must have been tried at least once - otherwise "
        "this test proves nothing about failover."
    )
    assert result["context_found"] is True
    assert result["answer"] == "synthesized answer from fallback"
    assert result["citations"], "A grounded answer must carry at least one citation."
