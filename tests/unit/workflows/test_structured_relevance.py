"""Structured RAG relevance evaluation honours a validated verdict, not prose (Req 10.8).

Locks in the L5.2 rework of `LLMCallMixin._evaluate_relevance`
(`app/workflows/rag_llm.py`): the eval agent's output is a validated
`RelevanceVerdict`, consumed directly with no `.strip().lower()` text
parsing, and both failure modes - a per-call timeout and an exhausted
output-retry budget - degrade to a safe insufficient verdict instead of
raising or misreading the model's words.
"""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.models.rag import RetrievedHit
from app.workflows.corrective_rag import CorrectiveRAGWorkflow
from tests.conftest import build_test_settings


def _hit(chunk_id: str, score: float = 0.9, text: str = "some chunk text") -> RetrievedHit:
    return RetrievedHit(chunk_id=chunk_id, text=text, score=score)


def _verdict_response(info: AgentInfo, *, sufficient: bool, rationale: str) -> ModelResponse:
    """Build the tool-call response a structured-output agent expects."""
    tool = info.output_tools[0]
    return ModelResponse(
        parts=[ToolCallPart(tool.name, {"sufficient": sufficient, "rationale": rationale})]
    )


class TestVerdictHonouredWithoutTextParsing:
    """A verdict whose rationale would defeat substring parsing must still be honoured."""

    @pytest.mark.asyncio
    async def test_insufficient_verdict_with_relevant_worded_rationale_still_widens_search(
        self,
    ) -> None:
        """A rationale that *mentions* "relevant" must not flip an insufficient verdict.

        The deleted cascade misread "irrelevant"/"not relevant" via substring
        matching. This asserts the structured field - not prose - decides:
        `sufficient=False` must widen the search even though the rationale
        text contains "relevant".
        """
        vector_store_calls = 0

        def eval_model(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal vector_store_calls
            if info.output_tools:
                return _verdict_response(
                    info,
                    sufficient=False,
                    rationale="Somewhat relevant, but not enough to answer the query.",
                )
            return ModelResponse(parts=[TextPart(content="synthesized answer")])

        from unittest.mock import AsyncMock

        vector_store = AsyncMock()
        vector_store.generation = 0
        vector_store.query_with_scores.return_value = [_hit("memory::0000")]

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=build_test_settings(rag_initial_k=2, rag_widened_k=4, rag_cache_ttl=0),
            llm_model=FunctionModel(eval_model),
        )

        await workflow.run(query="test", max_retries=2)

        assert vector_store.query_with_scores.call_count == 2, (
            "sufficient=False must widen the search, regardless of rationale wording"
        )
        second_kwargs = vector_store.query_with_scores.call_args_list[1].kwargs
        assert second_kwargs["top_k"] == 4

    @pytest.mark.asyncio
    async def test_sufficient_verdict_with_insufficient_worded_rationale_skips_retry(
        self,
    ) -> None:
        """A rationale that *mentions* "insufficient" must not flip a sufficient verdict."""

        def eval_model(messages: list, info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                return _verdict_response(
                    info,
                    sufficient=True,
                    rationale="Not insufficient at all - directly answers the query.",
                )
            return ModelResponse(parts=[TextPart(content="synthesized answer")])

        from unittest.mock import AsyncMock

        vector_store = AsyncMock()
        vector_store.generation = 0
        vector_store.query_with_scores.return_value = [_hit("memory::0000")]

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=build_test_settings(rag_initial_k=2, rag_widened_k=4, rag_cache_ttl=0),
            llm_model=FunctionModel(eval_model),
        )

        result = await workflow.run(query="test", max_retries=2)

        assert vector_store.query_with_scores.call_count == 1, (
            "sufficient=True must skip the retry, regardless of rationale wording"
        )
        assert result["context_found"] is True


class TestFailureModesDegradeToInsufficientVerdict:
    """Both a per-call timeout and exhausted output retries yield insufficient, not an error."""

    @pytest.mark.asyncio
    async def test_timeout_yields_insufficient_verdict_without_consuming_a_transient_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A per-call timeout falls back to insufficient and does not retry (Req 10.5)."""
        eval_call_count = 0

        async def slow_eval_fast_synth_model(messages: list, info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                nonlocal eval_call_count
                eval_call_count += 1
                await asyncio.sleep(10)
                return _verdict_response(info, sufficient=True, rationale="too slow to matter")
            return ModelResponse(parts=[TextPart(content="degraded synthesis answer")])

        settings = build_test_settings(
            llm_agent_timeout=5,
            llm_retry_max_attempts=3,
            rag_cache_ttl=0,
        )
        workflow = CorrectiveRAGWorkflow(
            vector_store=_stub_store([_hit("memory::0000")]),
            llm_settings=settings,
            llm_model=FunctionModel(slow_eval_fast_synth_model),
        )

        result = await workflow.run(query="test", max_retries=1)

        assert eval_call_count == 1, "a timeout must not consume the transient-retry budget"
        assert result["context_found"] is False

    @pytest.mark.asyncio
    async def test_exhausted_output_retry_budget_yields_insufficient_not_an_error(self) -> None:
        """An unparseable reply that exhausts pydantic-ai's output retries yields insufficient.

        The eval agent never calls its output tool, so pydantic-ai's own
        `retries={"output": ...}` budget (distinct from `_run_agent_with_retry`'s
        transient-error loop) is exhausted inside the single `agent.run()` call.
        """

        def never_calls_tool(messages: list, info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content="I refuse to call any tool.")])

        workflow = CorrectiveRAGWorkflow(
            vector_store=_stub_store([_hit("memory::0000")]),
            llm_settings=build_test_settings(rag_cache_ttl=0),
            llm_model=FunctionModel(never_calls_tool),
        )

        result = await workflow.run(query="test", max_retries=1)

        assert result["context_found"] is False
        assert "answer" in result


def _stub_store(hits: list[RetrievedHit]):
    from unittest.mock import AsyncMock

    store = AsyncMock()
    store.generation = 0
    store.query_with_scores.return_value = hits
    return store
