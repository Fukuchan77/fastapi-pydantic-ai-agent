"""Regression test: a structured False verdict widens retrieval regardless of wording.

Locks in the L5.2 rework of `LLMCallMixin._evaluate_relevance`
(`app/workflows/rag_llm.py`): the sufficiency decision is the validated
`RelevanceVerdict.sufficient` field, not prose. The naive `"relevant" in
response` substring check this rework deleted would have misread the
historical rationale text below ("irrelevant", "not relevant") as a positive
match, because both contain the substring "relevant" - this test drives the
eval agent's structured tool-call output directly (rather than plain text)
so it exercises the actual `sufficient` field the retry decision reads, not
pydantic-ai's exhausted-output-retry fallback that a plain-text mock would
hit instead (see `tests/unit/workflows/test_structured_relevance.py`'s
`test_exhausted_output_retry_budget_yields_insufficient_not_an_error`, which
covers that separate failure mode).
"""

from unittest.mock import AsyncMock

import pytest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.models.rag import RetrievedHit
from app.workflows.corrective_rag import CorrectiveRAGWorkflow
from tests.conftest import build_test_settings


def _hit(chunk_id: str, score: float, text: str = "some chunk text") -> RetrievedHit:
    return RetrievedHit(chunk_id=chunk_id, text=text, score=score)


def _verdict_response(info: AgentInfo, *, sufficient: bool, rationale: str) -> ModelResponse:
    """Build the tool-call response the structured-output eval agent expects."""
    tool = info.output_tools[0]
    return ModelResponse(
        parts=[ToolCallPart(tool.name, {"sufficient": sufficient, "rationale": rationale})]
    )


def _says_irrelevant(messages: list, info: AgentInfo) -> ModelResponse:
    """Eval agent: sufficient=False with a rationale that is just "irrelevant".

    Synth agent (no output_tools): plain text, since its output_type is str.
    """
    if info.output_tools:
        return _verdict_response(info, sufficient=False, rationale="irrelevant")
    return ModelResponse(parts=[TextPart(content="synthesized answer")])


def _says_not_relevant(messages: list, info: AgentInfo) -> ModelResponse:
    """Eval agent: sufficient=False with a free-form "not relevant" rationale."""
    if info.output_tools:
        return _verdict_response(
            info, sufficient=False, rationale="The chunks are not relevant to the query."
        )
    return ModelResponse(parts=[TextPart(content="synthesized answer")])


def _mock_vector_store() -> AsyncMock:
    """Build a vector-store double with a real int generation.

    Task 3.2: the generation-keyed cache (task 3.4) reads store.generation
    directly on every store CorrectiveRAGWorkflow.run() receives; an
    unconfigured AsyncMock attribute would leak a Mock object into the
    cache key instead of a real int.
    """
    store = AsyncMock()
    store.generation = 0
    return store


class TestMockVectorStoreGeneration:
    """Guard against an unconfigured Mock attribute leaking into the cache key.

    Both inline vector-store doubles in this module must carry a real int
    generation (task 3.2), so the generation-keyed cache (task 3.4) never
    keys on an unconfigured Mock attribute — this file's two stores exercise
    the same cached run() path CorrectiveRAGWorkflow uses.
    """

    def test_generation_is_a_real_int(self) -> None:
        """Guard against an unconfigured AsyncMock leaking into the cache key."""
        vector_store = _mock_vector_store()
        assert isinstance(vector_store.generation, int)


class TestNegativeVerdictTriggersRetry:
    """A structured False verdict must retry, even when the rationale contains "relevant"."""

    @pytest.mark.asyncio
    async def test_irrelevant_verdict_widens_retrieval(self) -> None:
        """The rationale "irrelevant" contains "relevant" but the verdict field is False."""
        vector_store = _mock_vector_store()
        vector_store.query_with_scores.return_value = [_hit("memory::0000", 0.9)]

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=build_test_settings(rag_initial_k=2, rag_widened_k=4, rag_cache_ttl=0),
            llm_model=FunctionModel(_says_irrelevant),
        )

        await workflow.run(query="test", max_retries=2)

        assert vector_store.query_with_scores.call_count == 2, (
            "a sufficient=False verdict must widen the search, regardless of rationale wording"
        )
        second_kwargs = vector_store.query_with_scores.call_args_list[1].kwargs
        assert second_kwargs["top_k"] == 4

    @pytest.mark.asyncio
    async def test_not_relevant_verdict_widens_retrieval(self) -> None:
        """A free-form rationale containing "relevant" must not flip a False verdict."""
        vector_store = _mock_vector_store()
        vector_store.query_with_scores.return_value = [_hit("memory::0000", 0.9)]

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=build_test_settings(rag_initial_k=2, rag_widened_k=4, rag_cache_ttl=0),
            llm_model=FunctionModel(_says_not_relevant),
        )

        await workflow.run(query="test", max_retries=2)

        assert vector_store.query_with_scores.call_count == 2, (
            "a sufficient=False verdict must widen the search, regardless of rationale wording"
        )
        second_kwargs = vector_store.query_with_scores.call_args_list[1].kwargs
        assert second_kwargs["top_k"] == 4
