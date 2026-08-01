"""Regression test: "irrelevant"/"not relevant" must not be misread as "relevant".

A naive `"relevant" in response` substring check treats the LLM's negative
verdict as positive, because both "irrelevant" and "not relevant" contain the
substring "relevant". That skips the widened-retrieval retry CRAG relies on
for grounding, so this locks in the fix in `LLMCallMixin._evaluate_relevance`
(`app/workflows/rag_llm.py`).
"""

from unittest.mock import AsyncMock

import pytest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.models.rag import RetrievedHit
from app.workflows.corrective_rag import CorrectiveRAGWorkflow
from tests.conftest import build_test_settings


def _hit(chunk_id: str, score: float, text: str = "some chunk text") -> RetrievedHit:
    return RetrievedHit(chunk_id=chunk_id, text=text, score=score)


def _says_irrelevant(messages: list, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content="irrelevant")])


def _says_not_relevant(messages: list, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content="The chunks are not relevant to the query.")])


class TestNegativeVerdictTriggersRetry:
    """A negative verdict ("irrelevant"/"not relevant") must retry, not skip."""

    @pytest.mark.asyncio
    async def test_irrelevant_verdict_widens_retrieval(self) -> None:
        """The verdict "irrelevant" contains "relevant" but must count as insufficient."""
        vector_store = AsyncMock()
        vector_store.query_with_scores.return_value = [_hit("memory::0000", 0.9)]

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=build_test_settings(rag_initial_k=2, rag_widened_k=4, rag_cache_ttl=0),
            llm_model=FunctionModel(_says_irrelevant),
        )

        await workflow.run(query="test", max_retries=2)

        assert vector_store.query_with_scores.call_count == 2, (
            "an 'irrelevant' verdict was misread as relevant and skipped the retry"
        )
        second_kwargs = vector_store.query_with_scores.call_args_list[1].kwargs
        assert second_kwargs["top_k"] == 4

    @pytest.mark.asyncio
    async def test_not_relevant_verdict_widens_retrieval(self) -> None:
        """A free-form "not relevant" sentence must also count as insufficient."""
        vector_store = AsyncMock()
        vector_store.query_with_scores.return_value = [_hit("memory::0000", 0.9)]

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=build_test_settings(rag_initial_k=2, rag_widened_k=4, rag_cache_ttl=0),
            llm_model=FunctionModel(_says_not_relevant),
        )

        await workflow.run(query="test", max_retries=2)

        assert vector_store.query_with_scores.call_count == 2, (
            "a 'not relevant' verdict was misread as relevant and skipped the retry"
        )
