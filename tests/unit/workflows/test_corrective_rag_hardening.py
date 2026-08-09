"""Unit tests for Corrective RAG hardening: early-stop, widen-k, and degraded return.

Covers AC 3.1 (skip LLM on zero hits), 3.2 (widen k on retry), 3.6 (degraded
grounded-subset answer at the retry limit instead of raising), and 3.8
(deterministic citation ordering).
"""

from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_litellm import LiteLLMModel

from app.agents.chat_agent import build_model
from app.config import Settings
from app.models.rag import RetrievedHit
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


def _hit(chunk_id: str, score: float, text: str = "some chunk text") -> RetrievedHit:
    return RetrievedHit(chunk_id=chunk_id, text=text, score=score)


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "api_key": SecretStr("test-api-key-12345678"),
        "llm_model": "openai:gpt-4o",
        "rag_cache_ttl": 0,  # disable caching so every call re-executes the workflow
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _always_insufficient(messages: list, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content="insufficient")])


def _always_relevant(messages: list, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content="relevant")])


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

    Every vector-store double this module builds must carry a real int
    generation (task 3.2), so the generation-keyed cache (task 3.4) never
    keys on an unconfigured Mock attribute.
    """

    def test_generation_is_a_real_int(self) -> None:
        """Guard against an unconfigured AsyncMock leaking into the cache key."""
        vector_store = _mock_vector_store()
        assert isinstance(vector_store.generation, int)


class TestZeroHitsEarlyStop:
    """AC 3.1: zero hits skip the LLM entirely and terminate early."""

    @pytest.mark.asyncio
    async def test_zero_hits_skips_llm_and_returns_no_context(self) -> None:
        """When query_with_scores returns no hits, the LLM must never be called."""
        vector_store = _mock_vector_store()
        vector_store.query_with_scores.return_value = []

        eval_calls = 0

        def counting_model(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal eval_calls
            eval_calls += 1
            return ModelResponse(parts=[TextPart(content="relevant")])

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=_settings(),
            llm_model=FunctionModel(counting_model),
        )

        result = await workflow.run(query="anything", max_retries=3)

        assert eval_calls == 0, "LLM must not be called when zero hits are retrieved"
        assert result["context_found"] is False
        assert result["citations"] == []
        assert vector_store.query_with_scores.call_count == 1, (
            "Zero hits should terminate early without retrying"
        )


class TestWidenKOnRetry:
    """AC 3.2: retry after insufficient grading widens k from rag_initial_k to rag_widened_k."""

    @pytest.mark.asyncio
    async def test_initial_search_uses_rag_initial_k(self) -> None:
        """The first search attempt should use settings.rag_initial_k."""
        vector_store = _mock_vector_store()
        vector_store.query_with_scores.return_value = [_hit("memory::0000", 0.9)]

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=_settings(rag_initial_k=2, rag_widened_k=4),
            llm_model=FunctionModel(_always_relevant),
        )

        await workflow.run(query="test", max_retries=1)

        first_call_kwargs = vector_store.query_with_scores.call_args_list[0].kwargs
        assert first_call_kwargs["top_k"] == 2

    @pytest.mark.asyncio
    async def test_retry_search_uses_rag_widened_k(self) -> None:
        """A retry search (after an insufficient grading) should use settings.rag_widened_k."""
        vector_store = _mock_vector_store()
        vector_store.query_with_scores.return_value = [_hit("memory::0000", 0.9)]

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=_settings(rag_initial_k=2, rag_widened_k=4),
            llm_model=FunctionModel(_always_insufficient),
        )

        await workflow.run(query="test", max_retries=2)

        assert vector_store.query_with_scores.call_count == 2
        first_kwargs = vector_store.query_with_scores.call_args_list[0].kwargs
        second_kwargs = vector_store.query_with_scores.call_args_list[1].kwargs
        assert first_kwargs["top_k"] == 2
        assert second_kwargs["top_k"] == 4


class TestDegradedReturnAtRetryLimit:
    """AC 3.6: at the retry limit, return a degraded grounded-subset answer, not an error."""

    @pytest.mark.asyncio
    async def test_degraded_answer_synthesized_when_hits_exist_at_retry_limit(self) -> None:
        """Retries exhausted with hits present should synthesize an answer and cite them."""
        vector_store = _mock_vector_store()
        vector_store.query_with_scores.return_value = [
            _hit("memory::0000", 0.9, text="Relevant-ish content"),
        ]

        synth_calls = 0

        def model_fn(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal synth_calls
            last_text = messages[-1].parts[0].content if messages else ""
            if "using the following context" in last_text.lower():
                synth_calls += 1
                return ModelResponse(parts=[TextPart(content="Best-effort degraded answer")])
            return ModelResponse(parts=[TextPart(content="insufficient")])

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=_settings(),
            llm_model=FunctionModel(model_fn),
        )

        result = await workflow.run(query="test", max_retries=1)

        assert result["context_found"] is False
        assert synth_calls == 1, "Synthesis must be attempted from the grounded subset"
        assert result["answer"] == "Best-effort degraded answer"
        assert len(result["citations"]) == 1
        assert result["citations"][0].chunk_id == "memory::0000"

    @pytest.mark.asyncio
    async def test_never_found_any_hits_still_uses_canned_message(self) -> None:
        """Zero hits ever (not just at the retry limit) keeps the graceful no-context message."""
        vector_store = _mock_vector_store()
        vector_store.query_with_scores.return_value = []

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=_settings(),
            llm_model=FunctionModel(_always_relevant),
        )

        result = await workflow.run(query="test", max_retries=2)

        assert result["context_found"] is False
        assert "couldn't find relevant information" in result["answer"].lower()
        assert result["citations"] == []


class TestDeterministicCitationOrdering:
    """AC 3.8: citations are ordered deterministically by (-score, chunk_id)."""

    @pytest.mark.asyncio
    async def test_citations_are_ordered_by_score_descending(self) -> None:
        """Higher-score hits should appear first in the citations list."""
        vector_store = _mock_vector_store()
        vector_store.query_with_scores.return_value = [
            _hit("b::0000", 0.2),
            _hit("a::0000", 0.9),
            _hit("c::0000", 0.5),
        ]

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=_settings(),
            llm_model=FunctionModel(_always_relevant),
        )

        result = await workflow.run(query="test", max_retries=1)

        assert [c.chunk_id for c in result["citations"]] == ["a::0000", "c::0000", "b::0000"]

    @pytest.mark.asyncio
    async def test_tied_scores_break_ties_by_chunk_id_ascending(self) -> None:
        """Equal-score hits should be ordered by chunk_id ascending for stability."""
        vector_store = _mock_vector_store()
        vector_store.query_with_scores.return_value = [
            _hit("z::0000", 0.5),
            _hit("a::0000", 0.5),
        ]

        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=_settings(),
            llm_model=FunctionModel(_always_relevant),
        )

        result = await workflow.run(query="test", max_retries=1)

        assert [c.chunk_id for c in result["citations"]] == ["a::0000", "z::0000"]


class TestDefaultModelResolution:
    """Req 11.1/11.2: no model supplied still resolves via the chat builder.

    With no explicit `llm_model`, resolution must go through the same
    builder the chat path uses, never a raw `"provider:model"` settings
    string passed straight to an `Agent` constructor.
    """

    def test_no_model_supplied_resolves_through_build_model(self) -> None:
        """The `None` branch must build via `build_model`, not `llm_settings.llm_model`."""
        settings = _settings(
            llm_model="ollama:granite3.3",
            llm_base_url="http://localhost:11434",
        )
        expected_model_name = build_model(settings).model_name

        workflow = CorrectiveRAGWorkflow(
            vector_store=_mock_vector_store(),
            llm_settings=settings,
        )

        assert isinstance(workflow._eval_agent.model, LiteLLMModel)
        assert workflow._eval_agent.model.model_name == expected_model_name
        assert isinstance(workflow._synth_agent.model, LiteLLMModel)
        assert workflow._synth_agent.model.model_name == expected_model_name

    def test_explicit_model_override_is_unaffected(self) -> None:
        """An explicitly-injected model (e.g. a test double) must still win."""
        injected = FunctionModel(_always_relevant)

        workflow = CorrectiveRAGWorkflow(
            vector_store=_mock_vector_store(),
            llm_settings=_settings(),
            llm_model=injected,
        )

        assert workflow._eval_agent.model is injected
        assert workflow._synth_agent.model is injected
