"""`Settings.rag_prompt_max_chars` governs prompt truncation, not a hardcoded literal (X-7).

Before this setting existed, `max_chars=15000` was hardcoded at three call
sites (`rag_llm.py._evaluate_relevance`/`_synthesize_answer`,
`corrective_rag.py`'s synthesize step). These tests prove the configured
value - not the old literal - is what actually governs truncation, and that
`PromptBuildingMixin._truncate_chunks`/`_truncate_hits` themselves still
default to 15000 for any direct caller that passes no `max_chars` (Stage 0
behavior, unchanged for anyone calling them directly).
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
from app.workflows.rag_prompts import PromptBuildingMixin
from tests.conftest import build_test_settings


_CHUNK = "x" * 600  # deliberately far larger than the small max_chars settings used below


class TestTruncateChunksDefault:
    """`_truncate_chunks` itself still defaults to 15000 for a direct caller."""

    def test_default_max_chars_is_15000(self) -> None:
        """A call with no `max_chars` argument truncates at 15000 chars."""
        mixin = PromptBuildingMixin()
        chunks = [_CHUNK] * 40  # 20000 chars total, over the 15000 default

        result = mixin._truncate_chunks(chunks)

        assert sum(len(c) for c in result) <= 15000

    def test_explicit_max_chars_overrides_the_default(self) -> None:
        """An explicit `max_chars` argument overrides the 15000 default."""
        mixin = PromptBuildingMixin()
        chunks = [_CHUNK, _CHUNK]

        result = mixin._truncate_chunks(chunks, max_chars=20)

        # Each chunk alone exceeds 20 chars, so the "always keep at least
        # one" floor applies: exactly the first chunk survives.
        assert result == [_CHUNK]


class TestTruncateHitsDefault:
    """`_truncate_hits` itself still defaults to 15000 for a direct caller."""

    def test_explicit_max_chars_overrides_the_default(self) -> None:
        """An explicit `max_chars` argument overrides the 15000 default."""
        mixin = PromptBuildingMixin()
        hits = [
            RetrievedHit(chunk_id="a::0", text=_CHUNK, score=1.0),
            RetrievedHit(chunk_id="a::1", text=_CHUNK, score=0.9),
        ]

        result = mixin._truncate_hits(hits, max_chars=20)

        assert [h.chunk_id for h in result] == ["a::0"]


class TestSettingGovernsEvaluateRelevance:
    """`_evaluate_relevance` truncates using `llm_settings.rag_prompt_max_chars`."""

    @pytest.mark.asyncio
    async def test_a_small_setting_truncates_chunks_a_large_one_would_not(self) -> None:
        """The same 2-chunk input is truncated under a small setting, not a large one."""
        captured_prompts: list[str] = []

        def eval_model(messages: list, info: AgentInfo) -> ModelResponse:
            captured_prompts.append(messages[-1].parts[-1].content)
            tool = info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool.name, {"sufficient": True, "rationale": "ok"})]
            )

        workflow = CorrectiveRAGWorkflow(
            vector_store=AsyncMock(),  # unused - called directly below
            llm_settings=build_test_settings(rag_prompt_max_chars=1000),
            llm_model=FunctionModel(eval_model),
        )

        await workflow._evaluate_relevance([_CHUNK, _CHUNK], "query")

        assert len(captured_prompts) == 1
        # With a 1000-char budget and 600-char chunks, only the first fits
        # (600 + 600 > 1000); the second chunk's content is entirely absent.
        assert captured_prompts[0].count(_CHUNK) == 1

    @pytest.mark.asyncio
    async def test_default_setting_keeps_both_small_chunks(self) -> None:
        """The default 15000-char setting is large enough that nothing here is cut."""
        captured_prompts: list[str] = []

        def eval_model(messages: list, info: AgentInfo) -> ModelResponse:
            captured_prompts.append(messages[-1].parts[-1].content)
            tool = info.output_tools[0]
            return ModelResponse(
                parts=[ToolCallPart(tool.name, {"sufficient": True, "rationale": "ok"})]
            )

        workflow = CorrectiveRAGWorkflow(
            vector_store=AsyncMock(),  # unused - called directly below
            llm_settings=build_test_settings(),  # rag_prompt_max_chars defaults to 15000
            llm_model=FunctionModel(eval_model),
        )

        await workflow._evaluate_relevance([_CHUNK, _CHUNK], "query")

        assert captured_prompts[0].count(_CHUNK) == 2


class TestSettingGovernsSynthesizeAnswer:
    """`_synthesize_answer` truncates using `llm_settings.rag_prompt_max_chars` too."""

    @pytest.mark.asyncio
    async def test_a_small_setting_truncates_hits(self) -> None:
        """Echoes the built prompt back as the "answer" to inspect it directly."""

        def echo_prompt_model(messages: list, info: AgentInfo) -> ModelResponse:
            prompt = messages[-1].parts[-1].content if messages else ""
            return ModelResponse(parts=[TextPart(content=prompt)])

        workflow = CorrectiveRAGWorkflow(
            vector_store=AsyncMock(),  # unused - called directly below
            llm_settings=build_test_settings(rag_prompt_max_chars=1000),
            llm_model=FunctionModel(echo_prompt_model),
        )
        hits = [
            RetrievedHit(chunk_id="a::0", text=_CHUNK, score=1.0),
            RetrievedHit(chunk_id="a::1", text=_CHUNK, score=0.9),
        ]

        echoed_prompt = await workflow._synthesize_answer(hits, "query")

        assert echoed_prompt.count(_CHUNK) == 1
