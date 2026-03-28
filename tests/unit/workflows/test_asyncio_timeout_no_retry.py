"""Tests for Task 20.6: asyncio.TimeoutError should NOT trigger retries.

Task 20.6: Exclude asyncio.TimeoutError from LLM retry logic.
asyncio.TimeoutError is a subclass of TimeoutError, but it indicates
the LLM is consistently too slow, not a transient failure.
Retrying wastes time and resources.
"""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.config import get_settings
from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


@pytest.mark.asyncio
async def test_asyncio_timeout_error_does_not_retry_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that asyncio.TimeoutError in evaluation does NOT trigger retries.

    Task 20.6: asyncio.TimeoutError indicates the LLM is consistently too slow,
    not a transient failure. Retrying wastes time.
    """
    # Configure settings with multiple retry attempts
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "5")  # 5 second timeout
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "3")  # 3 retries configured
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY", "1")  # 1 second base delay
    get_settings.cache_clear()
    settings = get_settings()

    # Track how many times the model is called
    call_count = 0

    async def timeout_model(
        messages: list,
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        # Simulate timeout by sleeping longer than the timeout
        await asyncio.sleep(10)
        return ModelResponse(parts=[TextPart(content="relevant")])

    model = FunctionModel(timeout_model)

    # Create workflow
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Test document"])
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=model,
    )

    # Run workflow - should NOT retry after asyncio.TimeoutError
    start_time = asyncio.get_event_loop().time()
    result = await workflow.run(query="test query", max_retries=1)
    elapsed = asyncio.get_event_loop().time() - start_time

    # CRITICAL: Verify model was called ONLY ONCE (no retries)
    # Currently FAILS: model is called 3 times due to retry logic
    assert call_count == 1, f"Expected 1 call (no retries), but got {call_count} calls"

    # Verify it completed quickly (no retry delays)
    # 3 retries would add ~7 seconds (1 + 2 + 4 seconds exponential backoff)
    # Should complete in ~5 seconds (just the timeout)
    assert elapsed < 7, f"Expected ~5s (no retries), but took {elapsed:.1f}s"

    # Verify graceful fallback
    assert result["context_found"] is False
    assert "couldn't find relevant information" in result["answer"].lower()


@pytest.mark.asyncio
async def test_asyncio_timeout_error_does_not_retry_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that asyncio.TimeoutError in synthesis does NOT trigger retries.

    Task 20.6: asyncio.TimeoutError indicates the LLM is consistently too slow,
    not a transient failure. Retrying wastes time.
    """
    # Configure settings with multiple retry attempts
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "5")  # 5 second timeout
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "3")  # 3 retries configured
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY", "1")  # 1 second base delay
    get_settings.cache_clear()
    settings = get_settings()

    # Track how many times synthesis is called
    call_count = 0

    async def eval_then_timeout_synth_model(
        messages: list,
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        # First call is evaluation - return quickly
        if call_count == 1:
            return ModelResponse(parts=[TextPart(content="relevant")])

        # Subsequent calls are synthesis - timeout
        await asyncio.sleep(10)
        return ModelResponse(parts=[TextPart(content="This is the synthesized answer")])

    model = FunctionModel(eval_then_timeout_synth_model)

    # Create workflow
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Test document with content"])
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=model,
    )

    # Run workflow - should NOT retry synthesis after asyncio.TimeoutError
    start_time = asyncio.get_event_loop().time()
    result = await workflow.run(query="test query", max_retries=1)
    elapsed = asyncio.get_event_loop().time() - start_time

    # CRITICAL: Verify model was called TWICE (1 eval + 1 synth, no retries)
    # Currently FAILS: model is called 4 times (1 eval + 3 synth attempts)
    assert call_count == 2, f"Expected 2 calls (1 eval + 1 synth), but got {call_count} calls"

    # Verify it completed quickly (no retry delays)
    assert elapsed < 7, f"Expected ~5s (no retries), but took {elapsed:.1f}s"

    # Verify graceful error message
    assert "encountered an error" in result["answer"].lower()
