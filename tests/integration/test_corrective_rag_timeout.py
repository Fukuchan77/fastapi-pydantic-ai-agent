"""Tests for LLM agent timeout in Corrective RAG workflow.

Task 19.3: Verify timeout is applied to _eval_agent.run() and _synth_agent.run()
to prevent indefinite hangs when LLM provider is slow or unresponsive.
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
async def test_evaluation_times_out_after_llm_agent_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that LLM evaluation times out after llm_agent_timeout seconds.

    Task 19.3: Verify asyncio.wait_for() wraps _eval_agent.run() with timeout.
    """
    # Set short timeout for fast test execution (5 seconds minimum per Settings validation)
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "5")  # 5 second timeout (minimum allowed)
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "1")  # Disable retries for this test
    get_settings.cache_clear()  # Clear cache to pick up new env vars
    settings = get_settings()

    # Create slow LLM that takes 10 seconds (exceeds 5 second timeout)
    async def slow_eval_model(
        messages: list,
        info: AgentInfo,
    ) -> ModelResponse:
        await asyncio.sleep(10)  # Simulate slow LLM (exceeds timeout)
        return ModelResponse(parts=[TextPart(content="relevant")])

    slow_model = FunctionModel(slow_eval_model)

    # Create workflow with slow model
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Test document"])
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=slow_model,
    )

    # Run workflow - should timeout during evaluation step
    start_time = asyncio.get_event_loop().time()
    result = await workflow.run(query="test query", max_retries=1)
    elapsed = asyncio.get_event_loop().time() - start_time

    # Verify timeout occurred (should complete in ~5 seconds, not 10 seconds)
    assert elapsed < 7, f"Expected timeout at 5s, but took {elapsed:.1f}s"

    # Verify graceful fallback (evaluation timeout returns "insufficient")
    assert result["context_found"] is False
    assert "couldn't find relevant information" in result["answer"].lower()


@pytest.mark.asyncio
async def test_synthesis_times_out_after_llm_agent_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that LLM synthesis times out after llm_agent_timeout seconds.

    Task 19.3: Verify asyncio.wait_for() wraps _synth_agent.run() with timeout.
    """
    # Set short timeout for fast test execution (5 seconds minimum per Settings validation)
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "5")  # 5 second timeout (minimum allowed)
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "1")  # Disable retries for this test
    get_settings.cache_clear()  # Clear cache to pick up new env vars
    settings = get_settings()

    # Create model that's fast for eval but slow for synthesis
    call_count = 0

    async def eval_then_slow_synth_model(
        messages: list,
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        # First call is evaluation - return quickly
        if call_count == 1:
            return ModelResponse(parts=[TextPart(content="relevant")])

        # Second call is synthesis - take 10 seconds (exceeds timeout)
        await asyncio.sleep(10)
        return ModelResponse(parts=[TextPart(content="This is the synthesized answer")])

    model = FunctionModel(eval_then_slow_synth_model)

    # Create workflow
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Test document with content"])
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=model,
    )

    # Run workflow - should timeout during synthesis step
    start_time = asyncio.get_event_loop().time()
    result = await workflow.run(query="test query", max_retries=1)
    elapsed = asyncio.get_event_loop().time() - start_time

    # Verify timeout occurred (should complete in ~5 seconds, not 10+ seconds)
    assert elapsed < 7, f"Expected timeout at 5s, but took {elapsed:.1f}s"

    # Verify graceful error message (synthesis timeout returns error message)
    assert "encountered an error" in result["answer"].lower()


@pytest.mark.asyncio
async def test_timeout_is_configurable_via_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that llm_agent_timeout setting is respected.

    Task 19.3: Verify timeout value comes from Settings.llm_agent_timeout.
    """
    # Set custom timeout of 10 seconds (higher than minimum 5)
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "10")  # 10 second timeout
    get_settings.cache_clear()  # Clear cache to pick up new env vars
    settings = get_settings()

    # Create model that takes 3 seconds (under 10 second timeout)
    async def moderate_speed_model(
        messages: list,
        info: AgentInfo,
    ) -> ModelResponse:
        await asyncio.sleep(3)  # Takes 3 seconds (under timeout)
        return ModelResponse(parts=[TextPart(content="relevant")])

    model = FunctionModel(moderate_speed_model)

    # Create workflow
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Test document"])
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=model,
    )

    # Run workflow - should NOT timeout (1.5s < 2s timeout)
    result = await workflow.run(query="test query", max_retries=1)

    # Verify NO timeout occurred (evaluation succeeded)
    # With 10 second timeout, 3-second model completes successfully
    assert result["context_found"] is True
    assert "couldn't find relevant information" not in result["answer"].lower()
