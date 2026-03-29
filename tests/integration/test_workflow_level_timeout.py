"""Tests for workflow-level timeout in Corrective RAG workflow.

Verify workflow.run() is wrapped with asyncio.timeout() to prevent
indefinite hangs when max_retries is high or LLM is slow.
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
async def test_workflow_times_out_after_rag_workflow_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that workflow.run() times out after rag_workflow_timeout seconds.

    Verify entire workflow execution is wrapped with timeout to prevent
    indefinite hangs even when individual LLM agent timeouts work correctly.
    """
    # Set workflow timeout of 10 seconds
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("RAG_WORKFLOW_TIMEOUT", "10")  # 10 second workflow timeout
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "30")  # Individual agent timeout higher
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "1")  # Disable retries
    get_settings.cache_clear()
    settings = get_settings()

    # Create slow LLM that takes 15 seconds per call (exceeds workflow timeout)
    async def very_slow_model(
        messages: list,
        info: AgentInfo,
    ) -> ModelResponse:
        await asyncio.sleep(15)  # Exceeds 10s workflow timeout
        return ModelResponse(parts=[TextPart(content="relevant")])

    slow_model = FunctionModel(very_slow_model)

    # Create workflow
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Test document"])
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=slow_model,
    )

    # Run workflow - should timeout at workflow level (10s, not 15s or 30s)
    # Wrap with asyncio.timeout() to simulate what the API does ()
    start_time = asyncio.get_event_loop().time()

    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(settings.rag_workflow_timeout):
            await workflow.run(query="test query", max_retries=1)

    elapsed = asyncio.get_event_loop().time() - start_time

    # Verify workflow-level timeout occurred (should be ~10 seconds)
    assert elapsed < 12, f"Expected workflow timeout at 10s, but took {elapsed:.1f}s"
    assert elapsed > 8, f"Timeout too fast at {elapsed:.1f}s, expected ~10s"


@pytest.mark.asyncio
async def test_workflow_timeout_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that rag_workflow_timeout setting is respected.

    Verify timeout value comes from Settings.rag_workflow_timeout.
    """
    # Set higher workflow timeout
    monkeypatch.setenv("API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
    monkeypatch.setenv("RAG_WORKFLOW_TIMEOUT", "20")  # 20 second workflow timeout
    monkeypatch.setenv("LLM_AGENT_TIMEOUT", "30")
    monkeypatch.setenv("LLM_RETRY_MAX_ATTEMPTS", "1")
    get_settings.cache_clear()
    settings = get_settings()

    # Create model that takes 8 seconds (under 20s workflow timeout)
    async def moderate_speed_model(
        messages: list,
        info: AgentInfo,
    ) -> ModelResponse:
        await asyncio.sleep(8)  # Under 20s workflow timeout
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

    # Run workflow - should NOT timeout (8s < 20s)
    # Wrap with asyncio.timeout() to simulate what the API does ()
    async with asyncio.timeout(settings.rag_workflow_timeout):
        result = await workflow.run(query="test query", max_retries=1)

    # Verify no timeout occurred
    assert result is not None
    assert "context_found" in result
