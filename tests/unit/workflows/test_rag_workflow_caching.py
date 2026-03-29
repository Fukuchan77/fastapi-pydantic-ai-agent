"""Unit tests for Task 26.3: Workflow caching performance optimization.

Task 26.3: The current implementation in app/deps/workflow.py creates a new
CorrectiveRAGWorkflow + LiteLLMModel + 2 Agents for every request. This is inefficient
because:
- Workflow instances are stateless (state lives in per-run Context)
- LiteLLMModel instances can be reused safely
- Agent instances can be reused safely

Expected behavior (after fix):
- Workflow instances should be cached and reused across requests
- Model instances should be cached and reused
- Agent instances should be cached and reused
- Performance should improve measurably

Current behavior (before fix - tests will FAIL):
- New workflow created per request (lines 10-31 in app/deps/workflow.py)
- New model created per request via build_model()
- New 2 agents created per workflow init (lines 74-85 in corrective_rag.py)
"""

import time
from unittest.mock import Mock

import pytest

from app.deps.workflow import get_rag_workflow


@pytest.mark.asyncio
async def test_workflow_instances_should_be_reused_across_requests() -> None:
    """Test that workflow instances are reused across multiple requests.

    Task 26.3: Currently creates new workflow per request - test will FAIL.
    After fix: Should return the same workflow instance - test will PASS.
    """
    # Create mock request with vector_store in app.state
    mock_request = Mock()
    mock_request.app.state.vector_store = Mock()

    # Call get_rag_workflow multiple times
    workflow1 = get_rag_workflow(mock_request)
    workflow2 = get_rag_workflow(mock_request)
    workflow3 = get_rag_workflow(mock_request)

    # Test that same workflow instance is returned (using `is` operator)
    # This will FAIL before fix because new instances are created
    assert workflow1 is workflow2, (
        "Workflow instances should be reused across requests. "
        "Current implementation creates new workflow per request."
    )
    assert workflow2 is workflow3, (
        "Workflow instances should be reused across requests. "
        "Current implementation creates new workflow per request."
    )


@pytest.mark.asyncio
async def test_llm_model_should_be_reused_across_workflows() -> None:
    """Test that LiteLLMModel instances are reused across workflows.

    Task 26.3: Currently creates new model per request via build_model().
    Test will FAIL before fix, PASS after fix.
    """
    # Create mock request with vector_store
    mock_request = Mock()
    mock_request.app.state.vector_store = Mock()

    # Get workflow instances
    workflow1 = get_rag_workflow(mock_request)
    workflow2 = get_rag_workflow(mock_request)

    # Test that same model instance is used
    # Access the model through workflow.llm_model
    assert workflow1.llm_model is workflow2.llm_model, (
        "LLM model instances should be reused. "
        "Current implementation calls build_model() per request."
    )


@pytest.mark.asyncio
async def test_agent_instances_should_be_reused_across_workflows() -> None:
    """Test that Agent instances are reused across workflows.

    Task 26.3: Currently creates new agents per workflow init.
    Test will FAIL before fix, PASS after fix.
    """
    # Create mock request with vector_store
    mock_request = Mock()
    mock_request.app.state.vector_store = Mock()

    # Get workflow instances
    workflow1 = get_rag_workflow(mock_request)
    workflow2 = get_rag_workflow(mock_request)

    # Test that same agent instances are used
    # Access agents through workflow._eval_agent and workflow._synth_agent
    assert workflow1._eval_agent is workflow2._eval_agent, (
        "Evaluation agent should be reused. Current implementation creates new agents per workflow."
    )
    assert workflow1._synth_agent is workflow2._synth_agent, (
        "Synthesis agent should be reused. Current implementation creates new agents per workflow."
    )


@pytest.mark.asyncio
async def test_workflow_caching_performance_improvement() -> None:
    """Test that workflow caching provides measurable performance improvement.

    Task 26.3: Measures time to create workflow instances.
    After caching, subsequent calls should be significantly faster.
    """
    # Create mock request with vector_store
    mock_request = Mock()
    mock_request.app.state.vector_store = Mock()

    # Measure time for first call (uncached)
    start_uncached = time.perf_counter()
    _ = get_rag_workflow(mock_request)
    time_uncached = time.perf_counter() - start_uncached

    # Measure time for subsequent calls (should be cached)
    times_cached = []
    for _ in range(10):
        start_cached = time.perf_counter()
        _ = get_rag_workflow(mock_request)
        times_cached.append(time.perf_counter() - start_cached)

    avg_time_cached = sum(times_cached) / len(times_cached)

    # Test that cached calls are at least 2x faster
    # This will FAIL before fix because all calls take similar time
    assert avg_time_cached < time_uncached / 2, (
        f"Cached calls should be at least 2x faster. "
        f"Uncached: {time_uncached * 1000:.2f}ms, "
        f"Cached avg: {avg_time_cached * 1000:.2f}ms"
    )
