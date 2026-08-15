"""Unit tests for Workflow caching performance optimization.

The current implementation in app/deps/workflow.py creates a new
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
from fastapi import FastAPI
from fastapi import Request

from app.agents.chat_agent import build_model
from app.deps.workflow import get_rag_workflow
from tests.conftest import build_test_settings


def _make_mock_request() -> Request:
    """Build a real `Request` with `app.state.settings`/`llm_model` populated.

    `get_rag_workflow` reads both from `app.state` (Req 3.3) - a bare
    `Mock()` request would auto-vivify both as `Mock` objects instead of
    raising the 503 this dependency now raises for a genuinely absent
    singleton, and passing a `Mock` through as `llm_model` breaks
    `pydantic_ai.models.infer_model()` downstream. `vector_store` stays a
    plain `Mock()`: it is only ever used as a `WeakKeyDictionary` cache key
    here, never called.
    """
    app = FastAPI()
    app.state.vector_store = Mock()
    settings = build_test_settings()
    app.state.settings = settings
    app.state.llm_model = build_model(settings)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "app": app,
    }
    return Request(scope=scope, receive=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_workflow_instances_should_be_reused_across_requests() -> None:
    """Test that workflow instances are reused across multiple requests.

    Currently creates new workflow per request - test will FAIL.
    After fix: Should return the same workflow instance - test will PASS.
    """
    mock_request = _make_mock_request()

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

    Currently creates new model per request via build_model().
    Test will FAIL before fix, PASS after fix.
    """
    # Create mock request with vector_store
    mock_request = _make_mock_request()

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

    Currently creates new agents per workflow init.
    Test will FAIL before fix, PASS after fix.
    """
    # Create mock request with vector_store
    mock_request = _make_mock_request()

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

    Measures time to create workflow instances.
    After caching, subsequent calls should be significantly faster.
    """
    # Create mock request with vector_store
    mock_request = _make_mock_request()

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
