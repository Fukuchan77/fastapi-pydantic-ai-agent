"""Unit tests for RAG workflow result caching.

Task 17.1: Implement query result caching to reduce redundant LLM calls
and vector store queries for identical requests.
"""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


@pytest.fixture
def mock_vector_store():
    """Create mock vector store for testing."""
    store = AsyncMock()
    store.query.return_value = ["Test document chunk"]
    return store


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.llm_model = "test:model"
    settings.max_output_retries = 3
    settings.llm_agent_timeout = 30
    settings.rag_workflow_timeout = 60
    settings.llm_retry_max_attempts = 3
    settings.llm_retry_base_delay = 1.0
    settings.rag_cache_ttl = 300  # 5 minutes
    settings.rag_cache_size = 100  # Max 100 cached results
    return settings


@pytest.mark.asyncio
async def test_identical_queries_return_cached_results(
    mock_vector_store,
    mock_settings,
):
    """Test that identical queries return cached results without re-executing workflow.

    Task 17.1: When the same query is executed twice within the cache TTL,
    the second call should return the cached result immediately without
    calling the vector store or LLM.
    """
    workflow = CorrectiveRAGWorkflow(
        vector_store=mock_vector_store,
        llm_settings=mock_settings,
        llm_model=TestModel(),
    )

    query = "What is FastAPI?"

    # First call - should execute workflow and cache result
    result1 = await workflow.run(query=query, max_retries=3)
    first_call_count = mock_vector_store.query.call_count

    # Second call with identical query - should return cached result
    result2 = await workflow.run(query=query, max_retries=3)
    second_call_count = mock_vector_store.query.call_count

    # Verify cached result was returned (vector store not called again)
    assert second_call_count == first_call_count, "Cached query should not call vector store again"

    # Verify results are identical
    assert result1 == result2, "Cached result should match original result"


@pytest.mark.asyncio
async def test_different_queries_are_not_cached_together(
    mock_vector_store,
    mock_settings,
):
    """Test that different queries execute independently (not cached together).

    Task 17.1: Each unique query should have its own cache entry.
    Different queries should not share cached results.
    """
    workflow = CorrectiveRAGWorkflow(
        vector_store=mock_vector_store,
        llm_settings=mock_settings,
        llm_model=TestModel(),
    )

    # Execute two different queries
    await workflow.run(query="What is FastAPI?", max_retries=3)
    call_count_after_first = mock_vector_store.query.call_count

    await workflow.run(query="What is Pydantic AI?", max_retries=3)
    call_count_after_second = mock_vector_store.query.call_count

    # Verify both queries executed (vector store called for both)
    assert call_count_after_second > call_count_after_first, (
        "Different queries should execute independently, not use cached results"
    )


@pytest.mark.asyncio
async def test_cache_respects_max_retries_parameter(
    mock_vector_store,
    mock_settings,
):
    """Test that cache key includes max_retries parameter.

    Task 17.1: The same query with different max_retries should be
    cached separately, as they may produce different results.
    """
    workflow = CorrectiveRAGWorkflow(
        vector_store=mock_vector_store,
        llm_settings=mock_settings,
        llm_model=TestModel(),
    )

    query = "What is FastAPI?"

    # First call with max_retries=3
    await workflow.run(query=query, max_retries=3)
    call_count_after_first = mock_vector_store.query.call_count

    # Second call with max_retries=5 (different parameter)
    await workflow.run(query=query, max_retries=5)
    call_count_after_second = mock_vector_store.query.call_count

    # Verify second call executed (not cached due to different max_retries)
    assert call_count_after_second > call_count_after_first, (
        "Same query with different max_retries should not use cached result"
    )


@pytest.mark.asyncio
async def test_cache_has_size_limit(
    mock_vector_store,
    mock_settings,
):
    """Test that cache has LRU eviction when size limit is reached.

    Task 17.1: The cache should have a maximum size (default 100 entries).
    When the limit is reached, the least recently used entry should be evicted.
    """
    mock_settings.rag_cache_size = 5  # Small cache for testing

    workflow = CorrectiveRAGWorkflow(
        vector_store=mock_vector_store,
        llm_settings=mock_settings,
        llm_model=TestModel(),
    )

    # Execute 6 different queries (exceeds cache size of 5)
    for i in range(6):
        await workflow.run(query=f"Query {i}", max_retries=3)

    # The first query should have been evicted due to LRU
    # Executing it again should call the vector store
    call_count_before = mock_vector_store.query.call_count
    await workflow.run(query="Query 0", max_retries=3)
    call_count_after = mock_vector_store.query.call_count

    assert call_count_after > call_count_before, (
        "LRU eviction should remove oldest entry when cache is full"
    )


@pytest.mark.asyncio
async def test_cache_statistics_are_tracked(
    mock_vector_store,
    mock_settings,
):
    """Test that cache hit/miss statistics are tracked.

    Task 17.1: The workflow should expose cache statistics
    (hits, misses, size) for monitoring and debugging.
    """
    workflow = CorrectiveRAGWorkflow(
        vector_store=mock_vector_store,
        llm_settings=mock_settings,
        llm_model=TestModel(),
    )

    # Should have cache_stats attribute or method
    assert hasattr(workflow, "cache_stats") or hasattr(workflow, "get_cache_stats"), (
        "Workflow should expose cache statistics"
    )

    # Execute queries to generate hits and misses
    await workflow.run(query="Query 1", max_retries=3)  # miss
    await workflow.run(query="Query 1", max_retries=3)  # hit
    await workflow.run(query="Query 2", max_retries=3)  # miss

    # Get stats (try both attribute and method)
    stats = workflow.cache_stats if hasattr(workflow, "cache_stats") else workflow.get_cache_stats()

    # Verify stats contain expected fields
    assert "hits" in stats, "Stats should include cache hits"
    assert "misses" in stats, "Stats should include cache misses"
    assert "size" in stats, "Stats should include current cache size"

    # Verify stats are reasonable
    assert stats["hits"] >= 1, "Should have at least 1 cache hit"
    assert stats["misses"] >= 2, "Should have at least 2 cache misses"
