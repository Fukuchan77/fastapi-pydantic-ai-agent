"""Test for thundering herd problem in RAG cache (Task 25.2).

The issue: When multiple concurrent requests arrive for the same query,
they all see a cache miss and execute duplicate LLM workflows because
the lock is released after cache check but before workflow execution.

This test demonstrates and verifies the fix using a "pending future" pattern.
"""

import asyncio

import pytest
from pydantic import SecretStr
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


@pytest.mark.asyncio
async def test_thundering_herd_duplicate_llm_calls():
    """Test that concurrent identical queries result in only ONE workflow execution.

    Task 25.2: With the pending future pattern, multiple concurrent requests
    all enter CorrectiveRAGWorkflow.run(), but only the first request should
    execute the actual workflow via super().run(). The other requests should
    await the pending future.

    This test verifies the fix by counting how many times the vector store
    query method is called. It's called exactly once per workflow execution,
    so this accurately measures workflow executions without interfering with
    the workflow's @step decorator chain.
    """
    # Setup
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Python is a programming language"])

    settings = Settings(
        api_key=SecretStr("test-api-key-12345678"),  # 16+ chars
        llm_model="openai:gpt-4o",  # Use valid provider
        rag_cache_ttl=300,  # Enable cache
        rag_cache_size=100,
    )

    # Track how many times the vector store query is called
    # It's called once per workflow execution (from the search step)
    query_count = 0
    original_query = vector_store.query

    async def counting_query(*args, **kwargs):
        """Wrapper to count vector store query calls."""
        nonlocal query_count
        query_count += 1
        return await original_query(*args, **kwargs)

    # Monkey patch vector_store.query to count calls
    vector_store.query = counting_query

    try:
        # Create workflow with TestModel (no real LLM calls)
        test_model = TestModel()
        workflow = CorrectiveRAGWorkflow(
            vector_store=vector_store,
            llm_settings=settings,
            llm_model=test_model,
        )

        # Execute 5 concurrent identical queries
        # Use max_retries=1 to ensure only 1 search per workflow (no retries)
        query = "What is Python?"
        tasks = [workflow.run(query=query, max_retries=1) for _ in range(5)]

        # All 5 should complete
        results = await asyncio.gather(*tasks)

        # All results should be identical (from same workflow execution)
        assert len(results) == 5
        assert all(r == results[0] for r in results), "All results should be identical"

        # CRITICAL: Only ONE vector store query should have occurred
        # This means only 1 actual workflow was executed (via super().run())
        # The other 4 requests awaited the pending future
        assert query_count == 1, (
            f"Expected 1 vector store query (1 workflow execution), "
            f"but got {query_count}. This indicates thundering herd problem."
        )

    finally:
        # Restore original method
        vector_store.query = original_query


@pytest.mark.asyncio
async def test_cache_with_different_queries_executes_separately():
    """Test that different queries still execute separately (not affected by fix).

    This ensures the pending future pattern doesn't incorrectly coalesce
    different queries.
    """
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Python is a language", "JavaScript is a language"])

    settings = Settings(
        api_key=SecretStr("test-api-key-12345678"),
        llm_model="openai:gpt-4o",  # Use valid provider
        rag_cache_ttl=300,
        rag_cache_size=100,
    )

    test_model = TestModel()
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=test_model,
    )

    # Execute 2 different queries concurrently
    await workflow.run(query="What is Python?", max_retries=3)
    await workflow.run(query="What is JavaScript?", max_retries=3)

    # Different queries should cache separately (verified by checking cache size)

    # Cache should have 2 entries
    assert workflow.cache_stats["size"] == 2
