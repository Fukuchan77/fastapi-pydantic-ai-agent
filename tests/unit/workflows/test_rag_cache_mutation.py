"""Unit tests for RAG workflow cache mutation protection.

Verifies that cached results cannot be mutated by callers.
"""

import pytest
from pydantic import SecretStr
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


@pytest.mark.asyncio
async def test_cache_returns_copy_not_reference():
    """Test that cached results are returned as copies, not references.

    Callers mutating the returned dict should not corrupt the cache.
    The workflow should return dict(cached_result) instead of cached_result.
    """
    # Create workflow with caching enabled
    vector_store = InMemoryVectorStore()
    settings = Settings(
        api_key=SecretStr("test-api-key-12345"),  # Must be at least 16 characters
        llm_model="openai:gpt-4",  # Must follow provider:model format
        rag_cache_ttl=3600,  # Enable cache with 1 hour TTL
        rag_cache_size=100,
    )

    # Use TestModel for fast, deterministic responses
    test_model = TestModel()
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=test_model,
    )

    # Add some documents to the vector store
    await vector_store.add_documents(["Test document about Python"])

    # First call - should cache the result
    result1 = await workflow.run(query="Python", max_retries=1)

    # Verify we got a result
    assert "answer" in result1
    assert isinstance(result1, dict)

    # Store original values
    original_answer = result1["answer"]
    original_context_found = result1["context_found"]

    # Mutate the returned dict (this is what callers might do)
    result1["answer"] = "CORRUPTED"
    result1["context_found"] = False
    result1["new_field"] = "should not appear in cache"

    # Second call with same parameters - should hit cache
    result2 = await workflow.run(query="Python", max_retries=1)

    # Verify cache was NOT corrupted by the mutation
    assert result2["answer"] == original_answer, "Cache was corrupted! Answer changed"
    assert result2["context_found"] == original_context_found, (
        "Cache was corrupted! context_found changed"
    )
    assert "new_field" not in result2, "Cache was corrupted! new_field appeared"

    # Verify result2 is a different object than result1 (not the same reference)
    assert result2 is not result1, "Cached result should be a copy, not the same reference"


@pytest.mark.asyncio
async def test_cache_isolation_between_calls():
    """Test that each cache hit returns an independent copy.

    Multiple callers should each get their own copy of the cached result,
    so mutations by one caller don't affect others.
    """
    # Create workflow with caching
    vector_store = InMemoryVectorStore()
    settings = Settings(
        api_key=SecretStr("test-api-key-12345"),  # Must be at least 16 characters
        llm_model="openai:gpt-4",  # Must follow provider:model format
        rag_cache_ttl=3600,
        rag_cache_size=100,
    )

    test_model = TestModel()
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=test_model,
    )

    await vector_store.add_documents(["Test document"])

    # First call - caches the result
    result1 = await workflow.run(query="test", max_retries=1)
    original_answer = result1["answer"]

    # Mutate result1
    result1["answer"] = "MUTATED_BY_CALLER_1"

    # Second call - cache hit
    result2 = await workflow.run(query="test", max_retries=1)

    # result2 should have original answer, not the mutated one
    assert result2["answer"] == original_answer

    # Mutate result2 differently
    result2["answer"] = "MUTATED_BY_CALLER_2"

    # Third call - cache hit
    result3 = await workflow.run(query="test", max_retries=1)

    # result3 should still have original answer
    assert result3["answer"] == original_answer

    # All three should be different objects
    assert result1 is not result2
    assert result2 is not result3
    assert result1 is not result3
