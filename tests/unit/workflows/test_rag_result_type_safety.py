"""Test for type safety of dict(result) calls in RAG workflow (Task 25.4).

The issue: Lines 191 and 206 in corrective_rag.py call dict(result) without
verifying that result is actually a dict. If Workflow.run() returns a non-dict
value (due to a bug or API change), this could cause unexpected behavior.

This test verifies that runtime type checking is in place to catch this issue.
"""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


@pytest.mark.asyncio
async def test_type_error_raised_for_non_dict_result_from_cache():
    """Test that TypeError is raised when cached result is not a dict.

    Task 25.4 RED phase: This test should FAIL before adding type checking.
    After GREEN phase, it should PASS by raising TypeError.
    """
    import time

    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Python is a programming language"])

    settings = Settings(
        api_key=SecretStr("test-api-key-12345678"),
        llm_model="openai:gpt-4o",
        rag_cache_ttl=300,
        rag_cache_size=100,
    )

    test_model = TestModel()
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=test_model,
    )

    # Directly corrupt the cache with a non-dict value (simulating bug or corruption)
    cache_key = workflow._generate_cache_key("What is Python?", 3)
    workflow._cache[cache_key] = ("not a dict", time.time())  # Invalid cached value

    # Before fix: dict("not a dict") might succeed or fail unpredictably
    # After fix: Should raise TypeError with clear message
    with pytest.raises(TypeError, match=r"Expected dict.*got.*str"):
        await workflow.run(query="What is Python?", max_retries=3)


@pytest.mark.asyncio
async def test_type_error_raised_for_non_dict_result_from_workflow():
    """Test that TypeError is raised when workflow execution returns non-dict.

    Task 25.4 RED phase: This test should FAIL before adding type checking.
    After GREEN phase, it should PASS by raising TypeError.
    """
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Python is a programming language"])

    settings = Settings(
        api_key=SecretStr("test-api-key-12345678"),
        llm_model="openai:gpt-4o",
        rag_cache_ttl=300,
        rag_cache_size=100,
    )

    test_model = TestModel()
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=test_model,
    )

    # Mock the parent workflow's run() to return a non-dict value
    with patch.object(workflow.__class__.__bases__[0], "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "not a dict"  # Invalid return value

        # Before fix: dict("not a dict") creates dict from string chars
        # After fix: Should raise TypeError with clear message
        with pytest.raises(TypeError, match=r"Expected dict.*got.*str"):
            await workflow.run(query="What is Python?", max_retries=3)


@pytest.mark.asyncio
async def test_type_error_raised_for_none_result():
    """Test that TypeError is raised when workflow returns None.

    Task 25.4: Verify None is caught as invalid type.
    """
    vector_store = InMemoryVectorStore()
    await vector_store.add_documents(["Python is a programming language"])

    settings = Settings(
        api_key=SecretStr("test-api-key-12345678"),
        llm_model="openai:gpt-4o",
        rag_cache_ttl=300,
        rag_cache_size=100,
    )

    test_model = TestModel()
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=test_model,
    )

    # Mock the parent workflow's run() to return None
    with patch.object(workflow.__class__.__bases__[0], "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = None

        # Should raise TypeError for None value
        with pytest.raises(TypeError, match=r"Expected dict.*got.*NoneType"):
            await workflow.run(query="What is Python?", max_retries=3)
