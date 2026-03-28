"""Tests for Task 21.1: Add asyncio.Lock to protect LRU cache in CorrectiveRAGWorkflow.

Race condition: Multiple coroutines accessing _cache concurrently without locking causes:
1. Cache miss race: Two simultaneous requests for same query both miss cache and
   execute workflow twice
2. Eviction race: _evict_expired_entries() and cache writes are not atomic
3. LRU race: move_to_end() and concurrent writes can corrupt OrderedDict
"""

import asyncio
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


@pytest.fixture
def mock_settings():
    """Create mock settings for workflow."""
    return Settings(
        api_key=SecretStr("test-api-key-12345"),
        llm_model="openai:gpt-4",
        rag_cache_ttl=300,
        rag_cache_size=100,
        llm_retry_max_attempts=3,
        llm_retry_base_delay=0.1,
        llm_agent_timeout=30,
    )


@patch("app.workflows.corrective_rag.Agent")
def test_workflow_has_cache_lock_attribute(mock_agent_class, mock_settings):
    """Task 21.1: Verify that CorrectiveRAGWorkflow has _cache_lock attribute.

    This test will FAIL initially (no lock attribute), then PASS after fix.
    """
    # Mock Agent instances to avoid OpenAI API initialization
    mock_agent_class.return_value = Mock()

    vector_store = InMemoryVectorStore()
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=mock_settings,
    )

    # Task 21.1: Workflow should have _cache_lock attribute
    assert hasattr(workflow, "_cache_lock"), "CorrectiveRAGWorkflow must have _cache_lock attribute"
    assert isinstance(workflow._cache_lock, asyncio.Lock), (
        "_cache_lock must be an asyncio.Lock instance"
    )


@patch("app.workflows.corrective_rag.Agent")
def test_cache_lock_is_initialized_in_constructor(mock_agent_class, mock_settings):
    """Task 21.1: Verify that _cache_lock is initialized in __init__().

    This ensures the lock is created when the workflow is instantiated.
    """
    # Mock Agent instances to avoid OpenAI API initialization
    mock_agent_class.return_value = Mock()

    vector_store = InMemoryVectorStore()
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=mock_settings,
    )

    # _cache_lock should be initialized and not None
    assert workflow._cache_lock is not None
    assert isinstance(workflow._cache_lock, asyncio.Lock)
