"""Unit tests for workflow cache isolation between tests.

Task 27.1 + Task 28.1: Ensures _workflow_cache doesn't pollute between test runs.
Updated for Task 28.1: Uses WeakKeyDictionary with vector_store objects as keys.
"""

import pytest

from app.config import get_settings
from app.deps import workflow as wf
from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


def test_workflow_cache_is_empty_at_test_start():
    """Verify workflow cache is cleared before each test.

    Task 27.1: This test should pass if cache clearing works.
    If it fails, it means cache pollution from other tests.
    """
    # Cache should be empty at test start
    assert len(wf._workflow_cache) == 0, (
        f"Cache should be empty but has {len(wf._workflow_cache)} entries"
    )


def test_workflow_cache_isolation_first():
    """First test that adds to cache.

    Task 27.1 + 28.1: This test populates the cache to verify next test sees clean state.
    Updated to use vector_store objects as keys (not integers).
    """
    # Simulate cache population using a vector_store object as key
    test_store = InMemoryVectorStore()
    settings = get_settings()
    model = wf._get_cached_model(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )
    test_workflow = CorrectiveRAGWorkflow(
        vector_store=test_store,
        llm_settings=settings,
        llm_model=model,
    )
    wf._workflow_cache[test_store] = test_workflow

    assert len(wf._workflow_cache) == 1
    assert wf._workflow_cache[test_store] is test_workflow


def test_workflow_cache_isolation_second():
    """Second test verifies cache was cleared.

    Task 27.1: This test will FAIL if cache clearing fixture is not present,
    because it will see the entry from test_workflow_cache_isolation_first.
    """
    # Cache should be empty - the previous test's entry should be cleared
    assert len(wf._workflow_cache) == 0, (
        f"Cache should be empty but has {len(wf._workflow_cache)} entries. "
        "This indicates cache pollution from previous test."
    )


@pytest.mark.parametrize("test_id", [1, 2, 3])
def test_workflow_cache_cleared_between_parametrized_tests(test_id):
    """Verify cache is cleared between parametrized test runs.

    Task 27.1 + 28.1: Parametrized tests should also have isolated cache state.
    Updated to use vector_store objects as keys (not integers).
    """
    # Each parametrized run should start with empty cache
    initial_size = len(wf._workflow_cache)
    assert initial_size == 0, f"Cache should be empty at start of test {test_id}"

    # Add entry for this test using a vector_store object as key
    test_store = InMemoryVectorStore()
    settings = get_settings()
    model = wf._get_cached_model(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )
    test_workflow = CorrectiveRAGWorkflow(
        vector_store=test_store,
        llm_settings=settings,
        llm_model=model,
    )
    wf._workflow_cache[test_store] = test_workflow

    # Verify our entry exists
    assert len(wf._workflow_cache) == 1
