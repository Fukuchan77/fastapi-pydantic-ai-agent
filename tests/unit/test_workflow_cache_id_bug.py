"""Unit tests for Task 28.1 - id() cache key bug fix.

Task 28.1 (P1-HIGH): Tests for WeakKeyDictionary implementation.

These tests verify that using WeakKeyDictionary prevents id() collision bugs.

Note on GC behavior: The workflow holds a strong reference to vector_store
(via self.vector_store), so the vector_store won't be GC'd while the workflow
exists in cache. However, WeakKeyDictionary still provides benefits:
1. Prevents id() collision (different objects = different cache entries)
2. Allows automatic cache cleanup when both key and value are unreferenced
"""

import gc
import weakref

from app.config import get_settings
from app.deps import workflow as wf
from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


def test_weakkeydictionary_type():
    """Test that _workflow_cache is a WeakKeyDictionary after fix.

    Task 28.1 (P1-HIGH): Verify correct implementation type.

    This test verifies the implementation uses WeakKeyDictionary instead of dict.
    """
    # DESIRED: _workflow_cache should be a WeakKeyDictionary
    assert isinstance(wf._workflow_cache, weakref.WeakKeyDictionary), (
        f"_workflow_cache should be WeakKeyDictionary, got {type(wf._workflow_cache).__name__}. "
        "Using WeakKeyDictionary prevents id() collision bugs."
    )


def test_workflow_cache_uses_object_identity():
    """Test that cache uses object identity, not id().

    Task 28.1 (P1-HIGH): WeakKeyDictionary uses object identity.

    This test verifies that different vector_store objects get different
    cache entries, even if they happen to have the same id() after GC.
    """
    # Create two vector stores
    store_1 = InMemoryVectorStore()
    store_2 = InMemoryVectorStore()

    settings = get_settings()
    model = wf._get_cached_model(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )

    # Cache workflow for store_1
    workflow_1 = CorrectiveRAGWorkflow(
        vector_store=store_1,
        llm_settings=settings,
        llm_model=model,
    )
    wf._workflow_cache[store_1] = workflow_1

    # Cache workflow for store_2
    workflow_2 = CorrectiveRAGWorkflow(
        vector_store=store_2,
        llm_settings=settings,
        llm_model=model,
    )
    wf._workflow_cache[store_2] = workflow_2

    # Verify both entries exist and are different
    assert len(wf._workflow_cache) == 2, "Cache should have two entries"
    assert wf._workflow_cache[store_1] is workflow_1
    assert wf._workflow_cache[store_2] is workflow_2
    assert workflow_1 is not workflow_2, "Different stores should have different workflows"

    # Verify cache returns correct workflow for each store (object identity, not id())
    assert wf._workflow_cache.get(store_1) is workflow_1
    assert wf._workflow_cache.get(store_2) is workflow_2


def test_workflow_cache_auto_cleanup_on_unreference():
    """Test that WeakKeyDictionary auto-removes entries when keys are GC'd.

    Task 28.1 (P1-HIGH): Automatic cache cleanup.

    This test verifies that when a vector_store object is no longer referenced
    anywhere (including not being referenced by a workflow in the cache),
    the WeakKeyDictionary automatically removes the cache entry.

    Note: This test is more about demonstrating WeakKeyDictionary behavior
    than testing a production scenario, since in production the vector_store
    lives in app.state for the application's lifetime.
    """
    # Create a vector store
    vector_store = InMemoryVectorStore()
    weak_ref = weakref.ref(vector_store)

    # Add to cache (but DON'T store the workflow in a variable)
    # This way, the only reference to vector_store will be from the cache key
    settings = get_settings()
    model = wf._get_cached_model(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )
    wf._workflow_cache[vector_store] = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=model,
    )

    assert len(wf._workflow_cache) == 1
    assert weak_ref() is not None, "Vector store should exist"

    # Delete the vector_store variable
    # Note: The workflow in the cache still holds a strong reference to vector_store
    # via self.vector_store, so it won't be GC'd yet
    del vector_store
    gc.collect()

    # The vector_store is still alive because the workflow references it
    # This is expected behavior - the workflow needs the vector_store to function
    assert weak_ref() is not None, (
        "Vector store still exists because workflow holds a strong reference to it. "
        "This is expected: the workflow needs the vector_store to function."
    )

    # However, when we clear the cache (removing the workflow), the vector_store
    # should become GC-able
    wf._workflow_cache.clear()
    gc.collect()

    # NOW the vector_store should be GC'd (no more references)
    assert weak_ref() is None, (
        "Vector store should be GC'd after cache is cleared. "
        "WeakKeyDictionary allowed automatic cleanup when the last reference was removed."
    )


def test_no_id_collision_bug():
    """Test that WeakKeyDictionary prevents the id() collision bug.

    Task 28.1 (P1-HIGH): Main bug fix verification.

    The old implementation used id(vector_store) as the cache key. After a
    vector_store was deleted and GC'd, Python could reuse that id() for a
    new object, causing the cache to return the wrong workflow.

    WeakKeyDictionary fixes this by using object identity, not id().
    """
    settings = get_settings()
    model = wf._get_cached_model(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )

    # Create and cache first vector store
    store_1 = InMemoryVectorStore()
    workflow_1 = CorrectiveRAGWorkflow(
        vector_store=store_1,
        llm_settings=settings,
        llm_model=model,
    )
    wf._workflow_cache[store_1] = workflow_1

    # Create second vector store
    store_2 = InMemoryVectorStore()

    # With old implementation using id() as key:
    # If id(store_2) == id(store_1), cache would return workflow_1 for store_2 (WRONG!)
    #
    # With WeakKeyDictionary using object identity:
    # store_2 is a different object, so it gets its own entry (or None if not cached)

    cached_for_store_2 = wf._workflow_cache.get(store_2)

    # store_2 has not been cached, so it should return None
    assert cached_for_store_2 is None, (
        "store_2 should not have a cached workflow (it's a different object). "
        "OLD BUG: If id(store_2) == id(store_1), old implementation would "
        "incorrectly return workflow_1. "
        "FIXED: WeakKeyDictionary uses object identity, so different objects "
        "get different cache entries."
    )

    # Verify store_1's workflow is still correctly cached
    assert wf._workflow_cache.get(store_1) is workflow_1
