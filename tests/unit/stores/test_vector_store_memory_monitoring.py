"""Tests for VectorStore memory usage monitoring - .

These tests verify that the VectorStore correctly:
- Tracks memory usage as documents are added
- Evicts oldest documents when memory limit is exceeded
- Provides accurate memory estimates
"""

import pytest

from app.stores.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_memory_monitoring_parameter_accepted():
    """Verify that InMemoryVectorStore accepts max_memory_bytes parameter."""
    # Should accept max_memory_bytes without error
    store = InMemoryVectorStore(max_memory_bytes=1_000_000)
    assert store.max_memory_bytes == 1_000_000


@pytest.mark.asyncio
async def test_memory_monitoring_defaults_to_none():
    """Verify that max_memory_bytes defaults to None (unlimited)."""
    store = InMemoryVectorStore()
    assert store.max_memory_bytes is None


@pytest.mark.asyncio
async def test_memory_usage_tracking():
    """Verify that memory usage is tracked as documents are added."""
    store = InMemoryVectorStore()

    # Initially, memory usage should be 0
    assert store.get_memory_usage() == 0

    # Add some documents
    await store.add_documents(["Hello world", "Test document"])

    # Memory usage should be > 0
    assert store.get_memory_usage() > 0


@pytest.mark.asyncio
async def test_memory_usage_estimate_accuracy():
    """Verify that memory usage estimate is reasonable."""
    store = InMemoryVectorStore()

    # Add a document with known size
    doc = "a" * 1000  # 1000 characters
    await store.add_documents([doc])

    memory_usage = store.get_memory_usage()

    # Memory usage should be at least the document size
    # (plus overhead for Python objects, lists, etc.)
    assert memory_usage >= 1000

    # But shouldn't be excessively large (< 10x the data size)
    assert memory_usage < 10000


@pytest.mark.asyncio
async def test_memory_eviction_when_limit_exceeded():
    """Verify that oldest documents are evicted when memory limit is exceeded."""
    # Set a memory limit that can hold 3 medium documents but not 4
    # Each ~305-char doc uses ~1330 bytes (doc + tokens + overhead)
    # 3 docs = ~3990 bytes, 4 docs = ~5320 bytes
    store = InMemoryVectorStore(max_memory_bytes=5000)

    # Add documents that fit within limit
    await store.add_documents(["doc1-" + "a" * 300])  # ~1330 bytes
    await store.add_documents(["doc2-" + "b" * 300])  # ~1330 bytes
    await store.add_documents(["doc3-" + "c" * 300])  # ~1330 bytes

    # Verify all 3 documents are present (total ~3990 bytes < 5000)
    assert len(store._documents) == 3

    # Add a document that pushes over limit (requires evicting older docs)
    new_doc = "doc4-" + "d" * 300  # ~600 bytes
    await store.add_documents([new_doc])

    # Memory limit should trigger eviction of oldest documents
    # The new document should be present
    assert new_doc in store._documents

    # Total documents should be <= 3 (some were evicted)
    assert len(store._documents) <= 3

    # doc1 (oldest) should have been evicted
    assert not any("doc1" in doc for doc in store._documents)


@pytest.mark.asyncio
async def test_memory_eviction_keeps_newest_documents():
    """Verify that memory eviction preserves newest documents (FIFO)."""
    store = InMemoryVectorStore(max_memory_bytes=3000)

    # Add documents incrementally
    await store.add_documents(["doc1-" + "a" * 500])
    await store.add_documents(["doc2-" + "b" * 500])
    await store.add_documents(["doc3-" + "c" * 500])
    await store.add_documents(["doc4-" + "d" * 500])
    await store.add_documents(["doc5-" + "e" * 500])

    # Memory limit should trigger eviction
    # The newest documents should be kept
    remaining_docs = store._documents

    # doc5 (newest) should definitely be present
    assert any("doc5" in doc for doc in remaining_docs)

    # doc1 (oldest) should have been evicted
    assert not any("doc1" in doc for doc in remaining_docs)


@pytest.mark.asyncio
async def test_memory_limit_works_with_document_limit():
    """Verify that memory limit and document limit work together."""
    # Set both limits
    store = InMemoryVectorStore(max_documents=10, max_memory_bytes=2000)

    # Add documents
    docs = [f"doc{i}-" + "x" * 100 for i in range(15)]
    await store.add_documents(docs)

    # Should enforce BOTH limits (whichever is more restrictive)
    assert len(store._documents) <= 10  # Document limit
    assert store.get_memory_usage() <= 2500  # Memory limit (with some tolerance)


@pytest.mark.asyncio
async def test_clear_resets_memory_usage():
    """Verify that clear() resets memory usage to 0."""
    store = InMemoryVectorStore()

    # Add documents
    await store.add_documents(["doc1", "doc2", "doc3"])
    assert store.get_memory_usage() > 0

    # Clear store
    await store.clear()

    # Memory usage should be 0
    assert store.get_memory_usage() == 0


@pytest.mark.asyncio
async def test_memory_eviction_maintains_sync():
    """Verify that memory eviction keeps _documents and _doc_tokens synchronized."""
    store = InMemoryVectorStore(max_memory_bytes=2000)

    # Add many small documents
    docs = [f"document number {i}" for i in range(100)]
    await store.add_documents(docs)

    # After eviction, lengths should match
    assert len(store._documents) == len(store._doc_tokens)

    # All documents should still be queryable
    results = await store.query("document")
    assert len(results) > 0
