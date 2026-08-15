"""Unit tests for InMemoryVectorStore construction, CRUD, and lifecycle.

Tests cover:
- Construction/configuration
- add_documents() (validation, atomicity, batching)
- FIFO eviction when max_documents exceeded
- clear()
- close()

Split out of test_vector_store.py per `.sdd/steering/file-size-policy.md`;
TF-IDF ranking, query() validation, IDF caching, and query_with_scores()
tests live in sibling files.
"""

import pytest

from app.stores.vector_store import InMemoryVectorStore


class TestInMemoryVectorStoreConstruction:
    """Test InMemoryVectorStore initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Store initializes with default parameters."""
        store = InMemoryVectorStore()
        assert store.max_documents == 1000
        assert store.max_chunk_size == 100_000

    def test_custom_max_documents(self) -> None:
        """Store accepts custom max_documents parameter."""
        store = InMemoryVectorStore(max_documents=500)
        assert store.max_documents == 500

    def test_custom_max_chunk_size(self) -> None:
        """Store accepts custom max_chunk_size parameter."""
        store = InMemoryVectorStore(max_chunk_size=50_000)
        assert store.max_chunk_size == 50_000

    def test_custom_parameters(self) -> None:
        """Store accepts both custom parameters."""
        store = InMemoryVectorStore(max_documents=100, max_chunk_size=20_000)
        assert store.max_documents == 100
        assert store.max_chunk_size == 20_000


class TestAddDocuments:
    """Test add_documents() operation."""

    @pytest.mark.asyncio
    async def test_add_single_document(self) -> None:
        """Store accepts single document."""
        store = InMemoryVectorStore()
        await store.add_documents(["First document"])
        results = await store.query("First", top_k=1)
        assert len(results) == 1
        assert results[0] == "First document"

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self) -> None:
        """Store accepts multiple documents."""
        store = InMemoryVectorStore()
        await store.add_documents(["Doc one", "Doc two", "Doc three"])
        results = await store.query("Doc", top_k=10)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_add_empty_list(self) -> None:
        """Store accepts empty list without error."""
        store = InMemoryVectorStore()
        await store.add_documents([])
        results = await store.query("test", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_add_chunk_exceeding_max_size(self) -> None:
        """Store rejects chunks exceeding max_chunk_size."""
        store = InMemoryVectorStore(max_chunk_size=100)
        large_chunk = "x" * 101
        with pytest.raises(ValueError, match="Document chunk too large"):
            await store.add_documents([large_chunk])


class TestGeneration:
    """Test the `generation` content-version counter (Req 2.1, 2.2, 2.4).

    Boundary correction for task 3.1
    (`.sdd/specs/002-review-roadmap-remediation/tasks.md`): 3.1's own
    boundary has no test file, so this class is the RED step ahead of task
    3.3's formal parametrized conformance widening.
    """

    def test_generation_starts_at_zero(self) -> None:
        """A freshly constructed store starts at generation 0."""
        store = InMemoryVectorStore()
        assert store.generation == 0

    @pytest.mark.asyncio
    async def test_generation_increments_after_successful_add(self) -> None:
        """A successful add_documents() call advances generation."""
        store = InMemoryVectorStore()
        await store.add_documents(["First document"])
        assert store.generation == 1
        await store.add_documents(["Second document"])
        assert store.generation == 2

    @pytest.mark.asyncio
    async def test_generation_increments_after_clear(self) -> None:
        """A clear() call advances generation."""
        store = InMemoryVectorStore()
        await store.add_documents(["First document"])
        await store.clear()
        assert store.generation == 2

    @pytest.mark.asyncio
    async def test_generation_unchanged_on_failed_add(self) -> None:
        """A rejected add_documents() call (oversized chunk) leaves generation unchanged."""
        store = InMemoryVectorStore(max_chunk_size=100)
        large_chunk = "x" * 101
        with pytest.raises(ValueError, match="Document chunk too large"):
            await store.add_documents([large_chunk])
        assert store.generation == 0

    @pytest.mark.asyncio
    async def test_add_chunk_at_max_size_boundary(self) -> None:
        """Store accepts chunks exactly at max_chunk_size."""
        store = InMemoryVectorStore(max_chunk_size=100)
        chunk = "x" * 100
        await store.add_documents([chunk])
        results = await store.query("x", top_k=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_add_multiple_batches(self) -> None:
        """Store accumulates documents from multiple add_documents calls."""
        store = InMemoryVectorStore()
        await store.add_documents(["Doc one"])
        await store.add_documents(["Doc two"])
        results = await store.query("Doc", top_k=10)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_atomic_validation(self) -> None:
        """When one chunk fails validation, no chunks are added."""
        store = InMemoryVectorStore(max_chunk_size=50)
        await store.add_documents(["valid"])
        with pytest.raises(ValueError, match="Document chunk too large"):
            await store.add_documents(["ok", "x" * 51, "also ok"])
        # Only the first valid document should be present
        results = await store.query("valid ok also", top_k=10)
        assert len(results) == 1
        assert results[0] == "valid"


class TestFIFOEviction:
    """Test FIFO eviction when max_documents is exceeded."""

    @pytest.mark.asyncio
    async def test_fifo_eviction_after_exceeding_max(self) -> None:
        """Oldest documents are evicted when max_documents is exceeded."""
        store = InMemoryVectorStore(max_documents=3)
        await store.add_documents(["doc1", "doc2", "doc3"])
        await store.add_documents(["doc4", "doc5"])
        # After adding 5 docs with max=3, only last 3 should remain
        results = await store.query("doc", top_k=10)
        assert len(results) == 3
        # doc1 and doc2 should be evicted
        assert "doc1" not in results
        assert "doc2" not in results
        assert "doc3" in results
        assert "doc4" in results
        assert "doc5" in results

    @pytest.mark.asyncio
    async def test_fifo_eviction_single_batch(self) -> None:
        """FIFO eviction works when single add_documents exceeds max."""
        store = InMemoryVectorStore(max_documents=2)
        await store.add_documents(["doc1", "doc2", "doc3", "doc4"])
        results = await store.query("doc", top_k=10)
        assert len(results) == 2
        # Only last 2 documents should remain
        assert "doc3" in results
        assert "doc4" in results

    @pytest.mark.asyncio
    async def test_no_eviction_when_under_limit(self) -> None:
        """No documents are evicted when under max_documents limit."""
        store = InMemoryVectorStore(max_documents=100)
        await store.add_documents(["doc1", "doc2", "doc3"])
        results = await store.query("doc", top_k=10)
        assert len(results) == 3


class TestClear:
    """Test clear() operation."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_documents(self) -> None:
        """Clear removes all documents from store."""
        store = InMemoryVectorStore()
        await store.add_documents(["doc1", "doc2", "doc3"])
        await store.clear()
        results = await store.query("doc", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_clear_on_empty_store(self) -> None:
        """Clear on empty store does not raise error."""
        store = InMemoryVectorStore()
        await store.clear()
        results = await store.query("test", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_add_after_clear(self) -> None:
        """Documents can be added after clearing."""
        store = InMemoryVectorStore()
        await store.add_documents(["old doc"])
        await store.clear()
        await store.add_documents(["new doc"])
        results = await store.query("doc", top_k=5)
        assert len(results) == 1
        assert results[0] == "new doc"

    @pytest.mark.asyncio
    async def test_clear_invalidates_idf_cache(self) -> None:
        """Clear invalidates IDF cache (indirect test via behavior)."""
        store = InMemoryVectorStore()
        await store.add_documents(["first batch"])
        await store.query("first", top_k=1)  # Populate IDF cache
        await store.clear()
        await store.add_documents(["second batch"])
        results = await store.query("second", top_k=1)
        # If cache wasn't cleared, "second" wouldn't be found
        assert len(results) == 1
        assert results[0] == "second batch"


class TestClose:
    """Test close() operation for resource cleanup."""

    @pytest.mark.asyncio
    async def test_in_memory_vector_store_has_close_method(self) -> None:
        """InMemoryVectorStore has close() method."""
        store = InMemoryVectorStore()
        assert hasattr(store, "close")
        assert callable(store.close)

    @pytest.mark.asyncio
    async def test_close_does_not_raise_error(self) -> None:
        """close() completes without raising error."""
        store = InMemoryVectorStore()
        await store.add_documents(["test doc"])
        # Should not raise any exception
        await store.close()

    @pytest.mark.asyncio
    async def test_close_on_empty_store(self) -> None:
        """close() on empty store does not raise error."""
        store = InMemoryVectorStore()
        # Should not raise any exception
        await store.close()
