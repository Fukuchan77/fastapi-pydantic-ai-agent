"""Unit tests for VectorStore Protocol and InMemoryVectorStore implementation."""

import pytest

from app.stores.vector_store import InMemoryVectorStore
from app.stores.vector_store import VectorStore


class TestVectorStoreProtocol:
    """Test that VectorStore Protocol is correctly defined."""

    def test_vector_store_protocol_has_required_methods(self) -> None:
        """VectorStore Protocol must define add_documents, query, and clear methods."""
        assert hasattr(VectorStore, "add_documents")
        assert hasattr(VectorStore, "query")
        assert hasattr(VectorStore, "clear")


class TestInMemoryVectorStore:
    """Test InMemoryVectorStore TF-IDF implementation."""

    @pytest.fixture
    def store(self) -> InMemoryVectorStore:
        """Provide a fresh InMemoryVectorStore instance."""
        return InMemoryVectorStore()

    @pytest.mark.asyncio
    async def test_empty_store_returns_empty_list(self, store: InMemoryVectorStore) -> None:
        """Query on empty corpus must return empty list."""
        results = await store.query("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_add_documents_and_query(self, store: InMemoryVectorStore) -> None:
        """Documents can be added and queried."""
        chunks = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a programming language",
            "FastAPI is a web framework for Python",
        ]
        await store.add_documents(chunks)

        results = await store.query("Python programming", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_tfidf_ranking_order(self, store: InMemoryVectorStore) -> None:
        """Results must be ranked by TF-IDF cosine similarity score."""
        chunks = [
            "Machine learning is a subset of artificial intelligence",
            "Python machine learning libraries include scikit-learn",
            "The weather is nice today",
        ]
        await store.add_documents(chunks)

        # Query that should match the second chunk most closely
        results = await store.query("Python machine learning", top_k=3)

        # The second chunk should be ranked first (most similar)
        assert "Python machine learning" in results[0]
        # The third chunk (weather) should be ranked last (least similar)
        assert "weather" in results[-1]

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, store: InMemoryVectorStore) -> None:
        """Query must respect top_k parameter."""
        chunks = [f"Document number {i}" for i in range(10)]
        await store.add_documents(chunks)

        results = await store.query("Document", top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_top_k_default_value(self, store: InMemoryVectorStore) -> None:
        """Query must use default top_k=5 when not specified."""
        chunks = [f"Document number {i}" for i in range(10)]
        await store.add_documents(chunks)

        results = await store.query("Document")
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_clear_empties_store(self, store: InMemoryVectorStore) -> None:
        """Clear must remove all documents from the store."""
        chunks = ["Document 1", "Document 2", "Document 3"]
        await store.add_documents(chunks)

        # Verify documents were added
        results_before = await store.query("Document", top_k=5)
        assert len(results_before) == 3

        # Clear the store
        await store.clear()

        # Verify store is empty
        results_after = await store.query("Document", top_k=5)
        assert results_after == []

    @pytest.mark.asyncio
    async def test_query_with_no_matching_terms(self, store: InMemoryVectorStore) -> None:
        """Query with no matching terms should return documents ranked by corpus statistics."""
        chunks = [
            "The quick brown fox",
            "Python programming language",
            "Machine learning algorithms",
        ]
        await store.add_documents(chunks)

        # Query with completely different terms
        results = await store.query("zebra elephant giraffe", top_k=3)
        # Should still return documents (all with zero similarity)
        # or return empty list if no similarity threshold is met
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self, store: InMemoryVectorStore) -> None:
        """TF-IDF matching should be case-insensitive."""
        chunks = [
            "Python Programming Language",
            "Java programming language",
        ]
        await store.add_documents(chunks)

        results = await store.query("python programming", top_k=2)
        assert len(results) > 0
        assert "Python" in results[0] or "python" in results[0].lower()

    @pytest.mark.asyncio
    async def test_multiple_add_documents_calls(self, store: InMemoryVectorStore) -> None:
        """Multiple add_documents calls should accumulate documents."""
        await store.add_documents(["First batch document"])
        await store.add_documents(["Second batch document"])

        results = await store.query("document", top_k=5)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_query_string(self, store: InMemoryVectorStore) -> None:
        """Empty query string should return empty list or handle gracefully."""
        chunks = ["Document 1", "Document 2"]
        await store.add_documents(chunks)

        results = await store.query("", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_add_empty_chunks_list(self, store: InMemoryVectorStore) -> None:
        """Adding empty list should not raise error."""
        await store.add_documents([])

        results = await store.query("test", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_max_documents_default_value(self) -> None:
        """InMemoryVectorStore should have default max_documents of 10000."""
        store = InMemoryVectorStore()
        # Access the max_documents parameter (will fail until implemented)
        assert store.max_documents == 10000

    @pytest.mark.asyncio
    async def test_max_documents_custom_value(self) -> None:
        """InMemoryVectorStore should accept custom max_documents parameter."""
        store = InMemoryVectorStore(max_documents=100)
        assert store.max_documents == 100

    @pytest.mark.asyncio
    async def test_fifo_eviction_when_exceeding_max_documents(self) -> None:
        """Documents exceeding max_documents should be evicted FIFO (oldest first)."""
        store = InMemoryVectorStore(max_documents=3)

        # Add 3 documents (should all be kept)
        await store.add_documents(["doc1", "doc2", "doc3"])
        results = await store.query("doc", top_k=10)
        assert len(results) == 3
        assert "doc1" in results
        assert "doc2" in results
        assert "doc3" in results

        # Add 2 more documents (should evict doc1 and doc2, keeping doc3, doc4, doc5)
        await store.add_documents(["doc4", "doc5"])
        results = await store.query("doc", top_k=10)
        assert len(results) == 3
        assert "doc1" not in results  # Oldest, should be evicted
        assert "doc2" not in results  # Second oldest, should be evicted
        assert "doc3" in results  # Should be kept
        assert "doc4" in results  # Should be kept
        assert "doc5" in results  # Should be kept

    @pytest.mark.asyncio
    async def test_fifo_eviction_keeps_last_max_documents(self) -> None:
        """After eviction, exactly max_documents should remain (the most recent ones)."""
        store = InMemoryVectorStore(max_documents=5)

        # Add 10 documents in batches
        await store.add_documents([f"batch1_doc{i}" for i in range(3)])
        await store.add_documents([f"batch2_doc{i}" for i in range(4)])
        await store.add_documents([f"batch3_doc{i}" for i in range(3)])

        # Should keep only the last 5 documents
        results = await store.query("doc", top_k=10)
        assert len(results) == 5

        # First 5 documents should be evicted
        assert "batch1_doc0" not in results
        assert "batch1_doc1" not in results
        assert "batch1_doc2" not in results
        assert "batch2_doc0" not in results
        assert "batch2_doc1" not in results

        # Last 5 documents should be kept
        assert "batch2_doc2" in results
        assert "batch2_doc3" in results
        assert "batch3_doc0" in results
        assert "batch3_doc1" in results
        assert "batch3_doc2" in results

    @pytest.mark.asyncio
    async def test_no_eviction_when_below_max_documents(self) -> None:
        """No eviction should occur when document count is below max_documents."""
        store = InMemoryVectorStore(max_documents=10)

        # Add 5 documents (below limit)
        await store.add_documents([f"doc{i}" for i in range(5)])

        # All should be present
        results = await store.query("doc", top_k=10)
        assert len(results) == 5
        for i in range(5):
            assert f"doc{i}" in results

    @pytest.mark.asyncio
    async def test_top_k_must_be_at_least_one(self, store: InMemoryVectorStore) -> None:
        """Query must raise ValueError when top_k is less than 1."""
        await store.add_documents(["Document 1", "Document 2"])

        with pytest.raises(ValueError, match="top_k must be at least 1"):
            await store.query("Document", top_k=0)

        with pytest.raises(ValueError, match="top_k must be at least 1"):
            await store.query("Document", top_k=-1)

        with pytest.raises(ValueError, match="top_k must be at least 1"):
            await store.query("Document", top_k=-100)

    @pytest.mark.asyncio
    async def test_top_k_cannot_exceed_1000(self, store: InMemoryVectorStore) -> None:
        """Query must raise ValueError when top_k exceeds 1000."""
        await store.add_documents(["Document 1", "Document 2"])

        with pytest.raises(ValueError, match="top_k cannot exceed 1000"):
            await store.query("Document", top_k=1001)

        with pytest.raises(ValueError, match="top_k cannot exceed 1000"):
            await store.query("Document", top_k=5000)

    @pytest.mark.asyncio
    async def test_top_k_boundary_values(self, store: InMemoryVectorStore) -> None:
        """Query must accept boundary values top_k=1 and top_k=1000."""
        await store.add_documents(["Document 1", "Document 2"])

        # top_k=1 should work (minimum valid value)
        results = await store.query("Document", top_k=1)
        assert len(results) == 1

        # top_k=1000 should work (maximum valid value)
        results = await store.query("Document", top_k=1000)
        assert len(results) == 2  # Only 2 documents available
