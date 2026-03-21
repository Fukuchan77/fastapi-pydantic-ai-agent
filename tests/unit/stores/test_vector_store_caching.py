"""Unit tests for VectorStore tokenization and IDF caching optimization."""

import pytest

from app.stores.vector_store import InMemoryVectorStore


class TestVectorStoreCaching:
    """Test that tokenization and IDF computation are cached to avoid redundant work."""

    @pytest.fixture
    def store(self) -> InMemoryVectorStore:
        """Provide a fresh InMemoryVectorStore instance."""
        return InMemoryVectorStore()

    @pytest.mark.asyncio
    async def test_doc_tokens_cached_at_index_time(self, store: InMemoryVectorStore) -> None:
        """Documents must be tokenized once during add_documents, not during query.

        This test verifies that _doc_tokens is populated when documents are added,
        eliminating the need to re-tokenize on every query.
        """
        chunks = ["Python programming", "Machine learning", "Data science"]
        await store.add_documents(chunks)

        # Verify _doc_tokens was populated
        assert hasattr(store, "_doc_tokens"), "_doc_tokens field should exist"
        assert len(store._doc_tokens) == 3, "Should have tokenized all 3 documents"

        # Verify tokens are correct
        assert store._doc_tokens[0] == ["python", "programming"]
        assert store._doc_tokens[1] == ["machine", "learning"]
        assert store._doc_tokens[2] == ["data", "science"]

    @pytest.mark.asyncio
    async def test_doc_tokens_extended_on_multiple_adds(self, store: InMemoryVectorStore) -> None:
        """Multiple add_documents calls must extend the _doc_tokens cache."""
        await store.add_documents(["First batch"])
        assert len(store._doc_tokens) == 1

        await store.add_documents(["Second batch", "Third batch"])
        assert len(store._doc_tokens) == 3

        # Verify all tokens are present
        assert store._doc_tokens[0] == ["first", "batch"]
        assert store._doc_tokens[1] == ["second", "batch"]
        assert store._doc_tokens[2] == ["third", "batch"]

    @pytest.mark.asyncio
    async def test_idf_cache_invalidated_on_add_documents(self, store: InMemoryVectorStore) -> None:
        """add_documents must invalidate IDF cache since corpus changed.

        When documents are added, the IDF weights are no longer valid and
        must be recomputed on the next query.
        """
        # Add initial documents and run a query to populate IDF cache
        await store.add_documents(["Python programming", "Java programming"])
        await store.query("programming", top_k=2)

        # At this point, _idf_cache should be populated
        assert hasattr(store, "_idf_cache"), "_idf_cache field should exist"
        assert store._idf_cache is not None, "IDF cache should be populated after query"
        assert len(store._idf_cache) > 0, "IDF cache should contain weights"

        # Add more documents - this should invalidate the cache
        await store.add_documents(["Ruby programming"])

        # Cache should now be invalidated (set to None)
        assert store._idf_cache is None, "IDF cache must be invalidated when documents are added"

    @pytest.mark.asyncio
    async def test_idf_cache_computed_on_first_query(self, store: InMemoryVectorStore) -> None:
        """IDF cache must be computed on first query and reused on subsequent queries."""
        await store.add_documents(["Python programming", "Machine learning"])

        # Before first query, cache should be None
        assert hasattr(store, "_idf_cache"), "_idf_cache field should exist"
        assert store._idf_cache is None, "IDF cache should be None before first query"

        # First query should compute and cache IDF
        await store.query("programming", top_k=2)
        assert store._idf_cache is not None, "IDF cache should be populated after query"
        assert len(store._idf_cache) > 0, "IDF cache should contain weights"

        # Capture the cached IDF dict
        first_idf_cache = store._idf_cache
        first_idf_id = id(store._idf_cache)

        # Second query should reuse the same cache (same object ID)
        await store.query("learning", top_k=2)
        assert store._idf_cache is not None, "IDF cache should still be populated"
        assert id(store._idf_cache) == first_idf_id, (
            "Second query should reuse cached IDF dict (same object)"
        )
        assert store._idf_cache == first_idf_cache, "IDF cache contents should be unchanged"

    @pytest.mark.asyncio
    async def test_query_uses_cached_doc_tokens(self, store: InMemoryVectorStore) -> None:
        """query() must use cached _doc_tokens instead of re-tokenizing documents.

        This test verifies that the tokenization work is done once at index time,
        not repeatedly at query time.
        """
        chunks = ["Python programming language", "Java programming language"]
        await store.add_documents(chunks)

        # Verify tokens were cached at index time
        assert len(store._doc_tokens) == 2
        cached_tokens = store._doc_tokens.copy()

        # Run query - should use cached tokens, not re-tokenize
        await store.query("programming", top_k=2)

        # Verify _doc_tokens hasn't changed (query didn't modify it)
        assert store._doc_tokens == cached_tokens, "Cached tokens should be unchanged"

    @pytest.mark.asyncio
    async def test_clear_resets_caches(self, store: InMemoryVectorStore) -> None:
        """clear() must reset both _doc_tokens and _idf_cache."""
        # Populate store and caches
        await store.add_documents(["Document one", "Document two"])
        await store.query("document", top_k=2)

        # Verify caches are populated
        assert len(store._doc_tokens) == 2
        assert store._idf_cache is not None

        # Clear the store
        await store.clear()

        # Verify caches are reset
        assert len(store._doc_tokens) == 0, "_doc_tokens should be empty after clear"
        assert store._idf_cache is None, "_idf_cache should be None after clear (or empty dict)"

    @pytest.mark.asyncio
    async def test_fifo_eviction_maintains_doc_tokens_sync(
        self, store: InMemoryVectorStore
    ) -> None:
        """FIFO eviction must keep _doc_tokens synchronized with _documents.

        When documents are evicted due to max_documents limit, the corresponding
        tokenized entries must also be evicted to maintain sync.
        """
        store = InMemoryVectorStore(max_documents=3)

        # Add 3 documents
        await store.add_documents(["doc1 first", "doc2 second", "doc3 third"])
        assert len(store._documents) == 3
        assert len(store._doc_tokens) == 3

        # Add 2 more - should evict first 2 documents
        await store.add_documents(["doc4 fourth", "doc5 fifth"])
        assert len(store._documents) == 3
        assert len(store._doc_tokens) == 3, "_doc_tokens must stay in sync with _documents"

        # Verify the correct documents and tokens remain (last 3)
        assert "doc3 third" in store._documents
        assert "doc4 fourth" in store._documents
        assert "doc5 fifth" in store._documents

        # Verify corresponding tokens are present
        assert ["doc3", "third"] in store._doc_tokens
        assert ["doc4", "fourth"] in store._doc_tokens
        assert ["doc5", "fifth"] in store._doc_tokens
