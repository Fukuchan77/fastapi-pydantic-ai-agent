"""Unit tests for VectorStore Protocol and InMemoryVectorStore.

Tests cover:
- Protocol interface definition
- TF-IDF ranking correctness
- CRUD operations (add_documents, query, clear)
- FIFO eviction when max_documents exceeded
- Chunk size validation
- Query parameter validation
- Edge cases (empty query, empty corpus)
- IDF caching behavior
"""

import math

import pytest

from app.stores.vector_store import InMemoryVectorStore
from app.stores.vector_store import VectorStore


class TestVectorStoreProtocol:
    """Test VectorStore Protocol interface definition."""

    def test_protocol_has_add_documents_method(self) -> None:
        """VectorStore Protocol defines add_documents method."""
        assert hasattr(VectorStore, "add_documents")

    def test_protocol_has_query_method(self) -> None:
        """VectorStore Protocol defines query method."""
        assert hasattr(VectorStore, "query")

    def test_protocol_has_clear_method(self) -> None:
        """VectorStore Protocol defines clear method."""
        assert hasattr(VectorStore, "clear")

    def test_in_memory_vector_store_implements_protocol(self) -> None:
        """InMemoryVectorStore implements all VectorStore Protocol methods."""
        store = InMemoryVectorStore()
        assert hasattr(store, "add_documents")
        assert hasattr(store, "query")
        assert hasattr(store, "clear")


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


class TestQuery:
    """Test query() operation and TF-IDF ranking."""

    @pytest.mark.asyncio
    async def test_query_empty_corpus_returns_empty(self) -> None:
        """Query on empty corpus returns empty list."""
        store = InMemoryVectorStore()
        results = await store.query("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_empty_string_returns_empty(self) -> None:
        """Empty query string returns empty list."""
        store = InMemoryVectorStore()
        await store.add_documents(["Some document"])
        results = await store.query("", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only query returns empty list."""
        store = InMemoryVectorStore()
        await store.add_documents(["Some document"])
        results = await store.query("   \t\n  ", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_returns_up_to_top_k(self) -> None:
        """Query returns at most top_k results."""
        store = InMemoryVectorStore()
        await store.add_documents([f"doc{i}" for i in range(10)])
        results = await store.query("doc", top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_returns_fewer_than_top_k_if_corpus_smaller(self) -> None:
        """Query returns all documents if corpus smaller than top_k."""
        store = InMemoryVectorStore()
        await store.add_documents(["doc1", "doc2"])
        results = await store.query("doc", top_k=10)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_top_k_validation_minimum(self) -> None:
        """Query rejects top_k < 1."""
        store = InMemoryVectorStore()
        await store.add_documents(["test"])
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            await store.query("test", top_k=0)

    @pytest.mark.asyncio
    async def test_query_top_k_validation_maximum(self) -> None:
        """Query rejects top_k > 1000."""
        store = InMemoryVectorStore()
        await store.add_documents(["test"])
        with pytest.raises(ValueError, match="top_k cannot exceed 1000"):
            await store.query("test", top_k=1001)

    @pytest.mark.asyncio
    async def test_query_top_k_boundary_values(self) -> None:
        """Query accepts top_k boundary values 1 and 1000."""
        store = InMemoryVectorStore()
        await store.add_documents(["test document"])
        # top_k = 1 (minimum)
        results = await store.query("test", top_k=1)
        assert len(results) == 1
        # top_k = 1000 (maximum)
        results = await store.query("test", top_k=1000)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_length_validation(self) -> None:
        """Query rejects queries exceeding 10000 characters."""
        store = InMemoryVectorStore()
        await store.add_documents(["test"])
        long_query = "x" * 10001
        with pytest.raises(ValueError, match="Query string too long"):
            await store.query(long_query, top_k=5)

    @pytest.mark.asyncio
    async def test_query_length_at_boundary(self) -> None:
        """Query accepts queries exactly at 10000 character limit."""
        store = InMemoryVectorStore()
        await store.add_documents(["x" * 100])
        query = "x" * 10000
        results = await store.query(query, top_k=5)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_character_limit_hit_before_token_limit(self) -> None:
        """Query character limit (10000) is hit before token limit (10000 tokens).

        With whitespace tokenization, token count is always <= character count,
        so the character validation is the effective limit. Token validation
        provides defense-in-depth for future tokenization changes.
        """
        store = InMemoryVectorStore()
        await store.add_documents(["test"])
        # Create query with 10001 chars (will hit character limit first)
        long_query = "x" * 10001
        with pytest.raises(ValueError, match="Query string too long"):
            await store.query(long_query, top_k=5)


class TestTFIDFRanking:
    """Test TF-IDF ranking correctness.

    These tests verify that the TF-IDF algorithm correctly ranks documents
    by relevance, which is the core requirement from tasks.md section 12.8.
    """

    @pytest.mark.asyncio
    async def test_exact_match_ranks_highest(self) -> None:
        """Document with exact query match ranks highest."""
        store = InMemoryVectorStore()
        await store.add_documents(
            [
                "machine learning algorithms",
                "deep neural networks",
                "natural language processing",
            ]
        )
        results = await store.query("machine learning", top_k=3)
        # Document containing both "machine" and "learning" should rank first
        assert results[0] == "machine learning algorithms"

    @pytest.mark.asyncio
    async def test_partial_match_ranks_lower(self) -> None:
        """Documents with partial matches rank lower than exact matches."""
        store = InMemoryVectorStore()
        await store.add_documents(
            [
                "python programming language",
                "python snake species",
                "java programming language",
            ]
        )
        results = await store.query("python programming", top_k=3)
        # "python programming language" has both terms
        assert results[0] == "python programming language"
        # Other docs have only one term each
        assert "python snake species" in results[1:]
        assert "java programming language" in results[1:]

    @pytest.mark.asyncio
    async def test_term_frequency_affects_ranking(self) -> None:
        """Term frequency affects ranking when combined with other differentiating terms.

        Note: Cosine similarity normalizes by vector magnitude, so documents that differ
        only in the frequency of a single term will have equal cosine similarity (angle).
        To observe TF effects, we need documents with other differentiating terms.

        This test uses a multi-term query where TF differences combine with other
        term presence to create meaningful angle differences.
        """
        store = InMemoryVectorStore()
        await store.add_documents(
            [
                "python programming language tutorial",  # 4 terms
                "python python programming guide",  # "python" appears twice (higher TF)
                "java programming language tutorial",  # Different primary term
            ]
        )
        results = await store.query("python programming", top_k=3)
        # Doc with "python python programming" has highest combined relevance
        # for both query terms (python appears twice, programming appears once)
        assert results[0] == "python python programming guide"
        # Doc with "python programming" both once ranks second
        assert results[1] == "python programming language tutorial"
        # Doc with only "programming" (missing "python") ranks third
        assert results[2] == "java programming language tutorial"

    @pytest.mark.asyncio
    async def test_idf_weights_rare_terms_higher(self) -> None:
        """Rare terms (appearing in fewer documents) get higher IDF weights."""
        store = InMemoryVectorStore()
        # "unique" appears in only 1 doc, "common" appears in all 3
        await store.add_documents(
            [
                "unique term here",
                "common term",
                "common term again",
            ]
        )
        results = await store.query("unique common", top_k=3)
        # Doc with rare term "unique" should rank highest due to higher IDF
        assert results[0] == "unique term here"

    @pytest.mark.asyncio
    async def test_cosine_similarity_normalization(self) -> None:
        """Cosine similarity normalizes by document length."""
        store = InMemoryVectorStore()
        await store.add_documents(
            [
                "test",  # Short doc with "test"
                "test " * 100,  # Long doc repeating "test" 100 times
            ]
        )
        results = await store.query("test", top_k=2)
        # Both docs should rank similarly (cosine normalizes by length)
        # The order might vary slightly due to floating point, so just check both present
        assert len(results) == 2
        assert "test" in results
        assert "test " * 100 in results

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self) -> None:
        """Query matching is case-insensitive."""
        store = InMemoryVectorStore()
        await store.add_documents(
            [
                "Python Programming",
                "PYTHON LANGUAGE",
                "python script",
            ]
        )
        results = await store.query("PYTHON", top_k=3)
        # All three documents should match (case-insensitive)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_with_no_matching_terms(self) -> None:
        """Query with no matching terms returns documents with zero scores."""
        store = InMemoryVectorStore()
        await store.add_documents(
            [
                "cat dog bird",
                "fish turtle snake",
            ]
        )
        results = await store.query("elephant tiger", top_k=5)
        # Still returns documents even though no terms match
        # (cosine similarity will be 0.0 for all, but they're still returned)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_stopword_like_behavior(self) -> None:
        """Common terms appearing in all documents get low IDF weights."""
        store = InMemoryVectorStore()
        await store.add_documents(
            [
                "the cat sat on the mat",
                "the dog ran in the park",
                "the bird flew over the tree",
            ]
        )
        results = await store.query("cat", top_k=3)
        # "cat" is unique to first doc, should rank highest despite "the" appearing everywhere
        assert results[0] == "the cat sat on the mat"


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


class TestIDFCaching:
    """Test IDF caching behavior."""

    @pytest.mark.asyncio
    async def test_idf_cache_populated_on_first_query(self) -> None:
        """IDF cache is populated on first query after add_documents."""
        store = InMemoryVectorStore()
        await store.add_documents(["test document"])
        # IDF cache should be None before first query
        assert store._idf_cache is None
        await store.query("test", top_k=1)
        # IDF cache should be populated after first query
        assert store._idf_cache is not None
        assert isinstance(store._idf_cache, dict)

    @pytest.mark.asyncio
    async def test_idf_cache_reused_on_subsequent_queries(self) -> None:
        """IDF cache is reused on subsequent queries (performance optimization)."""
        store = InMemoryVectorStore()
        await store.add_documents(["doc one", "doc two"])
        await store.query("doc", top_k=1)
        first_cache = store._idf_cache
        await store.query("one two", top_k=1)
        second_cache = store._idf_cache
        # Cache should be the same object (not recomputed)
        assert first_cache is second_cache

    @pytest.mark.asyncio
    async def test_idf_cache_invalidated_on_add_documents(self) -> None:
        """IDF cache is invalidated when new documents are added."""
        store = InMemoryVectorStore()
        await store.add_documents(["first doc"])
        await store.query("first", top_k=1)
        first_cache = store._idf_cache
        # Add more documents
        await store.add_documents(["second doc"])
        # Cache should be invalidated (set to None)
        assert store._idf_cache is None
        # Query should repopulate cache
        await store.query("doc", top_k=2)
        second_cache = store._idf_cache
        # New cache should be different object
        assert first_cache is not second_cache

    @pytest.mark.asyncio
    async def test_idf_values_update_with_corpus_changes(self) -> None:
        """IDF values change when corpus changes."""
        store = InMemoryVectorStore()
        # Add one document with "rare" term
        await store.add_documents(["rare unique term"])
        await store.query("rare", top_k=1)
        # IDF for "rare" should be log(1/1) = 0.0 (appears in all docs)
        assert store._idf_cache is not None
        first_cache = store._idf_cache
        first_idf = first_cache.get("rare", 0.0)
        assert first_idf == pytest.approx(0.0)
        # Add more documents without "rare"
        await store.add_documents(["common word", "another common word"])
        await store.query("rare common", top_k=3)
        # IDF for "rare" should now be log(3/1) ≈ 1.0986 (appears in 1 of 3 docs)
        assert store._idf_cache is not None
        second_cache = store._idf_cache
        second_idf = second_cache.get("rare", 0.0)
        assert second_idf == pytest.approx(math.log(3 / 1))
        assert second_idf > first_idf


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_document_with_only_whitespace(self) -> None:
        """Document containing only whitespace is stored and returned with zero score.

        The implementation doesn't filter out zero-score documents; it still returns
        them in the results (with cosine similarity = 0.0). This is reasonable behavior
        since filtering would add complexity and the caller can filter if needed.
        """
        store = InMemoryVectorStore()
        await store.add_documents(["   \t\n   "])
        results = await store.query("test", top_k=1)
        # Document is returned even though it has no tokens (cosine similarity = 0.0)
        assert len(results) == 1
        assert results[0] == "   \t\n   "

    @pytest.mark.asyncio
    async def test_query_with_repeated_terms(self) -> None:
        """Query with repeated terms calculates TF correctly."""
        store = InMemoryVectorStore()
        await store.add_documents(
            [
                "cat cat cat",
                "dog dog dog",
            ]
        )
        results = await store.query("cat cat", top_k=2)
        # Doc with "cat" should rank first
        assert results[0] == "cat cat cat"

    @pytest.mark.asyncio
    async def test_unicode_text_handling(self) -> None:
        """Store handles Unicode text correctly."""
        store = InMemoryVectorStore()
        await store.add_documents(
            [
                "Hello 世界",
                "Python プログラミング",
                "Machine 学習",
            ]
        )
        results = await store.query("世界", top_k=1)
        assert len(results) == 1
        assert results[0] == "Hello 世界"

    @pytest.mark.asyncio
    async def test_special_characters_are_preserved(self) -> None:
        """Special characters are preserved in tokenization."""
        store = InMemoryVectorStore()
        await store.add_documents(
            [
                "user@example.com email address",
                "test@domain.org another email",
            ]
        )
        # Note: tokenization splits on whitespace, so "user@example.com" is one token
        results = await store.query("user@example.com", top_k=1)
        assert len(results) == 1
        assert results[0] == "user@example.com email address"

    @pytest.mark.asyncio
    async def test_empty_string_document(self) -> None:
        """Empty string document is stored and returned with zero score.

        Similar to whitespace-only documents, empty documents are returned
        even though they produce no tokens and have zero cosine similarity.
        """
        store = InMemoryVectorStore()
        await store.add_documents([""])
        results = await store.query("test", top_k=1)
        # Document is returned even though it's empty (cosine similarity = 0.0)
        assert len(results) == 1
        assert results[0] == ""

    @pytest.mark.asyncio
    async def test_single_character_tokens(self) -> None:
        """Single character tokens are handled correctly."""
        store = InMemoryVectorStore()
        await store.add_documents(["a b c", "d e f"])
        results = await store.query("a", top_k=1)
        assert len(results) == 1
        assert results[0] == "a b c"
