"""Unit tests for ChromaVectorStore with embedding-based semantic search.

Task 17.3: Add embedding-based VectorStore (Chroma) to replace TF-IDF with semantic search.

Tests cover:
- Protocol interface implementation
- Basic CRUD operations (add_documents, query, clear)
- Semantic search capabilities (synonyms, paraphrases)
- Configuration and initialization
- Edge cases and error handling
"""

import uuid

import pytest

from app.stores.vector_store import ChromaVectorStore


@pytest.fixture
def chroma_store() -> ChromaVectorStore:
    """Create a ChromaVectorStore with a unique collection name for test isolation.

    Each test gets a fresh store with its own collection to prevent data
    leakage between tests.
    """
    # Use a unique collection name to ensure test isolation
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    return ChromaVectorStore(collection_name=collection_name)


class TestChromaVectorStoreProtocol:
    """Test ChromaVectorStore implements VectorStore Protocol."""

    def test_chroma_vector_store_implements_protocol(self, chroma_store: ChromaVectorStore) -> None:
        """ChromaVectorStore implements all VectorStore Protocol methods."""
        assert hasattr(chroma_store, "add_documents")
        assert hasattr(chroma_store, "query")
        assert hasattr(chroma_store, "clear")


class TestChromaVectorStoreConstruction:
    """Test ChromaVectorStore initialization and configuration."""

    def test_default_initialization(self, chroma_store: ChromaVectorStore) -> None:
        """Store initializes with default parameters."""
        assert chroma_store is not None

    def test_custom_collection_name(self) -> None:
        """Store accepts custom collection name parameter."""
        store = ChromaVectorStore(collection_name="test_collection")
        assert store.collection_name == "test_collection"

    def test_custom_embedding_model(self) -> None:
        """Store accepts custom embedding model parameter."""
        store = ChromaVectorStore(embedding_model="all-MiniLM-L6-v2")
        assert store.embedding_model == "all-MiniLM-L6-v2"


class TestAddDocuments:
    """Test add_documents() operation with embeddings."""

    @pytest.mark.asyncio
    async def test_add_single_document(self, chroma_store: ChromaVectorStore) -> None:
        """Store accepts single document and generates embeddings."""
        await chroma_store.add_documents(["First document"])
        results = await chroma_store.query("First", top_k=1)
        assert len(results) == 1
        assert results[0] == "First document"

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, chroma_store: ChromaVectorStore) -> None:
        """Store accepts multiple documents."""
        await chroma_store.add_documents(["Doc one", "Doc two", "Doc three"])
        results = await chroma_store.query("Doc", top_k=10)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_add_empty_list(self, chroma_store: ChromaVectorStore) -> None:
        """Store accepts empty list without error."""
        await chroma_store.add_documents([])
        results = await chroma_store.query("test", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_add_multiple_batches(self, chroma_store: ChromaVectorStore) -> None:
        """Store accumulates documents from multiple add_documents calls."""
        await chroma_store.add_documents(["Doc one"])
        await chroma_store.add_documents(["Doc two"])
        results = await chroma_store.query("Doc", top_k=10)
        assert len(results) == 2


class TestSemanticSearch:
    """Test semantic search capabilities using embeddings.

    These tests verify that ChromaVectorStore provides better semantic
    matching than TF-IDF by understanding synonyms and paraphrases.
    """

    @pytest.mark.asyncio
    async def test_synonym_matching(self, chroma_store: ChromaVectorStore) -> None:
        """Store matches documents with synonyms of query terms."""
        await chroma_store.add_documents(
            [
                "The automobile is fast",
                "The bicycle is slow",
                "The airplane is expensive",
            ]
        )
        # Query for "car" should match "automobile" (synonym)
        results = await chroma_store.query("car", top_k=1)
        assert len(results) == 1
        assert "automobile" in results[0]

    @pytest.mark.asyncio
    async def test_paraphrase_matching(self, chroma_store: ChromaVectorStore) -> None:
        """Store matches documents with similar meaning but different words."""
        await chroma_store.add_documents(
            [
                "Machine learning is a subset of artificial intelligence",
                "Dogs are loyal pets",
                "Python is a programming language",
            ]
        )
        # Query with paraphrase should match the first document
        results = await chroma_store.query("AI includes machine learning", top_k=1)
        assert len(results) == 1
        assert "artificial intelligence" in results[0]

    @pytest.mark.asyncio
    async def test_semantic_ranking(self, chroma_store: ChromaVectorStore) -> None:
        """Documents are ranked by semantic similarity, not just keyword overlap."""
        await chroma_store.add_documents(
            [
                "Climate change threatens polar bears",
                "Global warming affects Arctic wildlife",
                "The weather today is sunny",
            ]
        )
        # Query should rank semantically similar docs higher
        results = await chroma_store.query("environmental impact on polar regions", top_k=3)
        # First two docs are semantically similar to the query
        assert "polar bears" in results[0] or "Arctic wildlife" in results[0]
        assert "polar bears" in results[1] or "Arctic wildlife" in results[1]
        # Weather doc should rank lower despite having some overlap
        assert "sunny" in results[2]


class TestQuery:
    """Test query() operation with embedding-based retrieval."""

    @pytest.mark.asyncio
    async def test_query_empty_corpus_returns_empty(self, chroma_store: ChromaVectorStore) -> None:
        """Query on empty corpus returns empty list."""
        results = await chroma_store.query("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_empty_string_returns_empty(self, chroma_store: ChromaVectorStore) -> None:
        """Empty query string returns empty list."""
        await chroma_store.add_documents(["Some document"])
        results = await chroma_store.query("", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_returns_up_to_top_k(self, chroma_store: ChromaVectorStore) -> None:
        """Query returns at most top_k results."""
        await chroma_store.add_documents([f"doc{i}" for i in range(10)])
        results = await chroma_store.query("doc", top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_query_returns_fewer_than_top_k_if_corpus_smaller(
        self, chroma_store: ChromaVectorStore
    ) -> None:
        """Query returns all documents if corpus smaller than top_k."""
        await chroma_store.add_documents(["doc1", "doc2"])
        results = await chroma_store.query("doc", top_k=10)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_top_k_validation_minimum(self, chroma_store: ChromaVectorStore) -> None:
        """Query rejects top_k < 1."""
        await chroma_store.add_documents(["test"])
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            await chroma_store.query("test", top_k=0)


class TestClear:
    """Test clear() operation."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_documents(self, chroma_store: ChromaVectorStore) -> None:
        """Clear removes all documents from store."""
        await chroma_store.add_documents(["doc1", "doc2", "doc3"])
        await chroma_store.clear()
        results = await chroma_store.query("doc", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_clear_on_empty_store(self, chroma_store: ChromaVectorStore) -> None:
        """Clear on empty store does not raise error."""
        await chroma_store.clear()
        results = await chroma_store.query("test", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_add_after_clear(self, chroma_store: ChromaVectorStore) -> None:
        """Documents can be added after clearing."""
        await chroma_store.add_documents(["old doc"])
        await chroma_store.clear()
        await chroma_store.add_documents(["new doc"])
        results = await chroma_store.query("doc", top_k=5)
        assert len(results) == 1
        assert results[0] == "new doc"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_unicode_text_handling(self, chroma_store: ChromaVectorStore) -> None:
        """Store handles Unicode text correctly."""
        await chroma_store.add_documents(
            [
                "Hello 世界",
                "Python プログラミング",
                "Machine 学習",
            ]
        )
        results = await chroma_store.query("世界", top_k=1)
        assert len(results) == 1
        assert "Hello 世界" in results[0]

    @pytest.mark.asyncio
    async def test_long_document_handling(self, chroma_store: ChromaVectorStore) -> None:
        """Store handles long documents correctly."""
        long_doc = "word " * 1000  # 1000 words
        await chroma_store.add_documents([long_doc, "short doc"])
        results = await chroma_store.query("word", top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_string_document(self, chroma_store: ChromaVectorStore) -> None:
        """Empty string document is handled gracefully."""
        await chroma_store.add_documents(["", "valid doc"])
        results = await chroma_store.query("valid", top_k=2)
        # Should return at least the valid doc
        assert len(results) >= 1
        assert "valid doc" in results
