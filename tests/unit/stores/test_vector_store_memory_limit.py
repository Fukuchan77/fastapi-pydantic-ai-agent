"""Tests for VectorStore memory limit enforcement.

These tests verify that the VectorStore has reasonable default limits
to prevent excessive memory consumption (up to 10GB).
"""

import pytest

from app.stores.vector_store import InMemoryVectorStore


class TestVectorStoreMemoryLimits:
    """Test suite for memory limit enforcement in VectorStore."""

    def test_default_max_documents_prevents_excessive_memory(self):
        """Default max_documents should prevent >1GB memory consumption.

        Memory calculation:
        - Each document can be up to max_chunk_size (100,000 chars)
        - Python strings: ~2 bytes/char average (Unicode)
        - Tokenized copies: ~1 byte/char (tokens are references)
        - IDF cache: negligible for reasonable vocab sizes

        Total per document: ~300KB (100K chars x 3x overhead)
        Safe limit for 1GB: ~3,300 documents

        This test verifies max_documents <= 1000 to stay well under 1GB.
        """
        store = InMemoryVectorStore()

        # Default max_documents should be 1000 or less for memory safety
        assert store.max_documents <= 1000, (
            f"max_documents={store.max_documents} could cause excessive memory. "
            f"Expected <= 1000 to prevent >1GB consumption."
        )

    def test_default_max_chunk_size_is_reasonable(self):
        """Default max_chunk_size should remain at 100KB."""
        store = InMemoryVectorStore()

        # Verify chunk size limit is still enforced
        assert store.max_chunk_size == 100_000

    def test_max_memory_calculation_with_defaults(self):
        """Verify theoretical max memory with defaults is under 1GB.

        Conservative estimate:
        - Raw documents: max_documents x max_chunk_size x 2 bytes/char
        - Tokenized copies: max_documents x max_chunk_size x 1 byte/token
        - Total: max_documents x max_chunk_size x 3 bytes
        """
        store = InMemoryVectorStore()

        # Calculate theoretical max memory (bytes)
        max_memory_bytes = store.max_documents * store.max_chunk_size * 3
        max_memory_gb = max_memory_bytes / (1024**3)

        # Should be under 1GB
        assert max_memory_gb < 1.0, (
            f"Theoretical max memory is {max_memory_gb:.2f}GB, "
            f"which exceeds the 1GB safety limit. "
            f"max_documents={store.max_documents}, "
            f"max_chunk_size={store.max_chunk_size}"
        )

    def test_custom_limits_can_be_set(self):
        """Users should be able to set custom limits if needed."""
        # Test with stricter limits
        store = InMemoryVectorStore(max_documents=100, max_chunk_size=10_000)
        assert store.max_documents == 100
        assert store.max_chunk_size == 10_000

        # Test with higher limits (user accepts memory risk)
        store = InMemoryVectorStore(max_documents=5000, max_chunk_size=200_000)
        assert store.max_documents == 5000
        assert store.max_chunk_size == 200_000

    @pytest.mark.asyncio
    async def test_fifo_eviction_prevents_unbounded_growth(self):
        """FIFO eviction should prevent memory growth beyond max_documents."""
        store = InMemoryVectorStore(max_documents=100)

        # Add more documents than the limit
        chunks = [f"document {i}" for i in range(150)]
        await store.add_documents(chunks)

        # Should only retain the last 100 (FIFO eviction)
        assert len(store._documents) == 100
        assert len(store._doc_tokens) == 100

        # Verify oldest documents were evicted
        assert store._documents[0] == "document 50"
        assert store._documents[-1] == "document 149"

    @pytest.mark.asyncio
    async def test_chunk_size_enforcement_prevents_dos(self):
        """Oversized chunks should be rejected to prevent memory exhaustion."""
        store = InMemoryVectorStore(max_chunk_size=1000)

        # Try to add a chunk that exceeds the limit
        oversized_chunk = "x" * 1001

        with pytest.raises(ValueError, match="Document chunk too large"):
            await store.add_documents([oversized_chunk])

        # Store should remain empty
        assert len(store._documents) == 0
