"""Unit tests for InMemoryVectorStore per-chunk size validation."""

import pytest

from app.stores.vector_store import InMemoryVectorStore


class TestVectorStoreChunkSizeValidation:
    """Test per-chunk size validation in InMemoryVectorStore."""

    @pytest.mark.asyncio
    async def test_oversized_chunk_raises_value_error(self) -> None:
        """Adding a chunk exceeding max_chunk_size must raise ValueError."""
        store = InMemoryVectorStore()

        # Create a chunk larger than the default 100,000 chars
        oversized_chunk = "x" * 100_001

        with pytest.raises(ValueError, match=r"Document chunk too large \(max \d+ chars\)"):
            await store.add_documents([oversized_chunk])

    @pytest.mark.asyncio
    async def test_chunk_exactly_at_limit_passes(self) -> None:
        """Adding a chunk exactly at max_chunk_size limit must succeed."""
        store = InMemoryVectorStore()

        # Create a chunk exactly at the 100,000 char limit
        chunk_at_limit = "x" * 100_000

        # Should not raise
        await store.add_documents([chunk_at_limit])

        # Verify it was added
        results = await store.query("x", top_k=1)
        assert len(results) == 1
        assert len(results[0]) == 100_000

    @pytest.mark.asyncio
    async def test_empty_string_chunk_passes(self) -> None:
        """Adding an empty string chunk must succeed."""
        store = InMemoryVectorStore()

        # Empty string should be allowed
        await store.add_documents([""])

        # Verify store accepted it (though query won't return empty strings)
        # The fact that no exception was raised is the main assertion

    @pytest.mark.asyncio
    async def test_small_max_chunk_size_via_constructor(self) -> None:
        """Custom max_chunk_size passed via constructor must be enforced."""
        store = InMemoryVectorStore(max_chunk_size=50)

        # Chunk within limit should pass
        small_chunk = "x" * 50
        await store.add_documents([small_chunk])

        # Chunk exceeding custom limit should raise
        oversized_chunk = "x" * 51
        with pytest.raises(ValueError, match=r"Document chunk too large \(max 50 chars\)"):
            await store.add_documents([oversized_chunk])

    @pytest.mark.asyncio
    async def test_mixed_chunks_fails_on_first_violation(self) -> None:
        """When adding multiple chunks, validation fails on first oversized chunk."""
        store = InMemoryVectorStore(max_chunk_size=100)

        chunks = [
            "valid chunk",  # Valid
            "x" * 101,  # Invalid - should trigger error
            "another valid chunk",  # Should not be processed
        ]

        with pytest.raises(ValueError, match=r"Document chunk too large \(max 100 chars\)"):
            await store.add_documents(chunks)

        # Verify no chunks were added (atomic failure)
        results = await store.query("valid", top_k=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_max_chunk_size_default_value(self) -> None:
        """InMemoryVectorStore should have default max_chunk_size of 100_000."""
        store = InMemoryVectorStore()
        assert store.max_chunk_size == 100_000

    @pytest.mark.asyncio
    async def test_boundary_values_with_default_limit(self) -> None:
        """Test chunks at and near the default 100,000 character limit."""
        store = InMemoryVectorStore()

        # Just under limit - should pass
        chunk_under = "x" * 99_999
        await store.add_documents([chunk_under])

        # Exactly at limit - should pass
        chunk_at = "x" * 100_000
        await store.add_documents([chunk_at])

        # Just over limit - should fail
        chunk_over = "x" * 100_001
        with pytest.raises(ValueError, match=r"Document chunk too large"):
            await store.add_documents([chunk_over])
