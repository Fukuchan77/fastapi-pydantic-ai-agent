"""Unit tests for query string length validation in InMemoryVectorStore."""

import pytest

from app.stores.vector_store import InMemoryVectorStore


class TestQueryLengthValidation:
    """Test query string length validation to prevent DoS attacks."""

    @pytest.fixture
    def store(self) -> InMemoryVectorStore:
        """Provide a fresh InMemoryVectorStore instance with sample data."""
        store = InMemoryVectorStore()
        return store

    @pytest.mark.asyncio
    async def test_query_length_under_limit_succeeds(self, store: InMemoryVectorStore) -> None:
        """Query with length under 10000 chars should succeed."""
        await store.add_documents(["test document"])

        # Create a query just under the limit (9999 chars)
        long_query = "a" * 9999
        # Should not raise
        results = await store.query(long_query, top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_query_length_at_limit_succeeds(self, store: InMemoryVectorStore) -> None:
        """Query with exactly 10000 chars should succeed."""
        await store.add_documents(["test document"])

        # Create a query exactly at the limit (10000 chars)
        long_query = "a" * 10000
        # Should not raise
        results = await store.query(long_query, top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_query_length_exceeds_limit_raises_error(
        self, store: InMemoryVectorStore
    ) -> None:
        """Query exceeding 10000 chars should raise ValueError."""
        await store.add_documents(["test document"])

        # Create a query over the limit (10001 chars)
        oversized_query = "a" * 10001

        with pytest.raises(ValueError, match="Query string too long \\(max 10000 chars\\)"):
            await store.query(oversized_query, top_k=5)

    @pytest.mark.asyncio
    async def test_query_length_far_exceeds_limit_raises_error(
        self, store: InMemoryVectorStore
    ) -> None:
        """Query far exceeding limit should raise ValueError."""
        await store.add_documents(["test document"])

        # Create a very large query (50000 chars)
        oversized_query = "a" * 50000

        with pytest.raises(ValueError, match="Query string too long \\(max 10000 chars\\)"):
            await store.query(oversized_query, top_k=5)
