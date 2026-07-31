"""Unit tests for InMemoryVectorStore.query_with_scores() (citation substrate, Req 3.3/3.8).

Split out of test_vector_store.py per `.sdd/steering/file-size-policy.md`.
"""

import pytest

from app.models.rag import RetrievedHit
from app.stores.vector_store import InMemoryVectorStore
from app.stores.vector_store import VectorStore


class TestQueryWithScores:
    """Test query_with_scores() operation (citation substrate, Req 3.3/3.8)."""

    def test_protocol_has_query_with_scores_method(self) -> None:
        """VectorStore Protocol defines query_with_scores method."""
        assert hasattr(VectorStore, "query_with_scores")

    @pytest.mark.asyncio
    async def test_returns_retrieved_hit_instances(self) -> None:
        """query_with_scores returns a list of RetrievedHit."""
        store = InMemoryVectorStore()
        await store.add_documents(["First document", "Unrelated other text"])
        results = await store.query_with_scores("First", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedHit)
        assert results[0].text == "First document"
        assert results[0].score > 0.0

    @pytest.mark.asyncio
    async def test_chunk_id_format_is_source_ordinal(self) -> None:
        """chunk_id follows the 'source::NNNN' format with zero-padded ordinal."""
        store = InMemoryVectorStore()
        await store.add_documents(["First document"])
        results = await store.query_with_scores("First", top_k=1)
        assert results[0].chunk_id == "memory::0000"

    @pytest.mark.asyncio
    async def test_chunk_id_ordinals_are_stable_and_distinct_per_document(self) -> None:
        """Each document keeps a distinct, stable ordinal across repeated queries."""
        store = InMemoryVectorStore()
        await store.add_documents(["Doc one", "Doc two", "Doc three"])

        first_pass = await store.query_with_scores("Doc", top_k=10)
        second_pass = await store.query_with_scores("Doc", top_k=10)

        first_ids = {hit.text: hit.chunk_id for hit in first_pass}
        second_ids = {hit.text: hit.chunk_id for hit in second_pass}
        assert first_ids == second_ids
        assert len(set(first_ids.values())) == 3
        assert first_ids["Doc one"] == "memory::0000"
        assert first_ids["Doc two"] == "memory::0001"
        assert first_ids["Doc three"] == "memory::0002"

    @pytest.mark.asyncio
    async def test_chunk_id_survives_fifo_eviction_of_older_documents(self) -> None:
        """A surviving document's ordinal is unaffected by eviction of older ones."""
        store = InMemoryVectorStore(max_documents=2)
        await store.add_documents(["Doc one"])
        await store.add_documents(["Doc two"])
        await store.add_documents(["Doc three"])  # evicts "Doc one"

        results = await store.query_with_scores("Doc three", top_k=1)
        assert results[0].chunk_id == "memory::0002"

    @pytest.mark.asyncio
    async def test_results_ranked_by_score_descending(self) -> None:
        """query_with_scores ranks hits by score, highest first."""
        store = InMemoryVectorStore()
        await store.add_documents(["apple banana", "apple", "banana cherry"])
        results = await store.query_with_scores("apple", top_k=10)
        scores = [hit.score for hit in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_empty_corpus(self) -> None:
        """query_with_scores returns empty list when no documents are stored."""
        store = InMemoryVectorStore()
        results = await store.query_with_scores("test", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_validation_matches_query(self) -> None:
        """query_with_scores validates top_k identically to query()."""
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            await store.query_with_scores("test", top_k=0)

    @pytest.mark.asyncio
    async def test_legacy_query_contract_unchanged(self) -> None:
        """query() keeps returning bare strings after query_with_scores is added."""
        store = InMemoryVectorStore()
        await store.add_documents(["First document"])
        results = await store.query("First", top_k=1)
        assert results == ["First document"]

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        """close() can be called multiple times safely."""
        store = InMemoryVectorStore()
        await store.add_documents(["test doc"])
        await store.close()
        # Calling close() again should not raise error
        await store.close()
