"""Integration tests: `query_with_scores` on a real `ChromaVectorStore` (Req 3.3, 3.8).

Uses a real Chroma in-memory client and the cached sentence-transformers
embedding model (no network calls once the model is cached locally) - matching
`mise run test:integration`'s "real stores" tier, per the precedent in
`tests/integration/test_store_dry_run_startup.py` for real store construction.
Each test uses a unique collection name to avoid cross-test pollution from
Chroma's shared in-memory client.
"""

import uuid

import pytest

from app.models.rag import RetrievedHit
from app.stores.vector_store import ChromaVectorStore


def _unique_collection_name() -> str:
    """Build a collection name unique to this test invocation."""
    return f"test-query-with-scores-{uuid.uuid4().hex}"


@pytest.mark.asyncio
async def test_query_with_scores_returns_retrieved_hit_instances() -> None:
    """query_with_scores returns RetrievedHit instances with the collection as source."""
    store = ChromaVectorStore(collection_name=_unique_collection_name())
    try:
        await store.add_documents(["Python is a programming language", "Cats are mammals"])
        results = await store.query_with_scores("coding in Python", top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedHit)
        assert results[0].text == "Python is a programming language"
        assert results[0].chunk_id.startswith(f"{store.collection_name}::")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_query_with_scores_chunk_id_format_is_source_ordinal() -> None:
    """chunk_id follows the 'source::NNNN' format with zero-padded ordinal."""
    store = ChromaVectorStore(collection_name=_unique_collection_name())
    try:
        await store.add_documents(["First document"])
        results = await store.query_with_scores("First document", top_k=1)

        assert results[0].chunk_id == f"{store.collection_name}::0000"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_query_with_scores_chunk_id_is_stable_across_repeated_queries() -> None:
    """The same document keeps the same chunk_id across repeated queries."""
    store = ChromaVectorStore(collection_name=_unique_collection_name())
    try:
        await store.add_documents(["Doc one", "Doc two", "Doc three"])

        first_pass = await store.query_with_scores("Doc", top_k=10)
        second_pass = await store.query_with_scores("Doc", top_k=10)

        first_ids = {hit.text: hit.chunk_id for hit in first_pass}
        second_ids = {hit.text: hit.chunk_id for hit in second_pass}
        assert first_ids == second_ids
        assert len(set(first_ids.values())) == 3
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_query_with_scores_returns_empty_list_on_empty_corpus() -> None:
    """query_with_scores returns empty list when no documents are stored."""
    store = ChromaVectorStore(collection_name=_unique_collection_name())
    try:
        results = await store.query_with_scores("test", top_k=5)
        assert results == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_query_with_scores_top_k_validation() -> None:
    """query_with_scores raises ValueError for invalid top_k, matching query()."""
    store = ChromaVectorStore(collection_name=_unique_collection_name())
    try:
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            await store.query_with_scores("test", top_k=0)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_legacy_query_contract_unchanged_after_query_with_scores_added() -> None:
    """query() keeps returning bare strings after query_with_scores is added."""
    store = ChromaVectorStore(collection_name=_unique_collection_name())
    try:
        await store.add_documents(["First document"])
        results = await store.query("First document", top_k=1)
        assert results == ["First document"]
    finally:
        await store.close()
