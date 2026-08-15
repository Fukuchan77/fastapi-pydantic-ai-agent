"""Regression tests for the per-document TF-IDF vector cache.

`_score_snapshot` re-derived a TF-IDF vector for every stored document on every
query. Only the IDF weights were cached, so each request paid O(corpus) work
regardless of how few terms it searched for - up to 1000 documents of 100,000
characters each under the store's own defaults.

`asyncio.to_thread` already kept that work off the event loop, so the failure
mode was not a stalled loop but sustained CPU burn per request: concurrent
queries saturate the default thread pool and contend on the GIL. Caching the
vectors alongside the IDF weights removes the repeat cost entirely, leaving
scoring proportional to the query's term count.

The cache must stay correct across ingests, so these tests pin both the reuse
and the invalidation.
"""

import asyncio

import pytest

from app.stores.vector_store import InMemoryVectorStore


DOCUMENTS = [
    "Python is a high-level programming language",
    "FastAPI is a modern web framework for building APIs",
    "LlamaIndex Workflows enable event-driven orchestration",
    "Pydantic AI provides type-safe agent definitions",
]


async def _store_with_documents() -> InMemoryVectorStore:
    """Build a store pre-loaded with the shared corpus."""
    store = InMemoryVectorStore()
    await store.add_documents(DOCUMENTS)
    return store


class TestDocumentVectorCache:
    """The derived vectors are built once and reused."""

    async def test_cache_is_empty_before_first_query(self) -> None:
        """Ingesting alone does not build the vectors."""
        store = await _store_with_documents()
        assert store._doc_vectors is None
        assert store._doc_norms is None

    async def test_first_query_populates_cache(self) -> None:
        """The first query derives and publishes the vectors."""
        store = await _store_with_documents()
        await store.query("Python", top_k=2)

        assert store._doc_vectors is not None
        assert store._doc_norms is not None
        assert len(store._doc_vectors) == len(DOCUMENTS)
        assert len(store._doc_norms) == len(DOCUMENTS)

    async def test_second_query_reuses_the_same_objects(self) -> None:
        """A repeat query must not rebuild the vectors."""
        store = await _store_with_documents()
        await store.query("Python", top_k=2)
        first_vectors = store._doc_vectors
        first_norms = store._doc_norms

        await store.query("framework", top_k=2)

        assert store._doc_vectors is first_vectors
        assert store._doc_norms is first_norms

    async def test_add_documents_invalidates_cache(self) -> None:
        """An ingest drops the vectors so the next query rebuilds them."""
        store = await _store_with_documents()
        await store.query("Python", top_k=2)
        assert store._doc_vectors is not None

        await store.add_documents(["Redis is an in-memory data store"])

        assert store._doc_vectors is None
        assert store._doc_norms is None

    async def test_clear_invalidates_cache(self) -> None:
        """Clearing the corpus drops the vectors too."""
        store = await _store_with_documents()
        await store.query("Python", top_k=2)
        assert store._doc_vectors is not None

        await store.clear()

        assert store._doc_vectors is None
        assert store._doc_norms is None

    async def test_cache_rebuilt_after_ingest_reflects_new_corpus(self) -> None:
        """The rebuilt cache covers the documents added since the last query."""
        store = await _store_with_documents()
        await store.query("Python", top_k=2)

        await store.add_documents(["Redis is an in-memory data store"])
        await store.query("Redis", top_k=1)

        assert store._doc_vectors is not None
        assert len(store._doc_vectors) == len(DOCUMENTS) + 1


class TestRankingUnchanged:
    """Caching must not alter which documents rank highest."""

    async def test_relevant_document_ranks_first(self) -> None:
        """Scores still identify the matching document."""
        store = await _store_with_documents()

        results = await store.query("FastAPI web framework", top_k=1)

        assert results == ["FastAPI is a modern web framework for building APIs"]

    async def test_ranking_is_stable_across_repeated_queries(self) -> None:
        """The cached path returns the same ordering as the deriving path."""
        store = await _store_with_documents()

        first = await store.query("event-driven orchestration", top_k=4)
        second = await store.query("event-driven orchestration", top_k=4)

        assert first == second

    async def test_single_document_corpus_still_returns_a_hit(self) -> None:
        """With one document every IDF is 0, but the slice is still returned."""
        store = InMemoryVectorStore()
        await store.add_documents(["only document"])

        assert await store.query("only", top_k=1) == ["only document"]

    async def test_scores_match_uncached_computation(self) -> None:
        """Cached scoring agrees with scoring that derives vectors inline."""
        store = await _store_with_documents()

        cached = await store.query_with_scores("Pydantic type-safe agent", top_k=4)

        fresh = InMemoryVectorStore()
        await fresh.add_documents(DOCUMENTS)
        uncached = await fresh.query_with_scores("Pydantic type-safe agent", top_k=4)

        assert [hit.chunk_id for hit in cached] == [hit.chunk_id for hit in uncached]
        for cached_hit, uncached_hit in zip(cached, uncached, strict=True):
            assert cached_hit.score == pytest.approx(uncached_hit.score)


class TestConcurrentMutationSafety:
    """A corpus change while scoring is in flight must not publish stale vectors."""

    async def test_ingest_during_query_does_not_publish_stale_cache(self) -> None:
        """Vectors describing a superseded generation are discarded."""
        store = await _store_with_documents()

        query_task = asyncio.create_task(store.query("Python", top_k=2))
        ingest_task = asyncio.create_task(store.add_documents(["Redis is an in-memory data store"]))
        await asyncio.gather(query_task, ingest_task)

        # Either the vectors were dropped (ingest landed last) or they match the
        # corpus now in the store - never a stale set of the wrong length.
        if store._doc_vectors is not None:
            assert len(store._doc_vectors) == len(store._documents)
            assert store._doc_norms is not None
            assert len(store._doc_norms) == len(store._documents)
