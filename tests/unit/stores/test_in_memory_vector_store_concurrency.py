"""Concurrency-safety regression tests for InMemoryVectorStore (Req 6, task 11.2).

Boundary correction for task 11.1 (`.sdd/specs/002-review-roadmap-remediation/tasks.md`):
11.1's own boundary is implementation-only (`app/stores/vector_store/in_memory.py`,
no test file), so this module is 11.1's RED step — the same precedent already
recorded for tasks 8.1/8.2 and 10.3/10.4. Both 11.1 and 11.2 close together.

`query()` scores off the event loop via `asyncio.to_thread`; before that offload
lands, nothing in this module's own coroutines ever suspends, so a real OS-level
race between a query's scoring thread and a concurrent `add_documents()` mutation
cannot occur at all pre-fix — the tearing window is only *reachable* once the
offload exists (plan.md L4.4). Rather than relying on GIL-timing luck to
occasionally reproduce a race, `test_query_snapshot_isolated_from_concurrent_ingest`
deterministically controls the interleaving by intercepting the process-wide
`asyncio.to_thread` call: it pauses the query exactly after the point where an
implementation must have captured its scoring snapshot, runs a concurrent
ingest that evicts the snapshotted documents, then resumes scoring. This both
proves the offload exists (`assert not query_task.done()` after a bare yield —
false pre-fix, since unfixed `query()` has no await point and runs to
completion synchronously) and proves the snapshot is isolated from the
concurrent mutation (Req 6.2, 6.3, 6.5, 6.6).
"""

import asyncio

import pytest

from app.stores.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_query_snapshot_isolated_from_concurrent_ingest(monkeypatch) -> None:
    """A query's scoring snapshot is unaffected by a concurrent ingest/eviction.

    Deterministically arranges: query captures its snapshot -> (pause) ->
    concurrent add_documents() evicts every pre-ingest document -> (resume) ->
    query finishes scoring. The result must reflect only the pre-ingest
    snapshot, never a torn mix with the post-ingest documents, and no
    collection-mutated-during-iteration error may be raised.
    """
    store = InMemoryVectorStore(max_documents=3)
    await store.add_documents(["alpha one", "alpha two", "alpha three"])

    proceed = asyncio.Event()
    real_to_thread = asyncio.to_thread

    async def controlled_to_thread(func, /, *args, **kwargs):
        await proceed.wait()
        return await real_to_thread(func, *args, **kwargs)

    # Patched on the `asyncio` module object itself (not the implementation
    # module's namespace) so this works regardless of how the implementation
    # imports `asyncio` -- attribute lookups on a module resolve at call time.
    monkeypatch.setattr(asyncio, "to_thread", controlled_to_thread)

    query_task = asyncio.create_task(store.query("alpha", top_k=10))
    await asyncio.sleep(0)

    # Fails pre-fix: an unfixed query() has no internal await point at all, so
    # by the time control returns here it has already run to completion.
    assert not query_task.done(), (
        "query() completed without ever yielding to the event loop -- "
        "scoring is not offloaded off the event loop (Req 6.1)"
    )

    # Concurrent ingest: exceeds max_documents=3, evicting every document the
    # in-flight query already snapshotted.
    await store.add_documents(["beta four", "beta five", "beta six"])

    proceed.set()
    result = await query_task

    assert result, "expected the pre-ingest snapshot to still be scored"
    assert set(result).issubset({"alpha one", "alpha two", "alpha three"}), (
        f"query result {result!r} leaked post-ingest documents -- torn read"
    )
    assert not (set(result) & {"beta four", "beta five", "beta six"})


@pytest.mark.asyncio
async def test_repeated_interleaved_queries_and_ingests_never_tear_or_error() -> None:
    """Many real-concurrency rounds of interleaved query + ingest stay consistent.

    Covers Req 6.7's regression requirement directly: repeated queries
    interleaved with concurrent ingests must never raise (6.5), must never
    report a torn document set (6.3), and must keep memory accounting
    consistent with what is actually stored (6.4).
    """
    store = InMemoryVectorStore(max_documents=20)
    await store.add_documents([f"seed document {i}" for i in range(20)])

    async def ingest_round(batch: int) -> None:
        await store.add_documents([f"batch{batch} document {i}" for i in range(10)])

    async def query_round() -> list[str]:
        return await store.query("document", top_k=20)

    for round_num in range(25):
        ingest_task = asyncio.create_task(ingest_round(round_num))
        query_tasks = [asyncio.create_task(query_round()) for _ in range(5)]
        results = await asyncio.gather(ingest_task, *query_tasks)

        for hits in results[1:]:
            # Every hit must be a real, currently-or-recently-stored document,
            # never a mismatched/blank/corrupted entry from a torn index read.
            for text in hits:
                assert isinstance(text, str)
                assert "document" in text

    # Memory accounting stays internally consistent with what is stored.
    expected = sum(
        store._estimate_memory(doc, tokens)
        for doc, tokens in zip(store._documents, store._doc_tokens, strict=True)
    )
    assert store.get_memory_usage() == expected
    assert len(store._documents) == len(store._doc_tokens) == len(store._ordinals)


@pytest.mark.asyncio
async def test_sequential_ranking_unchanged_after_concurrent_activity() -> None:
    """Concurrency safety does not alter TF-IDF ranking for sequential use (Req 6.6)."""
    store = InMemoryVectorStore()

    async def ingest(i: int) -> None:
        await store.add_documents([f"noise term {i}"])

    await asyncio.gather(*(ingest(i) for i in range(10)))
    await store.add_documents(
        [
            "python programming language",
            "python snake species",
            "java programming language",
        ]
    )

    results = await store.query("python programming", top_k=3)
    assert results[0] == "python programming language"
    assert "python snake species" in results[1:]
    assert "java programming language" in results[1:]
